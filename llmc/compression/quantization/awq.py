import gc
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import (_LLMC_LINEAR_TYPES_, _LLMC_LN_TYPES_,
                           _TRANSFORMERS_LINEAR_TYPES_,
                           _TRANSFORMERS_LN_TYPES_, FakeQuantLinear)
from .utils import check_do_quant, check_w_only, get_aquantizer, get_wquantizer


@ALGO_REGISTRY
class Awq(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        special_config = self.quant_config.get('special', {})
        self.trans = special_config.get('trans', True)
        self.trans_version = special_config.get('trans_version', 'v2')
        self.save_scale = special_config.get('save_scale', False)

    @torch.no_grad()
    def get_weight_scale(self, layers_dict):
        layers = list(layers_dict.values())
        weights = self.collect_layers_weights(layers)
        weights = torch.cat(weights, dim=0)
        org_shape = weights.shape
        wquantizer = get_wquantizer(
            self.block_idx,
            list(layers_dict.keys())[0],
            self.mix_bits_map,
            self.quantizer_mix_bits,
            self.wquantizer,
        )
        weights = wquantizer.reshape_tensor(weights)
        scale = weights.abs() / weights.abs().amax(dim=1, keepdim=True)
        try:
            scale = scale.view(org_shape)
        except RuntimeError:
            scale = wquantizer.restore_tensor(scale, org_shape)
        scale = scale.mean(0)
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        return scale

    @torch.no_grad()
    def get_act_scale(self, x):
        return x.abs().view(-1, x.shape[-1]).mean(0)

    @torch.no_grad()
    def get_original_out(self, x, inspect_module, subset_kwargs):
        with torch.no_grad():
            org_out = inspect_module(x, **subset_kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]
        return org_out

    @torch.no_grad()
    def search_scale_subset(self, layers_dict, input, inspect_module, subset_kwargs):
        w_max = self.get_weight_scale(layers_dict)
        # grid search for ratio
        best_error = float('inf')
        best_scales = None
        n_grid = 20
        org_sd = {k: v.cpu() for k, v in inspect_module.state_dict().items()}

        org_out_dict = {}
        for n in range(n_grid):
            loss_mean = 0
            scales_mean = 0
            for i in range(len(input)):
                input[i] = input[i].to(next(inspect_module.parameters()).device)
                x = input[i]
                if isinstance(subset_kwargs, list):
                    kwargs = subset_kwargs[i]
                else:
                    kwargs = subset_kwargs
                if i in org_out_dict:
                    org_out = org_out_dict[i]
                else:
                    org_out = self.get_original_out(x, inspect_module, kwargs)
                    org_out_dict[i] = org_out
                x_max = self.get_act_scale(x)

                ratio = n * 1 / n_grid
                if self.trans_version == 'v1':
                    scales = (
                        (x_max.pow(ratio) / w_max.pow(1 - ratio))
                        .clamp(min=1e-4)
                        .view(-1)
                    )
                elif self.trans_version == 'v2':
                    scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                for layer_name in layers_dict:
                    fc = layers_dict[layer_name]
                    fc.weight.mul_(scales.view(1, -1))

                    fc.weight.data = get_wquantizer(
                        self.block_idx,
                        layer_name,
                        self.mix_bits_map,
                        self.quantizer_mix_bits,
                        self.wquantizer,
                    ).fake_quant_weight_dynamic(fc.weight.data)

                x_tmp = x / scales.view(1, -1)
                if not check_w_only(
                    self.block_idx,
                    list(layers_dict.keys())[0],
                    self.mix_bits_map,
                    self.quantizer_mix_bits,
                    self.w_only,
                ):
                    x_tmp = get_aquantizer(
                        self.block_idx,
                        list(layers_dict.keys())[0],
                        self.mix_bits_map,
                        self.quantizer_mix_bits,
                        self.aquantizer,
                    ).fake_quant_act_dynamic(x_tmp)
                out = inspect_module(x_tmp, **kwargs)

                if isinstance(out, tuple):
                    out = out[0]

                if self.padding_mask and org_out.shape[1] == self.padding_mask[i].shape[-1]:
                    org_out = org_out * self.padding_mask[i].unsqueeze(dim=-1).to(org_out.device)  # noqa
                    out = out * self.padding_mask[i].unsqueeze(dim=-1).to(out.device)

                loss = (org_out - out).float().pow(2).mean().item()

                if len(input) == 1:
                    n_samples = x.shape[0]
                else:
                    n_samples = self.n_samples

                loss_mean += x.shape[0] * 1.0 / n_samples * loss
                scales_mean += x.shape[0] * 1.0 / n_samples * scales
                inspect_module.load_state_dict(org_sd)
                is_best = loss_mean < best_error
                if is_best:
                    best_error = loss_mean
                    best_scales = scales_mean

        # Synchronize across ranks
        best_error_tensor = torch.tensor([best_error], device='cuda')
        dist.all_reduce(best_error_tensor, op=dist.ReduceOp.MIN)
        global_best_error = best_error_tensor.item()

        # Identify the rank with the minimum loss
        global_best_rank = torch.tensor([dist.get_rank()
                                        if abs(best_error - global_best_error) < 1e-5
                                        else -1],
                                        device='cuda')
        dist.all_reduce(global_best_rank, op=dist.ReduceOp.MAX)
        global_best_rank = global_best_rank.item()

        # Broadcast the best scales from the rank with the minimum loss to all ranks
        if dist.get_rank() == global_best_rank:
            dist.broadcast(best_scales, src=global_best_rank)
        else:
            best_scales = torch.zeros_like(best_scales, device='cuda')
            dist.broadcast(best_scales, src=global_best_rank)

        del org_out_dict
        gc.collect()
        torch.cuda.empty_cache()
        return best_scales

    @torch.no_grad()
    def block_transform(self, block, input_feat, block_kwargs):
        if self.trans:
            super().block_transform(block, input_feat, block_kwargs)

        if self.weight_clip:
            logger.info('auto_clip start')
            logger.info(f'clip version: {self.clip_version}')
            self.auto_clipper.run(
                block,
                self.block_idx,
                input_feat,
                n_sample_token=self.config.calib.get('seq_len', None)
            )
            logger.info('auto_clip finished')
        else:
            logger.info('disable weight clip')

    @torch.no_grad()
    def subset_transform(
        self,
        subset,
        input_feat,
        subset_kwargs,
    ):
        layers_dict = subset['layers']
        prev_op = subset['prev_op']
        input_name = subset['input'][0]
        inspect_module = subset['inspect']
        do_trans = subset.get('do_trans', True)
        if not do_trans:
            logger.info('do_trans is set to False. Do not transform this subset.')
            return

        if not check_do_quant(
            self.block_idx,
            list(layers_dict.keys())[0],
            self.mix_bits_map,
            self.quantizer_mix_bits,
        ):
            logger.info(
                'This subset is set to float. No need to transform this subset.'
            )
            return
        if self.config['model']['type'] == 'Starcoder':
            if isinstance(prev_op[0], (nn.Linear, FakeQuantLinear)):
                logger.info('Do not transform this subset.')
                return

        assert (
            len(prev_op) in (0, 1)
        ), 'Only support single prev_op. If multi prev_ops, code need to be updated.'

        if len(prev_op) == 0 or (len(prev_op) == 1 and prev_op[0] is None):
            logger.info('Cannot apply scale. Do not transform this subset.')
            return

        if isinstance(
            prev_op[0],
            tuple(
                _LLMC_LN_TYPES_ +
                _TRANSFORMERS_LN_TYPES_ +
                _LLMC_LINEAR_TYPES_ +
                _TRANSFORMERS_LINEAR_TYPES_
            ),
        ):
            layers = list(layers_dict.values())

            if (
                isinstance(prev_op[0], (nn.Linear, FakeQuantLinear))
                and prev_op[0].out_features != layers[0].in_features * 3
                and prev_op[0].out_features != layers[0].in_features * 2
                and prev_op[0].out_features != layers[0].in_features
            ):
                logger.info('Cannot apply scale. Do not transform this subset.')
                return

            scale = self.search_scale_subset(
                layers_dict, input_feat[input_name], inspect_module, subset_kwargs
            )

            self.apply_scale(scale, prev_op, layers)
            self.update_input_feat(scale, input_feat, layers_dict)

            if self.save_scale:
                for n in layers_dict:
                    layer_name = f'{self.model.block_name_prefix}.{self.block_idx}.{n}'
                    self.act_scales[layer_name] = scale
        else:
            logger.info('Do not transform this subset.')
