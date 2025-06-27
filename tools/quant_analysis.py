import argparse
import gc
import json
import os
import sys
import time
import copy

import torch
import torch.distributed as dist
import torch.nn as nn

import yaml
import matplotlib.pyplot as plt
import shutil

from easydict import EasyDict
from loguru import logger
from torch.distributed import destroy_process_group, init_process_group
from easydict import EasyDict as edict
from llmc.compression.quantization import *
from llmc.compression.sparsification import *
from llmc.compression.token_reduction import *
from llmc.data import BaseDataset
from llmc.eval.utils import eval_model, get_eval_list
from llmc.models import *
from llmc.utils import (
    check_config,
    deploy_all_modality,
    get_modality,
    mkdirs,
    print_important_package_version,
    seed_all,
    update_autoawq_quant_config,
    update_vllm_quant_config,
)
from llmc.utils.registry_factory import ALGO_REGISTRY, MODEL_REGISTRY
from llmc.compression.quantization.module_utils import (
    _LLMC_LINEAR_TYPES_,
    _TRANSFORMERS_LINEAR_TYPES_,
)

from tqdm import tqdm
import numpy as np
import functools
import concurrent.futures

# GROUP_SIZE = [128, 64, 32, 16]
GROUP_SIZE = [32, 16]
FLOAT_BIT = ["e2m1", "e4m3"]
INT_BIT = [4, 8]
GRANULARITY = [
    "per_channel",
    "per_group",
]  # exclude "per_tensor" in search, seldom use it
QUANT_TYPE = ["int-quant", "float-quant"]

LLAMA_ATTENTION_OP = ["q_proj", "k_proj", "v_proj", "o_proj"]
LLAMA_MLP_OP = ["up_proj", "down_proj", "gate_proj"]


def easydict_to_dict(ed):
    """Convert EasyDict to dict for clean YAML output."""
    d = {}
    for k, v in ed.items():
        if isinstance(v, edict):
            d[k] = easydict_to_dict(v)
        else:
            d[k] = v
    return d


def calculate_kurtosis_channel(signal):
    """Calculates the kurtosis of a given signal.

    Args:
        signal (torch.Tensor): Input signal, shape (4096, 1024).

    Returns:
        float: The average kurtosis value of the rows.
    """
    signal = signal.float()
    mean = torch.mean(signal, dim=1, keepdim=True)
    std = torch.std(signal, dim=1, keepdim=True)

    std[std == 0] = 1e-8  # Avoid division by zero

    standardized_signal = (signal - mean) / std
    kurtosis = torch.mean(
        standardized_signal**4, dim=1
    )  # Calculate kurtosis for each row

    average_kurtosis = torch.mean(kurtosis)

    return average_kurtosis.item()


def calculate_kurtosis(signal):
    """Calculates the kurtosis of a given signal.

    Args:
        signal (torch.Tensor): Input signal, shape (N, *).

    Returns:
        float: The kurtosis value.
    """
    signal = signal.float()
    signal = signal.view(1, -1)
    mean = torch.mean(signal)
    std = torch.std(signal)

    if std == 0:
        return float("inf")

    standardized_signal = (signal - mean) / (std + 1e-8)

    kurtosis = torch.mean(standardized_signal**4)  # - 3

    return kurtosis.item()


def draw(save_path, save_name, X, Y1, Y2):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, Y1)
    ax.plot(X, Y2)
    plt.xlabel("channel")
    plt.ylabel("value")
    plt.title(save_name)
    fig.savefig(f"{save_path}/{save_name}.jpg")
    plt.close(fig)
    plt.cla()


def analysis_block_cosine(res, t_res, args, is_input=True):
    cosine_sim = nn.CosineSimilarity()

    for name in res:
        oups = res[name]
        t_oups = t_res[name]

        layer_cosine_dict = {}
        for j in range(oups.shape[0]):
            cos = cosine_sim(oups[j].float().view(1, -1), t_oups[j].float().view(1, -1))

            if name not in layer_cosine_dict:
                layer_cosine_dict[name] = []

            layer_cosine_dict[name].append(cos.item())

        for name in layer_cosine_dict:
            cos_values = layer_cosine_dict[name]
            min_cos = min(cos_values)
            avg_cos = sum(cos_values) / len(cos_values)
            if is_input:
                logger.info(f"{name} input min_cos : {min_cos} avg_cos : {avg_cos}")
            else:
                logger.info(f"{name} output min_cos : {min_cos} avg_cos : {avg_cos}")


def analysis_block_mse(res, t_res, args):
    mse_sim = nn.MSELoss()

    for name in res:
        oups = res[name]
        t_oups = t_res[name]

        layer_mse_dict = {}
        for j in range(oups.shape[0]):
            mse = mse_sim(oups[j].float().view(1, -1), t_oups[j].float().view(1, -1))

            if name not in layer_mse_dict:
                layer_mse_dict[name] = []

            layer_mse_dict[name].append(mse.item())

        for name in layer_mse_dict:
            mse_values = layer_mse_dict[name]
            min_mse = min(mse_values)
            avg_mse = sum(mse_values) / len(mse_values)
            logger.info(f"{name} output min_mse : {min_mse} avg_mse : {avg_mse}")


def avg_k_a(a, k):
    result = (a[:, None] * k[None, :]).sum(dim=0)

    total_sum = result.sum()
    print(result.shape)

    average = total_sum / result.numel()
    return average


def analysis_block_outlier(res, t_res, org_w, trans_w, trans_wquantizer, arg):
    if args.prof_gra in ["per_channel", "per_group"]:
        kurt_func = calculate_kurtosis_channel
    else:
        kurt_func = calculate_kurtosis

    for name in res:
        weight = org_w[name]
        t_weight = trans_w[name]

        if args.prof_gra == "per_group":
            weight = trans_wquantizer.reshape_tensor(weight)
            t_weight = trans_wquantizer.reshape_tensor(t_weight)

        # if '_2.' in name and 'down_proj' in name:
        #     import numpy as np
        #     np.save(f'{name}.npy', weight.detach().to(torch.float).cpu().numpy())
        #     np.save(f'{name}_t.npy', t_weight.detach().to(torch.float).cpu().numpy())

        k_w = kurt_func(weight)
        k_t_w = kurt_func(t_weight)

        tensor = res[name].mean(dim=0)
        tensor = tensor.float()

        t_tensor = t_res[name].mean(dim=0)
        t_tensor = t_tensor.float()

        k_a = kurt_func(tensor)
        k_t_a = kurt_func(t_tensor)

        logger.info(
            f"{name} kurtosis org weight: {k_w} trans weight {k_t_w} org act {k_a} trans act {k_t_a}"
        )

        if args.draw:
            save_outlier_path = os.path.join(args.save_path, "outlier")
            save_t_outlier_path = os.path.join(args.save_path, "t_outlier")

            t_min_val = t_tensor.amin(dim=0).detach().cpu().numpy()
            t_max_val = t_tensor.amax(dim=0).detach().cpu().numpy()

            min_val = tensor.amin(dim=0).detach().cpu().numpy()
            max_val = tensor.amax(dim=0).detach().cpu().numpy()

            if not os.path.exists(args.save_path):
                mkdirs(save_outlier_path)
                mkdirs(save_t_outlier_path)

            draw(
                save_path=save_outlier_path,
                save_name=name,
                X=range(tensor.shape[-1]),
                Y1=min_val,
                Y2=max_val,
            )

            draw(
                save_path=save_t_outlier_path,
                save_name=name,
                X=range(t_tensor.shape[-1]),
                Y1=t_min_val,
                Y2=t_max_val,
            )


def analysis_input_kur(res, t_res, args):
    if args.prof_gra in ["per_channel", "per_group"]:
        kurt_func = calculate_kurtosis_channel
    else:
        kurt_func = calculate_kurtosis

    for name in res:
        tensor = res[name].mean(dim=0)
        tensor = tensor.float()

        t_tensor = t_res[name].mean(dim=0)
        t_tensor = t_tensor.float()

        k_a = kurt_func(tensor)
        k_t_a = kurt_func(t_tensor)

        logger.info(f"{name} kurtosis input act {k_a} trans input act {k_t_a}")


def register_hook(block, idx, args):
    hooks = []
    for name, m in block.named_modules():
        if not args.cosine:
            if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(
                            stat_input_hook,
                            w=m.weight.data,
                            name=name,
                            idx=idx,
                            args=args,
                        )
                    )
                )
        else:
            if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(
                            stat_output_hook, name=name, idx=idx, args=args
                        )
                    )
                )

    return hooks


def register_full_hook(block, idx, args):
    hooks = []
    for name, m in block.named_modules():
        if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(
                        stat_full_hook,
                        w=m.weight.data,
                        name=name,
                        idx=idx,
                        args=args,
                    )
                )
            )

    return hooks


def stat_input_hook(m, x, y, w, name, idx, args):
    global t
    global res
    global t_res
    global org_w
    global trans_w
    if isinstance(x, tuple):
        x = x[0]

    layer_name = f"block_{idx}.{name}"

    if args.online_rotate and t:
        if "down_proj" in layer_name:
            x = down_rotater.rotate(x)
        elif "o_proj" in layer_name:
            x = o_rotater.rotate(x)

    if t:
        t_res[layer_name] = x
        trans_w[layer_name] = w
    else:
        res[layer_name] = x
        org_w[layer_name] = w


def stat_output_hook(m, x, y, name, idx, args):
    global t
    global res
    global t_res
    global org_w
    global trans_w
    if isinstance(y, tuple):
        y = y[0]
    layer_name = f"block_{idx}.{name}"
    if t:
        t_res[layer_name] = y
    else:
        res[layer_name] = y


def stat_full_hook(m, x, y, w, name, idx, args):
    global t
    global input_tensors
    global t_input_tensors
    global result_tensors
    global t_result_tensors
    global org_weight
    global trans_weight
    if isinstance(x, tuple):
        x = x[0]

    layer_name = f"block_{idx}.{name}"

    if args.online_rotate and t:
        if "down_proj" in layer_name:
            x = down_rotater.rotate(x)
        elif "o_proj" in layer_name:
            x = o_rotater.rotate(x)

    if t:
        t_input_tensors[layer_name] = x
        t_result_tensors[layer_name] = y
        trans_weight[layer_name] = w
    else:
        try_quant = False
        if try_quant:
            for b in [4, 8]:
                for g in [128, 64, 32, 16]:
                    print(f"now to test {b} {g}")
                    qer = IntegerQuantizer(
                        bit=4,
                        symmetric=False,
                        granularity="per_group",
                        group_size=g,
                        quant_type="int-quant",
                    )
                    q_w = qer.fake_quant_weight_dynamic(w)
                    mse_sim = nn.MSELoss()
                    mse = mse_sim(w.float().view(1, -1), q_w.float().view(1, -1))
                    logger.info(f"{layer_name} {b} {g} weight mse {mse.item()}")
                    del qer
                    del q_w
        input_tensors[layer_name] = x
        result_tensors[layer_name] = y
        org_weight[layer_name] = w


t = True
input_tensors = {}
t_input_tensors = {}
result_tensors = {}
t_result_tensors = {}

org_weight = {}
trans_weight = {}


class block_config:
    def __init__(self):
        self.bits = []
        self.configs = {}
        self.tried = {}
        self.trying_bit = -1
        self.trying_config = -1
        self.bit_index = -1
        self.config_index = -1

    def add_config(self, bit, config):
        if bit in self.bits:
            self.configs[bit].append(config)
            self.tried[bit].append(False)
        else:
            self.bits.append(bit)
            self.configs[bit] = [config]
            self.tried[bit] = [False]

    def step(self):
        found = False
        if self.bit_index == -1:
            self.trying_bit = 0
        tmp_config_idx = self.config_index + 1
        while True:
            while tmp_config_idx < len(self.configs[self.bits[self.trying_bit]]):
                if self.tried[self.bits[self.trying_bit]][tmp_config_idx] == False:
                    self.trying_config = tmp_config_idx
                    return self.configs[self.bits[self.trying_bit]][tmp_config_idx]
                tmp_config_idx = tmp_config_idx + 1
            self.trying_bit = self.trying_bit + 1
            tmp_config_idx = 0
            if self.trying_bit < len(self.bits):
                self.trying_config = 0
                continue
            else:
                return None

    def step_bit(self):
        # assume first bit is the same as default bit
        if self.bit_index == -1 and len(self.bits) > 1:
            self.trying_bit = 1
            self.trying_config = 0
        else:
            tmp_bit = self.bit_index + 1
            if tmp_bit >= len(self.bits):
                return None
            else:
                while tmp_bit < len(self.bits):
                    if self.tried[self.bits[tmp_bit]][0] == False:
                        self.trying_bit = tmp_bit
                        self.trying_config = 0
                        return self.configs[self.bits[self.trying_bit]][0]
                    tmp_bit = tmp_bit + 1
                return None

    def get(self):
        if self.bit_index >= 0 and self.config_index >= 0:
            return self.configs[self.bits[self.bit_index]][self.config_index]
        else:
            return None

    def set(self, good=False):
        # set the reuslt every time get a new try
        if good:
            self.bit_index = self.trying_bit
            self.config_index = self.trying_config
        else:
            self.tried[self.bits[self.trying_bit]][self.trying_config] = True


def opt_model(config, save=True, eval=False):
    model = MODEL_REGISTRY[config.model.type](config)

    seed_all(config.base.seed + int(os.environ["RANK"]))

    logger.info(
        f"optimization model with: {yaml.dump(easydict_to_dict(config), indent=4, sort_keys=False)}"
    )
    blockwise_opts = []
    results = []
    modalities, modality_configs = get_modality(config)
    for modality, modality_config in zip(modalities, modality_configs):
        model.set_modality(modality)
        if not config.get("calib", False):
            print("must provide calibration configuration")
            sys.exit(1)
        else:
            dataset = BaseDataset(
                model.get_tokenizer(), config.calib, model.batch_process
            )
            calib_data, padding_mask = dataset.get_calib_dataset()
            model.collect_first_block_input(calib_data, padding_mask)
            del calib_data
            gc.collect()
            torch.cuda.empty_cache()
            blockwise_opt = ALGO_REGISTRY[modality_config.method](
                model,
                modality_config,
                model.get_first_block_input(),
                model.get_padding_mask(),
                config,
            )
            blockwise_opt.run_block_loop()
            blockwise_opts.append(blockwise_opt)
            dist.barrier()
            save_trans_path = os.path.join(config.save.save_path, "transformed_model")
            if int(os.environ["RANK"]) == 0 and save:
                blockwise_opt.save_model(save_trans_path)
                with open(f"{save_trans_path}/config_opt.yaml", "w") as file:
                    yaml.dump(easydict_to_dict(config), file, sort_keys=False)
                logger.info(f"optimization model end, saved to {save_trans_path}")
            if eval:
                deploy_all_modality(blockwise_opts, "fake_quant")
                eval_list = get_eval_list(model, config)
                for eval_class, config_for_eval in eval_list:
                    result = eval_class.eval(model)
                    eval_name = config_for_eval.eval.type
                    dataset_name = config_for_eval.eval.name
                    results.append(result)
                    logger.info(f"EVAL: {eval_name} on {dataset_name} is {result}")
    return results


def check_init_model_and_config(config):
    t_config = copy.deepcopy(config)
    t_path = t_config.model.get("t_path", None)
    t_save = t_config.save.get("save_trans", False)
    t_save_path = t_config.save.get("save_path", None)
    t_mix = t_config.quant.get("mix_bits", None)
    if t_path is None or not os.path.exists(
        f"{t_save_path}/transformed_model"
    ):  # not initialized with default yet, use algo with default non-mix version
        t_config.save.save_trans = True
        if t_save_path is None:
            t_save_path = "./qa_transformed"
            t_config.save.save_path = t_save_path
            if not os.path.exists(t_save_path):
                shutil.mkdirs(t_save_path)
        if os.path.exists(f"{t_save_path}/transformed_model"):
            shutil.rmtree(f"{t_save_path}/transformed_model")
        t_config.model.t_path = f"{t_save_path}/transformed_model"
        if t_mix is not None:
            print(f"warning!!! mix bit setting cleared for initial opt")
            t_config.quant.mix_bits = None
        opt_model(t_config)
        return True, t_config
    else:
        return False, t_config


"""
    mix_bits:
        setting_0:
            layer_name: [down_proj#0-1-2-3-28-29-30-31]
            do_quant: True
            weight:
                bit: 8
                symmetric: False
                granularity: per_group
                group_size: 128
        setting_1:
            layer_name: [o_proj]
            do_quant: False
"""


def construct_mix_setting(
    layer_name,
    do_quant,
    weight_bit,
    weight_symmetric,
    weight_granularity,
    weight_group_size,
    weight_quant_type,
    weight_calib_algo,
    use_qtorch,
    act_config,
):
    mix_setting = {
        "layer_name": [layer_name],
        "do_quant": do_quant,
        "weight": {
            "bit": weight_bit,
            "symmetric": weight_symmetric,
            "granularity": weight_granularity,
            "group_size": weight_group_size,
            "quant_type": weight_quant_type,
            "calib_algo": weight_calib_algo,
            "use_qtorch": use_qtorch,
        },
    }
    if act_config is not None:
        mix_setting["act"] = act_config
    return mix_setting


def construct_skip_quant_mix_setting(layer_name):
    mix_setting = {"layer_name": [layer_name], "do_quant": False}
    return mix_setting


def construct_skip_o_gate_proj_setting(blks, o_gate="o_proj"):
    l_name = ""
    for b in blks:
        l_name = f"{l_name}-{b}"
    l_name = f"{o_gate}#{l_name[1:]}"
    return construct_skip_quant_mix_setting(l_name)


def try_skip_o_gate(config, target_acc, oprj_or_gate="o_proj"):
    blk_cnt = get_block_count(config)
    t_config = copy.deepcopy(config)
    s_idx = 0
    if t_config.quant.get("mix_bits", None) is None:
        t_config.quant.mix_bits = {}
        s_idx = 0
    else:
        s_idx = len(t_config.quant.mix_bits)
    t_config.quant.mix_bits[f"setting_{s_idx}"] = construct_skip_o_gate_proj_setting(
        range(blk_cnt), oprj_or_gate
    )
    acc = opt_model(t_config, save=False, eval=True)[0]
    if acc < target_acc:
        return True, t_config.quant.mix_bits
    else:
        return False, t_config.quant.mix_bits


def construct_skip_gate_proj_setting(blks):
    l_name = ""
    for b in blks:
        l_name = f"{l_name}-{b}"
    l_name = f"gate_proj#{l_name[1:]}"
    return construct_skip_quant_mix_setting(l_name)


def construct_weight_block(bit, symmetric, granularity, group_size, quant_type):
    if (
        granularity not in GRANULARITY
        or group_size not in GROUP_SIZE
        or quant_type not in QUANT_TYPE
    ):
        logger.error(f"Invalid setting for quantization")
        exit(1)
    if quant_type == "float-quant" and symmetric == False:
        logger.error(f"Don't support asymmetric for float quant")
        exit(1)
    if quant_type == "float-quant" and bit not in FLOAT_BIT:
        logger.error(f"Invalid bit setting for int quant")
        exit(1)
    if quant_type == "int-quant" and bit not in INT_BIT:
        logger.error(f"Invalid bit setting for int quant")
        exit(1)
    if quant_type == "float-quant":
        use_qtorch = True
    else:
        use_qtorch = False
    weight_config = {
        "bit": bit,
        "symmetric": symmetric,
        "granularity": granularity,
        "group_size": group_size,
        "quant_type": quant_type,
        "use_qtorch": use_qtorch,
    }

    return weight_config


def least_granularity_weight_int():
    return construct_weight_block(
        INT_BIT[0], False, GRANULARITY[0], GROUP_SIZE[0], quant_type="int-quant"
    )


def least_granularity_weight_float():
    return construct_weight_block(
        FLOAT_BIT[0], True, GRANULARITY[0], GROUP_SIZE[0], quant_type="float-quant"
    )


def least_granularity_mix_int_block(block_idx):
    layer_name = ""
    for l in LLAMA_ATTENTION_OP:
        layer_name = f"{l}#{block_idx},{layer_name}"
    for l in LLAMA_MLP_OP:
        layer_name = f"{l}#{block_idx},{layer_name}"
    layer_name = layer_name[:-1]
    return construct_mix_setting(
        layer_name,
        True,
        INT_BIT[0],
        False,
        GRANULARITY[0],
        GROUP_SIZE[0],
        "int-quant",
        "learnable",
        False,
        None,
    )


def skip_quant_except(block_num, block_idx):
    skip = {}
    skip_idx = 0
    layer_name = ""
    for l in LLAMA_ATTENTION_OP:
        layer_name = f"{l}#"
        for i in range(block_num):
            if i == block_idx:
                continue
            else:
                layer_name = f"{layer_name}{i}-"
        layer_name = layer_name[:-1]
        skip[f"setting_{skip_idx}"] = construct_skip_quant_mix_setting(layer_name)
        skip_idx = skip_idx + 1
    for l in LLAMA_MLP_OP:
        layer_name = f"{l}#"
        for i in range(block_num):
            if i == block_idx:
                continue
            else:
                layer_name = f"{layer_name}{i}-"
        layer_name = layer_name[:-1]
        skip[f"setting_{skip_idx}"] = construct_skip_quant_mix_setting(layer_name)
        skip_idx = skip_idx + 1
    return skip


def skip_quant_blocks(block_num, block_idxs, skip_idx=0):
    skip = {}
    layer_name = ""
    for l in LLAMA_ATTENTION_OP:
        layer_name = f"{l}#"
        for i in range(block_num):
            if i not in block_idxs:
                continue
            else:
                layer_name = f"{layer_name}{i}-"
        layer_name = layer_name[:-1]
        skip[f"setting_{skip_idx}"] = construct_skip_quant_mix_setting(layer_name)
        skip_idx = skip_idx + 1
    for l in LLAMA_MLP_OP:
        layer_name = f"{l}#"
        for i in range(block_num):
            if i not in block_idxs:
                continue
            else:
                layer_name = f"{layer_name}{i}-"
        layer_name = layer_name[:-1]
        skip[f"setting_{skip_idx}"] = construct_skip_quant_mix_setting(layer_name)
        skip_idx = skip_idx + 1
    return skip


def skip_quant_mlp_blocks(block_num, block_idxs, skip_idx=0):
    skip = {}
    layer_name = ""
    for l in LLAMA_MLP_OP:
        if l == "gate_proj":
            continue
        layer_name = f"{l}#"
        for i in range(block_num):
            if i not in block_idxs:
                continue
            else:
                layer_name = f"{layer_name}{i}-"
        layer_name = layer_name[:-1]
        skip[f"setting_{skip_idx}"] = construct_skip_quant_mix_setting(layer_name)
        skip_idx = skip_idx + 1
    return skip


def weightless_config(config):
    t_config = copy.deepcopy(config)
    t_config.quant.weight = None
    return t_config


def get_block_count(config):
    model = MODEL_REGISTRY[config.model.type](config)
    blocks = 0
    if isinstance(model, Llama) or isinstance(model, Qwen2):
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

        for name, module in model.model.model.named_modules():
            if type(module) == LlamaDecoderLayer or type(module) == Qwen2DecoderLayer:
                blocks = blocks + 1
    else:
        print(f"not Llama qwen, not support")
        sys.exit(1)
    del model
    return blocks


def get_block_sensitivity(config):
    sensitivities = []
    blocks = get_block_count(config)
    for blk_idx in range(blocks):
        t_config = copy.deepcopy(config)
        t_config.weight = least_granularity_weight_int()
        if t_config.quant.get("mix_bits", None) is None:
            t_config.quant.mix_bits = {}
        t_config.quant.mix_bits = skip_quant_except(blocks, blk_idx)
        # t_config.quant.mix_bits = skip_quant_blocks(blocks, [blk_idx])
        print(f"block {blk_idx}: \n")
        print(yaml.dump(easydict_to_dict(t_config), indent=4, sort_keys=False))
        sens = opt_model(t_config, save=False, eval=True)
        gc.collect()
        torch.cuda.empty_cache()
        sensitivities.append(sens[0])
    sort_sens = list(enumerate(sensitivities))
    sort_sens = sorted(sort_sens, key=lambda x: x[1], reverse=True)
    print(f"sensitivities of blocks : {sort_sens}")
    return sort_sens


def construct_search_configs(config, blk_to_try, sorted_sens):
    init_bit = config.quant.weight.bit
    init_weight_dtype = config.quant.weight.quant_type
    # if (isinstance(init_bit,str) and len(init_bit) != 1) or init_weight_dtype == 'float-quant':
    #     logger.error('not support search on float quant yet')
    #     return {}
    init_granularity = config.quant.weight.granularity
    init_groupsize = config.quant.weight.group_size
    configs = {}
    if config.quant.get("act", None) is not None:
        act_config = config.quant.act
    else:
        act_config = None
    if blk_to_try < 0:
        blk_to_try = len(sorted_sens)
    for blk_idx in range(blk_to_try):
        _blk_idx, _ = sorted_sens[blk_idx]
        configs[_blk_idx] = []
        bit_set = [x for x in INT_BIT if x >= init_bit]
        for bit in bit_set:
            if init_granularity == GRANULARITY[0] and bit == init_bit:
                group_size_set = GROUP_SIZE
                granularity = ["per_group"] * len(group_size_set)
            else:
                if bit == init_bit:
                    group_size_set = [x for x in GROUP_SIZE if x < init_groupsize]
                    granularity = ["per_group"] * len(group_size_set)
                else:
                    group_size_set = [-1] + GROUP_SIZE
                    granularity = ["per_channel"] + ["per_group"] * len(GROUP_SIZE)
            layer_names = [x for x in LLAMA_ATTENTION_OP + LLAMA_MLP_OP]
            layer_name_suffix = f"#{_blk_idx}"
            layer_names = [x + layer_name_suffix for x in layer_names]
            for gra, group in zip(granularity, group_size_set):
                block_config = []
                for l in layer_names:
                    _config = construct_mix_setting(
                        l,
                        True,
                        bit,
                        True,
                        gra,
                        group,
                        "float-quant",
                        "minmax",
                        True,
                        act_config,
                    )
                    block_config.append(_config)
                _blk_idx, _ = sorted_sens[blk_idx]
                configs[_blk_idx].append(block_config)
    return configs


def construct_search_configs_v2(config, blk_to_try, sorted_sens, block_configs):
    init_bit = config.quant.weight.bit
    init_weight_dtype = config.quant.weight.quant_type
    # if (isinstance(init_bit,str) and len(init_bit) != 1) or init_weight_dtype == 'float-quant':
    #     logger.error('not support search on float quant yet')
    #     return {}
    init_granularity = config.quant.weight.granularity
    init_groupsize = config.quant.weight.group_size
    # configs = {}
    # config_bits = {}
    if config.quant.get("act", None) is not None:
        act_config = config.quant.act
    else:
        act_config = None
    if blk_to_try < 0:
        blk_to_try = len(sorted_sens)
    for blk_idx in range(blk_to_try):
        _blk_idx, _ = sorted_sens[blk_idx]
        # configs[_blk_idx] = []
        # config_bits[_blk_idx] = []
        _block_config = block_config()
        bit_set = [x for x in INT_BIT if x >= init_bit]
        for bit in bit_set:
            bit_config = []
            if init_granularity == GRANULARITY[0] and bit == init_bit:
                group_size_set = GROUP_SIZE
                granularity = ["per_group"] * len(group_size_set)
            else:
                if bit == init_bit:
                    group_size_set = [x for x in GROUP_SIZE if x < init_groupsize]
                    granularity = ["per_group"] * len(group_size_set)
                else:
                    group_size_set = [-1] + GROUP_SIZE
                    granularity = ["per_channel"] + ["per_group"] * len(GROUP_SIZE)
            layer_names = [x for x in LLAMA_ATTENTION_OP + LLAMA_MLP_OP]
            layer_name_suffix = f"#{_blk_idx}"
            layer_names = [x + layer_name_suffix for x in layer_names]
            for gra, group in zip(granularity, group_size_set):
                layer_config = []
                for l in layer_names:
                    _config = construct_mix_setting(
                        l,
                        True,
                        bit,
                        True,
                        gra,
                        group,
                        "float-quant",
                        "minmax",
                        True,
                        act_config,
                    )
                    layer_config.append(_config)
                _block_config.add_config(bit, layer_config)
                # bit_config.append(block_config)
            if len(_block_config.bits) > 0:
                block_configs[_blk_idx] = _block_config
            # if len(_config) > 0:
            #     configs[_blk_idx].append(bit_config)
            #     config_bits[_blk_idx].append(bit)
    return block_configs


def construct_search_configs_v3(
    config, blk_num, block_configs, except_o=False, except_gate=False, except_qkv=False
):
    init_bit = config.quant.weight.bit
    init_weight_dtype = config.quant.weight.quant_type
    # if (isinstance(init_bit,str) and len(init_bit) != 1) or init_weight_dtype == 'float-quant':
    #     logger.error('not support search on float quant yet')
    #     return {}
    init_granularity = config.quant.weight.granularity
    init_groupsize = config.quant.weight.group_size
    if config.quant.get("act", None) is not None:
        act_config = config.quant.act
    else:
        act_config = None
    for blk_idx in range(blk_num):
        _block_config = block_config()
        if init_weight_dtype == "float-quant":
            bit_set = [FLOAT_BIT[1]]
        else:
            bit_set = [x for x in INT_BIT if x > init_bit]
        for bit in bit_set:
            bit_config = []
            if init_granularity == GRANULARITY[0] and bit == init_bit:
                group_size_set = GROUP_SIZE
                granularity = ["per_group"] * len(group_size_set)
            else:
                if bit == init_bit:
                    group_size_set = [x for x in GROUP_SIZE if x < init_groupsize]
                    granularity = ["per_group"] * len(group_size_set)
                else:
                    group_size_set = [-1] + GROUP_SIZE
                    granularity = ["per_channel"] + ["per_group"] * len(GROUP_SIZE)
            layer_names = [x for x in LLAMA_ATTENTION_OP + LLAMA_MLP_OP]
            if except_o and "o_proj" in layer_names:
                layer_names.remove("o_proj")
            if except_gate and "gate_proj" in layer_names:
                layer_names.remove("gate_proj")
            if except_qkv and "q_proj" in layer_names:
                layer_names.remove("q_proj")
            if except_qkv and "k_proj" in layer_names:
                layer_names.remove("k_proj")
            if except_qkv and "v_proj" in layer_names:
                layer_names.remove("v_proj")
            layer_name_suffix = f"#{blk_idx}"
            layer_names = [x + layer_name_suffix for x in layer_names]
            for gra, group in zip(granularity, group_size_set):
                layer_config = []
                for l in layer_names:
                    if bit == 8:
                        _config = construct_mix_setting(
                            l,
                            True,
                            "e4m3",
                            False,
                            gra,
                            group,
                            "float-quant",
                            "minmax",
                            True,
                            act_config,
                        )
                    else:
                        if init_weight_dtype == "float-quant":
                            _config = construct_mix_setting(
                                l,
                                True,
                                bit,
                                True,
                                gra,
                                group,
                                "float-quant",
                                "minmax",
                                True,
                                act_config,
                            )
                        else:
                            _config = construct_mix_setting(
                                l,
                                True,
                                bit,
                                False,
                                gra,
                                group,
                                "int-quant",
                                "learnable",
                                False,
                                act_config,
                            )
                    layer_config.append(_config)
                _block_config.add_config(bit, layer_config)
            if len(_block_config.bits) > 0:
                block_configs[blk_idx] = _block_config
    return block_configs


def search_configs(config, set_to_try, target_acc):
    blocks = len(set_to_try)
    sets_perblk = len(set_to_try[list(set_to_try.keys())[0]])
    indexs = [-1] * blocks
    cur_idx = 0
    logger.info(
        f"TOTAL {blocks} blocks to opt and try {blocks*sets_perblk} times at most"
    )
    t_config = copy.deepcopy(config)
    blk_idxs = list(set_to_try.keys())
    results = []
    found = False
    while True:
        if t_config.quant.get("mix_bits", None) is not None:
            del t_config.quant.mix_bits
        t_config.quant["mix_bits"] = {}
        indexs[cur_idx] = indexs[cur_idx] + 1
        idx = 0
        for i in range(blocks):
            if indexs[i] >= 0 and indexs[i] < sets_perblk:
                for cfg in set_to_try[blk_idxs[i]][indexs[i]]:
                    tmp_setting = f"setting_{idx}"
                    t_config.quant.mix_bits[tmp_setting] = cfg
                    idx = idx + 1
        if idx == 0:
            print(f"search end")
            break
        else:
            logger.info(
                yaml.dump(easydict_to_dict(t_config), indent=4, sort_keys=False)
            )
            acc = opt_model(t_config, save=False, eval=True)[0]
            results.append({"acc": acc, "config": t_config.quant.mix_bits})
            if acc < target_acc:
                found = True
                break
        cur_idx = cur_idx + 1 if cur_idx + 1 < blocks else 0
    results = sorted(results, key=lambda x: x["acc"])
    return found, results


# v2 is the try the largest loss block every time and check if match the requirements
def search_configs_v2(config, block_configs, pretrained_ppl, target_acc, block_loss):
    blocks = len(block_configs)
    blk_idxs = list(block_configs.keys())
    losses = [(x[0], x[1], False) for x in block_loss if x[0] in block_configs.keys()]
    results = []
    found = False
    last_acc = pretrained_ppl + losses[0][1]
    while True:
        t_config = copy.deepcopy(config)
        if t_config.quant.get("mix_bits", None) is not None:
            del t_config.quant["mix_bits"]
        t_config.quant.mix_bits = {}
        losses = [x for x in losses if not x[2]]
        losses = sorted(losses, key=lambda x: x[1], reverse=True)
        losses_ = [x for x in losses if x[2] == False]
        if len(losses_) == 0:
            print("search end")
            break
        print(f"current losses is {losses_}, now to try block {losses_[0][0]}")
        config_ = block_configs[losses_[0][0]].step()
        if config_ is None:
            losses = [
                (x[0], x[1], x[2]) if x[0] != losses_[0][0] else (x[0], x[1], True)
                for x in losses
            ]
            continue
        idx = 0
        for cfg in config_:
            tmp_setting = f"setting_{idx}"
            t_config.quant.mix_bits[tmp_setting] = cfg
            idx = idx + 1
        for i in range(blocks):
            if blk_idxs[i] == losses_[0][0]:
                continue
            print(
                f"get block configs bit/config {block_configs[blk_idxs[i]].bit_index} {block_configs[blk_idxs[i]].config_index}"
            )
            config_ = block_configs[blk_idxs[i]].get()
            if config_ is None:
                continue
            for cfg in config_:
                tmp_setting = f"setting_{idx}"
                t_config.quant.mix_bits[tmp_setting] = cfg
                idx = idx + 1
        if idx == 0:
            print(f"search end")
            break
        else:
            logger.info(
                yaml.dump(easydict_to_dict(t_config), indent=4, sort_keys=False)
            )
            acc = opt_model(t_config, save=False, eval=True)[0]
            results.append({"acc": acc, "config": t_config.quant.mix_bits})
            loss = acc - pretrained_ppl
            if acc < last_acc:
                losses = [(losses_[0][0], loss, losses_[0][2])] + losses[1:]
                block_configs[losses_[0][0]].set(good=True)
                last_acc = acc
                print(
                    f"Search result, good {acc} with config {yaml.dump(easydict_to_dict(t_config),indent=4,sort_keys=False)}"
                )
            else:
                print(
                    f"Search result, bad {acc} with config {yaml.dump(easydict_to_dict(t_config),indent=4,sort_keys=False)}"
                )
                block_configs[losses_[0][0]].set(good=False)
            if acc < target_acc:
                found = True
                break
    results = sorted(results, key=lambda x: x["acc"])
    return found, results


def objective(param):
    _, blks, block_configs, config = param
    t_config = copy.deepcopy(config)
    if t_config.quant.get("mix_bits", None) is not None:
        skip_idx = len(t_config.quant.mix_bits)
    else:
        skip_idx = 0
        t_config.quant.mix_bits = {}
    for b in blks:
        config_ = block_configs[b].get()
        for cfg in config_:
            tmp_setting = f"setting_{skip_idx}"
            t_config.quant.mix_bits[tmp_setting] = cfg
            skip_idx = skip_idx + 1
    acc = opt_model(t_config, save=False, eval=True)[0]
    print(f"acc {acc} from {blks}")
    return acc, t_config.quant.mix_bits


def objective_float(param):
    _, blks, block_configs, config = param
    t_config = copy.deepcopy(config)
    if t_config.quant.get("mix_bits", None) is not None:
        skip_idx = len(t_config.quant.mix_bits)
        skip = skip_quant_mlp_blocks(len(block_configs), blks, skip_idx)
    else:
        skip_idx = 0
        t_config.quant.mix_bits = {}
        skip = skip_quant_blocks(len(block_configs), blks, skip_idx)
    if t_config.quant.get("mix_bits", None) is None:
        t_config.quant.mix_bits = skip
    else:
        t_config.quant.mix_bits.update(skip)
    acc = opt_model(t_config, save=False, eval=True)[0]
    print(f"acc {acc} from {blks}")
    return acc, t_config.quant.mix_bits


def selection(population, accs, k=3):
    selection_ix = np.random.randint(len(population))
    for ix in np.random.randint(0, len(population), k - 1):
        if accs[ix][0] < accs[selection_ix][0]:
            selection_ix = ix
    return population[selection_ix]


def selection_v2(population, accs, k=2):
    s = sorted([(p, a) for p, a in zip(population, accs)], key=lambda x: x[1][0])
    l = len(s)
    sel = s[: l // k]
    sele = []
    for i in range(k):
        sele = sele + sel
    p = [x[0] for x in sele]
    return p


def selection_v3(population, accs, blks, k=2):
    u = set()
    for p in population:
        u = u.union(set(p))
    u_a = [(e, 0, 0) for e in list(u)]  # all element with zero acc and zero count
    for p in population:
        for e in p:
            for i in range(len(u_a)):
                if u_a[i][0] == e:
                    u_a[i] = (
                        e,
                        u_a[i][1] + 1,
                        u_a[i][2] + accs[population.index(p)][0],
                    )
                    break
    for e in u_a:
        e = (e[0], e[1], e[2] / e[1])
    s = sorted(u_a, key=lambda x: x[2])
    l = len(s)
    if l > blks * k:
        p = [s_[0] for s_ in s[: blks * k]]
    elif l > blks:
        p = [s_[0] for s_ in s]
    else:
        print(f"less blks to select than expected! {l} vs {blks}")
        return []
    pop = []
    for i in range(len(population)):
        np.random.shuffle(p)
        pop.append(p[:blks])
    return pop


def crossover(p1, p2, r_cross=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if np.random.rand() < r_cross:
        use_concat = False
        if use_concat:
            if len(p1) > 2:  # 确保交叉点的选择是有效的
                cnt = 0
                while True:
                    valid = True
                    pt = np.random.randint(1, len(p1) - 1)
                    c1 = p1[:pt] + p2[pt:]
                    c2 = p2[:pt] + p1[pt:]
                    for c_ in c1:
                        if c_ in c2:
                            valid = False
                            c1 = p1.copy()
                            c2 = p2.copy()
                            break
                    for c_ in c2:
                        if c_ in c1:
                            valid = False
                            c1 = p1.copy()
                            c2 = p2.copy()
                            break
                    if valid:
                        break
                    cnt = cnt + 1
                    if cnt > 100:
                        print(f"can not find new crossover ! {p1} {p2}")
                        tmp = set(c1).union(set(c2))
                        len_ = len(c1)
                        c1 = list(tmp)[:len_]
                        c2 = list(tmp)[-len_:]
                        break
        else:
            aggressive = False
            if aggressive:
                l = len(c1)
                c = list(set(c1) & set(c2))
                d = list((set(c1) | set(c2)) - (set(c1) & set(c2)))
                if len(c) < l:
                    np.random.shuffle(d)
                    c1 = c + d[: l - len(c)]
                    np.random.shuffle(d)
                    c2 = c + d[: l - len(c)]
                else:
                    c1 = c
                    c2 = c
            else:
                a = list(set(c1).union(set(c2)))
                l = len(c1)
                np.random.shuffle(a)
                c1 = a[:l]
                np.random.shuffle(a)
                c2 = a[:l]
                print(f"crossover {p1} {p2} to {c1} {c2}")
    return [c1, c2]


def mutation(block_num, blocks, r_mut):
    for i in range(len(blocks)):
        if np.random.rand() < r_mut:
            tmp_blk = np.random.randint(block_num)
            while True:
                if tmp_blk not in blocks:
                    blocks[i] = tmp_blk
                    break
                else:
                    tmp_blk = np.random.randint(block_num)


# v3 is to try genetic algo to search the best config
def search_configs_v3(
    config,
    block_configs,
    pretrained_ppl,
    target_acc,
    block_num,
    try_blocks,
    mix_float=False,
):
    found = False
    block_num = len(block_configs)
    for i in range(block_num):
        # init the configs to use upper bitwidth, per channel
        b = block_configs[i]
        b.step_bit()
        b.set(good=True)
    POP = 16
    BLKS = try_blocks
    CROSS = 0.9
    GEN = 10
    # MUT = 1.0/BLKS
    MUT = (
        0.0004  # about 5% mutation for all population when select 6 blocks and popu 16
    )
    popu = []
    best = pretrained_ppl * 2
    best_cfg = None
    generation = 0
    use_mp = False

    for i in range(POP):
        popu.append(sorted(list(np.random.randint(block_num, size=BLKS))))
    for i in range(POP):
        for j in range(len(popu[i])):
            if popu[i].count(popu[i][j]) > 1:
                tmp_blk = np.random.randint(block_num)
                while True:
                    if tmp_blk not in popu[i]:
                        popu[i][j] = tmp_blk
                        break
                    else:
                        tmp_blk = np.random.randint(block_num)
        popu[i] = sorted(popu[i])

    # import pdb;pdb.set_trace()
    while generation < GEN:
        # import pdb;pdb.set_trace()
        accs = []
        if not use_mp:
            for i in range(POP):
                if mix_float:
                    accs.append((objective_float((i, popu[i], block_configs, config))))
                else:
                    accs.append((objective((i, popu[i], block_configs, config))))
                if accs[-1][0] < target_acc:
                    print(f"early stop, found good acc {accs[-1]} in gen {generation}")
                    break
        else:  # mp, not working ....
            para = []
            for i, p in enumerate(popu):
                para.extend([(i, p, block_configs, config)])
            print(para)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Map the worker function to the data
                accs = executor.map(objective, para)

        for acc in accs:
            if acc[0] < best:
                best = acc[0]
                best_cfg = acc[1]
        if best < target_acc:
            print(f"found good acc {best} < {target_acc} in gen {generation}")
            found = True
            break

        aggressive_sel = False
        aggressive_sel_v2 = False
        if aggressive_sel:
            selected = selection_v2(popu, accs)
        elif aggressive_sel_v2:
            selected = selection_v3(popu, accs, BLKS)
        else:
            selected = [selection(popu, accs) for _ in range(POP)]
        print(f"selected in {generation} gen {selected}")

        children = list()
        dup = list()
        for i in range(0, POP, 2):
            p1, p2 = selected[i], selected[i + 1]
            for c in crossover(p1, p2, CROSS):
                mutation(block_num, c, MUT)
                drop_dup = False
                if drop_dup:
                    dup_ = False
                    for c_ in children:
                        if set(c) == set(c_):
                            dup.append(c)
                            dup_ = True
                            break
                    if not dup_:
                        children.append(c)
                else:
                    children.append(c)
        if len(children) % 2 != 0:
            children.append(dup[0])
        p_s = [set(x) for x in children]
        if len(p_s) <= 2:
            print(
                f"population is too small, only {len(p_s)} unique configs, stop searching"
            )
            break
        while True:
            dup = False
            for p in p_s:
                if p_s.count(p) > 2:
                    p_s.remove(p)
                    dup = True
                    break
            if not dup:
                break
        popu = [list(x) for x in p_s]
        if len(popu) % 2 != 0:
            popu.append(popu[0])
        POP = len(popu)
        generation += 1
        MUT = MUT * 0.5
        a = set()
        for p in popu:
            a = a.union(set(p))
        print(f"after selection, gen {generation} covers {len(a)} blocks, {a}")

    print(f"best acc is {best} cfg is {best_cfg}")
    results = sorted(accs, key=lambda x: x[0])
    results = [{"acc": best, "config": best_cfg}] + accs[1:]
    return found, results


def main(config, args):
    transformed, init_config = check_init_model_and_config(config)
    t_config = copy.deepcopy(init_config)
    if transformed:
        args.entry_eval = True
    # t_save_path = f'{t_save_path}/transformed_model'
    t_config.model.path = t_config.model.t_path
    config.eval.eval_pos = "fake_quant"
    t_config.eval.eval_pos = "transform"

    model = MODEL_REGISTRY[config.model.type](config)
    t_model = MODEL_REGISTRY[config.model.type](t_config)

    eval_list = get_eval_list(model, config)
    t_eval_list = get_eval_list(t_model, t_config)

    logger.info(f"model: {model}")
    logger.info(f"tokenizer: {model.get_tokenizer()}")
    logger.info(f"t_model: {t_model}")
    logger.info(f"t_tokenizer: {t_model.get_tokenizer()}")

    dist.barrier()

    if True:
        blockwise_opts = []
        t_blockwise_opts = []
        modalities, modality_configs = get_modality(config)
        for modality, modality_config in zip(modalities, modality_configs):
            model.set_modality(modality)
            t_model.set_modality(modality)
            eval_list = get_eval_list(model, config)
            # eval_model(model, None, eval_list, eval_pos='pretrain')
            dataset = BaseDataset(
                model.get_tokenizer(), config.calib, model.batch_process
            )
            calib_data, padding_mask = dataset.get_calib_dataset()
            model.collect_first_block_input(calib_data, padding_mask)
            t_model.collect_first_block_input(calib_data, padding_mask)
            del calib_data
            gc.collect()
            torch.cuda.empty_cache()
            blockwise_opt = ALGO_REGISTRY[modality_config.method](
                model,
                modality_config,
                model.get_first_block_input(),
                model.get_padding_mask(),
                config,
            )
            # blockwise_opt.run_block_loop()
            blockwise_opts.append(blockwise_opt)
            t_blockwise_opt = ALGO_REGISTRY[modality_config.method](
                t_model,
                modality_config,
                t_model.get_first_block_input(),
                t_model.get_padding_mask(),
                t_config,
            )
            # blockwise_opt.run_block_loop()
            blockwise_opts.append(blockwise_opt)
            t_blockwise_opts.append(t_blockwise_opt)
            dist.barrier()

            deploy_all_modality(blockwise_opts, "origin_float")
            # deploy_all_modality(blockwise_opts, 'fake_quant')
            for eval_class, config_for_eval in eval_list:
                result = eval_class.eval(model)
                eval_name = config_for_eval.eval.type
                dataset_name = config_for_eval.eval.name
                logger.info(f"ENTRY EVAL: {eval_name} on {dataset_name} is {result}")
                pretrained_ppl = result
            deploy_all_modality(t_blockwise_opts, "fake_quant")
            if args.entry_eval:
                for eval_class, config_for_eval in t_eval_list:
                    result = eval_class.eval(t_model)
                    eval_name = config_for_eval.eval.type
                    dataset_name = config_for_eval.eval.name
                    logger.info(
                        f"ENTRY T_EVAL: {eval_name} on {dataset_name} is {result}"
                    )

            fp_inps = model.get_first_block_input()
            t_fp_inps = t_model.get_first_block_input()

            with torch.no_grad():
                global t
                global input_tensors
                global t_input_tensors
                global result_tensors
                global t_result_tensors
                global org_weight
                global trans_weight
                # for i in tqdm(range(len(model.blocks))):
                for i in range(len(model.blocks)):
                    block = model.blocks[i]
                    t_block = t_model.blocks[i]
                    block.cuda()
                    t_block.cuda()

                    # t_hooks = register_hook(t_block, i, args)
                    t_hooks = register_full_hook(t_block, i, args)
                    t = True
                    t_fp_inps["data"] = t_blockwise_opt.block_forward(t_block)

                    # hooks = register_hook(block, i, args)
                    hooks = register_full_hook(block, i, args)
                    t = False
                    fp_inps["data"] = blockwise_opt.block_forward(block)

                    block.cpu()
                    t_block.cpu()

                    for h in hooks:
                        h.remove()

                    for t_h in t_hooks:
                        t_h.remove()

                    # if args.cosine:
                    #     analysis_block_cosine(res, t_res, args)
                    # else:
                    #     analysis_block_outlier(res, t_res, org_w, trans_w, t_blockwise_opt.wquantizer, args)
                    analysis_block_cosine(
                        input_tensors, t_input_tensors, args, is_input=True
                    )
                    analysis_block_cosine(
                        result_tensors, t_result_tensors, args, is_input=False
                    )
                    analysis_block_mse(result_tensors, t_result_tensors, args)
                    analysis_block_outlier(
                        result_tensors,
                        t_result_tensors,
                        org_weight,
                        trans_weight,
                        t_blockwise_opt.wquantizer,
                        args,
                    )
                    analysis_input_kur(input_tensors, t_input_tensors, args)

                    input_tensors.clear()
                    t_input_tensors.clear()
                    result_tensors.clear()
                    t_result_tensors.clear()
                    org_weight.clear()
                    trans_weight.clear()

                    gc.collect()
                    torch.cuda.empty_cache()

        del model
        del t_model

    logger.info(f"now to search optimization by sensitivity")

    ALGO = ["grid", "topk", "genetic"]
    algo = "genetic"
    target_acc = pretrained_ppl * (1.0 + args.acc_threshold)
    print(f"target ppl is {target_acc}")

    if algo == "topk" or algo == "grid":
        sorted_sens = get_block_sensitivity(t_config)
        sorted_sens = sorted(sorted_sens, key=lambda x: x[1], reverse=True)
        print(f"sorted sens {sorted_sens}")
        # now to search for better accu with the order of decrease granularity for most sensitive blocks
        # alternatively

        # pretrained_ppl = 17.191104888916016 # qwen 0.5B
        # pretrained_ppl = 7.493585586547852 # llama 7b
        block_loss = [(x[0], pretrained_ppl * 2) for x in sorted_sens]
        block_idx, acc = sorted_sens[0]
    elif algo == "genetic":
        if (
            config.quant.weight.get("quant_type", None) is not None
            and config.quant.weight.quant_type == "int-quant"
        ):
            if (
                config.quant.get("act", None) is not None
                and config.quant.act.get("quant_type", None) == "float-quant"
            ):
                if not args.mix_float:
                    args.mix_float = (
                        True  # int 8 quant is hard to implement with f8 act, use float
                    )
        tmp_config = copy.deepcopy(t_config)
        if args.float_oproj:
            print(f"now to try skip o")
            met, c = try_skip_o_gate(tmp_config, target_acc, "o_proj")
            if tmp_config.quant.get("mix_bits", None) is not None:
                del tmp_config.quant["mix_bits"]
            tmp_config.quant.mix_bits = c
            if met:
                print(
                    f"found config meet acc requirement: {yaml.dump(easydict_to_dict(tmp_config),indent=4,sort_keys=False)}"
                )
                return
        if args.float_gateproj:
            print(f"now to try skip gate")
            met, c = try_skip_o_gate(tmp_config, target_acc, "gate_proj")
            tmp_config.quant.mix_bits = c
            if met:
                print(
                    f"found config meet acc requirement: {yaml.dump(easydict_to_dict(tmp_config),indent=4,sort_keys=False)}"
                )
                return
        if args.float_qkv:
            blk_cnt = get_block_count(tmp_config)
            for op in ["q_proj", "k_proj", "v_proj"]:
                s_idx = len(tmp_config.quant.mix_bits)
                tmp_config.quant.mix_bits[f"setting_{s_idx}"] = (
                    construct_skip_o_gate_proj_setting(range(blk_cnt), op)
                )
            acc = opt_model(tmp_config, save=False, eval=True)[0]
            if acc < target_acc:
                print(
                    f"found config meet acc requirement: {yaml.dump(easydict_to_dict(tmp_config),indent=4,sort_keys=False)}"
                )
                return
        if config.quant.get("mix_bits", None) is None:
            config.quant.mix_bits = copy.deepcopy(tmp_config.quant.mix_bits)
        else:
            m_idx = len(config.quant.mix_bits)
            for s in tmp_config.quant.mix_bits:
                config.quant.mix_bits[f"setting_{m_idx}"] = tmp_config.quant.mix_bits[s]
                m_idx = m_idx + 1

    if algo == "topk":
        block_configs = {}
        block_configs = construct_search_configs_v2(
            t_config, args.search_blocks, sorted_sens, block_configs
        )
        print(f"sets to search {block_configs[list(block_configs.keys())[0]].configs}")
    elif algo == "grid":
        configs_to_search = construct_search_configs(
            t_config, args.search_blocks, sorted_sens
        )
        print(f"sets to search {block_configs}")
    elif algo == "genetic":
        block_configs = {}
        block_num = get_block_count(config)
        # not used if mix_float
        block_configs = construct_search_configs_v3(
            t_config,
            block_num,
            block_configs,
            except_o=args.float_oproj,
            except_gate=args.float_gateproj,
            except_qkv=args.float_qkv,
        )
        print(f"sets to search in genetic {block_configs}")
    else:
        print(f"not support this algo yet")
        sys.exit(1)

    if algo == "topk":
        met, search_results = search_configs_v2(
            t_config, block_configs, pretrained_ppl, target_acc, block_loss
        )
        if len(search_results) > 0:
            m_config = copy.deepcopy(config)
            m_config.quant["mix_bits"] = search_results[0]["config"]
            if met:
                print(
                    f"found config meet acc requirement: {yaml.dump(easydict_to_dict(m_config),indent=4,sort_keys=False)}"
                )
            else:
                acc = search_results[0]["acc"]
                print(
                    f"did not find config meet acc requirement, the best {acc}: {yaml.dump(easydict_to_dict(m_config),indent=4,sort_keys=False)}"
                )
        else:
            m_config = None
        print(f"search result: {search_results}")
    elif algo == "grid":
        met, search_results = search_configs(t_config, configs_to_search, target_acc)
        if len(search_results) > 0:
            m_config = copy.deepcopy(config)
            m_config.quant["mix_bits"] = search_results[0]["config"]
            if met:
                print(
                    f"found config meet acc requirement: {yaml.dump(easydict_to_dict(m_config),indent=4,sort_keys=False)}"
                )
            else:
                acc = search_results[0]["acc"]
                print(
                    f"did not find config meet acc requirement, the best {acc}: {yaml.dump(easydict_to_dict(m_config),indent=4,sort_keys=False)}"
                )
        else:
            m_config = None
        print(f"search result: {search_results}")
    elif algo == "genetic":
        # used to use t_config, not good!
        print(
            f"base config before search: {yaml.dump(easydict_to_dict(config),indent=4,sort_keys=False)}"
        )
        met, search_results = search_configs_v3(
            config,
            block_configs,
            pretrained_ppl,
            target_acc,
            block_num,
            args.search_blocks,
            mix_float=args.mix_float,
        )
        if len(search_results) > 0:
            m_config = copy.deepcopy(config)
            m_config.quant["mix_bits"] = search_results[0]["config"]
            if met:
                print(
                    f"found config meet acc requirement: {yaml.dump(easydict_to_dict(m_config),indent=4,sort_keys=False)}"
                )
            else:
                acc = search_results[0]["acc"]
                print(
                    f"did not find config meet acc requirement, the best {acc}: {yaml.dump(easydict_to_dict(m_config),indent=4,sort_keys=False)}"
                )
        else:
            m_config = None
        print(f"search result: {search_results}")
    else:
        print(f"not support this algo yet")
        sys.exit(1)

    dist.barrier()


if __name__ == "__main__":
    logger.add(sys.stdout, level="INFO")
    llmc_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./save")
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--entry_eval", action="store_true")
    parser.add_argument("--log_dir", type=str, default="log.txt")
    parser.add_argument("--prof_gra", type=str, default="per_tensor")
    parser.add_argument(
        "--search_blocks",
        type=int,
        default=6,
        help="find the most serious n blocks to try",
    )
    parser.add_argument(
        "--acc_threshold", type=float, default=0.01, help="percentage of ppl increase"
    )
    parser.add_argument(
        "--mix_float",
        action="store_true",
        help="mix sensitive bloks with float but not 8bit",
    )
    parser.add_argument(
        "--float_oproj",
        action="store_true",
        help="use float for all o_proj ops before further search",
    )
    parser.add_argument(
        "--float_gateproj",
        action="store_true",
        help="use float for all gate_proj ops before further search",
    )
    parser.add_argument(
        "--float_qkv",
        action="store_true",
        help="use float for all qkv_proj ops before further search",
    )

    parser.add_argument("--online_rotate", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)

    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    if int(os.environ["RANK"]) != 0:
        logger.remove()

    check_config(config)

    logger.info(f"args: {args}")
    logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")

    print_important_package_version()

    logger.info(f'WORLD_SIZE : {int(os.environ["WORLD_SIZE"])}')

    seed_all(config.base.seed + int(os.environ["RANK"]))

    # Ensure only the main process creates directories
    if int(os.environ["RANK"]) == 0:
        if "save" in config:
            if config.save.get("save_trans", False):
                save_trans_path = os.path.join(
                    config.save.save_path, "transformed_model"
                )
                mkdirs(save_trans_path)
            if config.save.get("save_trtllm", False):
                save_trtllm_trans_path = os.path.join(
                    config.save.save_path, "trtllm_transformed_model"
                )
                mkdirs(save_trtllm_trans_path)
                save_trtllm_engine_path = os.path.join(
                    config.save.save_path, "trtllm_engine"
                )
                mkdirs(save_trtllm_engine_path)
            if config.save.get("save_vllm", False):
                save_quant_path = os.path.join(
                    config.save.save_path, "vllm_quant_model"
                )
                mkdirs(save_quant_path)
            if config.save.get("save_lightllm", False):
                save_quant_path = os.path.join(
                    config.save.save_path, "lightllm_quant_model"
                )
                mkdirs(save_quant_path)
            if config.save.get("save_sgl", False):
                save_quant_path = os.path.join(config.save.save_path, "sgl_quant_model")
                mkdirs(save_quant_path)
            if config.save.get("save_autoawq", False):
                save_quant_path = os.path.join(
                    config.save.save_path, "autoawq_quant_model"
                )
                mkdirs(save_quant_path)
            if config.save.get("save_mlcllm", False):
                save_quant_path = os.path.join(
                    config.save.save_path, "mlcllm_quant_model"
                )
                mkdirs(save_quant_path)
            if config.save.get("save_fake", False):
                save_fake_path = os.path.join(config.save.save_path, "fake_quant_model")
                mkdirs(save_fake_path)

    # Synchronize all processes after directory creation
    dist.barrier()

    main(config, args)

    destroy_process_group()

    llmc_end_time = time.time()
    llmc_duration_time = llmc_end_time - llmc_start_time
    logger.info(f"llmc_duration_time: {llmc_duration_time} s")
    logger.info("--- llmc finished ---")
