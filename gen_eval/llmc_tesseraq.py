from typing import Dict, List, Optional, Union
import argparse
import json
import os
import sys
import numpy as np
import pdb
import torch
from easydict import EasyDict
from loguru import logger
import yaml
from llmc.utils.registry_factory import ALGO_REGISTRY, MODEL_REGISTRY
from llmc.eval.utils import eval_model, get_eval_list
from llmc.utils import (check_config, deploy_all_modality, get_modality,
                        mkdirs, print_important_package_version, seed_all,
                        update_autoawq_quant_config, update_vllm_quant_config)
from llmc.compression.quantization.module_utils import (_LLMC_LINEAR_TYPES_, _LLMC_LN_TYPES_,
                           _REALQUANT_LINEAR_MAP_, _TRANSFORMERS_LINEAR_TYPES_,
                           _TRANSFORMERS_LN_TYPES_, EffcientFakeQuantLinear,
                           FakeQuantLinear, LlmcActFn, OriginFloatLinear,
                           RotateLinear)
from collections import defaultdict
from llmc.models import *
import torch
from llmc.data import BaseDataset
from transformers.modeling_utils import load_sharded_checkpoint
from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
from opencompass.registry import LOAD_DATASET, MODELS
PromptType = Union[PromptList, str]




def change_models(blockwise_opt):
    blocks = blockwise_opt.blocks

    for i in range(len(blocks)):
        block_idx = i    
        block = blocks[block_idx]
        named_linears = blockwise_opt.model.get_block_linears(block)
        extra_modules = blockwise_opt.model.get_extra_modules(block)
        input_feat_modules = {
            k: v for d in [named_linears, extra_modules] for k, v in d.items()
        }
        input_feat = defaultdict(list)
        handles = blockwise_opt.register_hooks(input_feat_modules, input_feat)
        blockwise_opt.collect_block_qparams(block, input_feat)
        blockwise_opt.model.replace_module_block(
                FakeQuantLinear,
                block,
                i,
                blockwise_opt.get_replacement_params(
                    mode='fake_quant', w_only=False, name=None
                ),
        )
        blockwise_opt.set_non_linear_mode('fake_quant', block, False)
        blockwise_opt.set_rounding_opt_mode(block, on=True)



@MODELS.register_module()
class LLMC(BaseModel):

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        max_batch_size: int = 16,
        tokenizer_only: bool = False,
        tokenizer_path: Optional[str] = None,
        generation_kwargs: dict = dict(),
        meta_template: Optional[Dict] = None,
    ):  # noqa
        
        DEFAULT_GENERATION_KWARGS = {
            'config_path': path + '/quant_config.yml',
            'quant_type' : 'pretrain', 
            'quant_path' : path,
            'temperature': 0,
            'max_new_tokens': 512,
            'do_sample': False,
        }
        print("generation_kwargs",generation_kwargs)
        self.sampling_kwargs = DEFAULT_GENERATION_KWARGS.copy()
        self.sampling_kwargs.update(generation_kwargs)
        print("max_new_tokens:",self.sampling_kwargs['max_new_tokens'])
        self._load_model(path=path,
                         config_dict = self.sampling_kwargs
                        )
        self.max_seq_len = max_seq_len
        self.quant_type = self.sampling_kwargs['quant_type']
        self.template_parser = APITemplateParser(meta_template)
        self.logger = get_logger()
        self._print = True
        self.global_dict = {}
        self.step = 0

    def _load_model(self,
                    path: str,
                    config_dict: dict):

        config_path = config_dict['config_path']   
        quant_model_path = config_dict['quant_path']   

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config = EasyDict(self.config)
        check_config(self.config)
        self.config.model.path = path
  
        self.config.quant.special.iterations = 1
        self.config.quant.special.batch_size = 1
        self.config.quant.special.thresholds = [0.5]

        print('self.config.model.path:',self.config.model.path)
        # 直接以原始模型导入
        self.model = MODEL_REGISTRY[self.config.model.type](self.config,device_map='cuda:0')
        self.tokenizer = self.model.get_tokenizer()

        assert self.sampling_kwargs['quant_type'] in ['pretrain','fake_quant'], "only support 'pretrain' and 'fake_quant' mode for llmc model!, please check."
        if self.sampling_kwargs['quant_type'] == 'fake_quant':
            self.config.eval.eval_pos = ['pretrain','fake_quant']
            blockwise_opts = []
            modalities, modality_configs = get_modality(self.config)
            eval_list = get_eval_list(self.model, self.config)
            print(eval_list)
            for modality, modality_config in zip(modalities, modality_configs):
                print('modality:',modality)
                print('modality_config:',modality_config)
                self.model.set_modality(modality)
                dataset = BaseDataset(
                    self.model.get_tokenizer(), self.config.calib, self.model.batch_process
                )
                # 该算法需要导入数据才能初始化
                calib_data, padding_mask = dataset.get_calib_dataset()
                self.model.collect_first_block_input(calib_data, padding_mask)
                del calib_data
                blockwise_opt = ALGO_REGISTRY[modality_config.method](
                    self.model,
                    modality_config,
                    self.model.get_first_block_input(),
                    self.model.get_padding_mask(),
                    self.config,
                )
                torch.cuda.empty_cache()
                # 需要在原始模型基础上注册buffer
                change_models(blockwise_opt)
                blockwise_opts.append(blockwise_opt)
            # 在注册好buffer的模型结构上直接加载fake_quant权重数据
            load_sharded_checkpoint(blockwise_opt.model.get_model(), quant_model_path)
            blockwise_opt.deploy('fake_quant')

            self.mm_model = self.model.get_model()
            self.mm_model.eval()
            self.mm_model.cuda()



    @torch.no_grad()
    def forward_pre_hook(self, m, x):
        m.cuda()

    @torch.no_grad()
    def forward_hook(self, m, x, y):
        with ThreadPoolExecutor() as executor:
            executor.submit(self.load_layer_to_cpu, m)

    @torch.no_grad()
    def load_layer_to_cpu(self, m):
        m.cpu()

    def register_hooks(self, model):
        handles = []
        for layer in model.get_blocks():
            handles.append(layer.register_forward_pre_hook(self.forward_pre_hook))
        for layer in model.get_blocks():
            handles.append(layer.register_forward_hook(self.forward_hook))
        for layer in model.get_layers_except_blocks():
            layer.cuda()
        return handles

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        self.model.reset_kv()
        torch.cuda.empty_cache()
        inputs = [text.encode('utf-8').decode('utf-8') for text in inputs]
        encoded_inputs = self.tokenizer(
            inputs,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_seq_len
        )
        # print(encoded_inputs)
        input_ids = encoded_inputs['input_ids'].cuda()
        attention_mask = encoded_inputs['attention_mask'].cuda()
        
        if hasattr(self.tokenizer, 'pad_token_id'):
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = self.tokenizer.eos_token_id
        
        generation_tokens = self.model.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.sampling_kwargs['max_new_tokens'],
            temperature=self.sampling_kwargs['temperature'],
            top_p=0.8,
            do_sample=self.sampling_kwargs['do_sample'],
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        generated_texts = []
        for i in range(len(inputs)):
            generated_part = generation_tokens[i][len(input_ids[i]):]
            text = self.tokenizer.decode(generated_part, skip_special_tokens=True).strip()
            generated_texts.append(text)
        self.model.reset_kv()
        return generated_texts


    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, True, True))



# models = [
#     dict(
#         type= LLMC,
#         abbr='LLama_7b',
#         path='/your_path/LLama-7b/',#原始模型路径
#         batch_size=2,
#         generation_kwargs = {
#             'config_path':'/your_path/llmc/configs/quantization/methods/Tesseraq/tesseraq_llama_w8a8.yml',#llmc量化配置
#             'quant_path' : '/your_path/tesseraq_model/llama_7b_w8a8_noamp/fake_quant_model',#llmc量化后的fake节点模型
#             'quant_type' : 'fake_quant',
#             'temperature': 0,
#             'max_new_tokens': 1024,
#             'do_sample': False,
#         },
#         run_cfg=dict(num_gpus=1),
#     )
# ]