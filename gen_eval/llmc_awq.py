from typing import Dict, List, Optional, Union
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
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
from llmc.models import *
import torch

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
from opencompass.registry import LOAD_DATASET, MODELS
PromptType = Union[PromptList, str]

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
            'temperature': 0,
            'max_new_tokens': 512,
            'do_sample': False,
        }
        print("generation_kwargs",generation_kwargs)
        self.sampling_kwargs = DEFAULT_GENERATION_KWARGS.copy()
        self.sampling_kwargs.update(generation_kwargs)
        print("max_new_tokens:",self.sampling_kwargs['max_new_tokens'])
        self._load_model(path=path,
                         config_path = self.sampling_kwargs['config_path']
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
                    config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config = EasyDict(self.config)
        self.config.model.path = path
        print('self.config.model.path:',self.config.model.path)
        check_config(self.config)

        self.inference_per_block = self.config.eval.inference_per_block
        if self.inference_per_block:
            self.model = MODEL_REGISTRY[self.config.model.type](self.config)
        else:
            self.model = MODEL_REGISTRY[self.config.model.type](self.config,device_map='cuda:0')

        self.tokenizer = self.model.get_tokenizer()
        self.tokenizer.padding_side = 'left'
        handles = []
        assert self.sampling_kwargs['quant_type'] in ['pretrain','fake_quant'], "only support 'pretrain' and 'fake_quant' mode for llmc model!, please check."
        if self.sampling_kwargs['quant_type'] in ['pretrain','fake_quant']:
            self.config.eval.eval_pos = ['pretrain','fake_quant']
            blockwise_opts = []
            modalities, modality_configs = get_modality(self.config)
            eval_list = get_eval_list(self.model, self.config)
            print(eval_list)
            for modality, modality_config in zip(modalities, modality_configs):
                print('modality:',modality)
                print('modality_config:',modality_config)
                self.model.set_modality(modality)
                blockwise_opt = ALGO_REGISTRY[modality_config.method](
                    self.model,
                    modality_config,
                    input=None,
                    padding_mask=None,
                    config=self.config,
                )
                blockwise_opts.append(blockwise_opt)
            if self.sampling_kwargs['quant_type'] == 'fake_quant':
                deploy_all_modality(blockwise_opts, 'fake_quant')
            else:
                deploy_all_modality(blockwise_opts, 'origin_float')
            if self.inference_per_block:
                # handles = self.register_hooks(self.model)
                self.manager = GroupPipelineManager(self.model, 1)
                self.handles = self.manager.register_hooks()
                self.manager._move_group_sync(0, "cuda:0")
            else:
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
    
import torch
from concurrent.futures import ThreadPoolExecutor

class GroupPipelineManager:
    def __init__(self, model, group_size=4):
        self.model = model
        self.group_size = group_size
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # 初始化分组
        self._prepare_groups()
        self._setup_device_state()

    def _prepare_groups(self):
        all_blocks = self.model.get_blocks()
        if len(all_blocks) % self.group_size != 0:
            raise ValueError(f"Blocks数量{len(all_blocks)}必须能被group_size整除")
        
        self.groups = [
            all_blocks[i*self.group_size : (i+1)*self.group_size]
            for i in range(len(all_blocks) // self.group_size)
        ]
        print(f"Total groups: {len(self.groups)}, blocks per group: {self.group_size}")

    def _setup_device_state(self):
        for block in self.model.get_blocks():
            block.cpu()


        for layer in self.model.get_layers_except_blocks():
            layer.cuda()
        # torch.cuda.empty_cache()

    def register_hooks(self):
        handles = []
        for group_idx, group in enumerate(self.groups):
            first_block = group[0]
            handles.append(first_block.register_forward_pre_hook(
                self._create_group_pre_hook(group_idx))
            )
            
            last_block = group[-1]
            handles.append(last_block.register_forward_hook(
                self._create_group_post_hook(group_idx))
            )
        return handles

    def _create_group_pre_hook(self, group_idx):
        def pre_hook(module, inputs):
            self._move_group_sync(group_idx, "cuda:0")
            
            if group_idx + 1 < len(self.groups):
                self.executor.submit(
                    self._move_group_async, 
                    group_idx + 1, 
                    "cuda:0"
                )
        return pre_hook

    def _create_group_post_hook(self, group_idx):
        def post_hook(module, inputs, outputs):
            self.executor.submit(
                self._move_group_async,
                group_idx,
                "cpu"
            )
        return post_hook

    def _move_group_sync(self, group_idx, device):
        for block in self.groups[group_idx]:
            block.to(device, non_blocking=False)
        torch.cuda.synchronize()

    def _move_group_async(self, group_idx, device):
        torch.cuda.empty_cache()
        for block in self.groups[group_idx]:
            block.to(device, non_blocking=True)

