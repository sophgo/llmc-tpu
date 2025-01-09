import yaml
import argparse
import os
import subprocess
from collections import OrderedDict

def merge_dicts(source, target):
    for key, value in source.items():
        if key in target:
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                merge_dicts(value, target[key])
            else:
                target[key] = value

class OrderedLoader(yaml.SafeLoader):
    pass

class OrderedDumper(yaml.SafeDumper):
    pass

def construct_mapping(loader, node):
    """将 YAML 的映射类型解析为 OrderedDict"""
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))

def represent_ordered_dict(dumper, data):
    """将 OrderedDict 转为 YAML 格式"""
    return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

def validate_quant_config(quant_config):
    """
    验证 quant 配置是否符合要求。
    
    条件：
    1. 如果 bit = 4,则 symmetric 必须为 False,granularity 必须是 'per_group' 或 'per_channel'。
    2. 如果 bit = 8,则 symmetric 必须为 True,granularity 必须是 'per_channel'。
    """
    # 获取配置
    bit = quant_config.get('bit')
    symmetric = quant_config.get('symmetric')
    granularity = quant_config.get('granularity')
    
    # 条件判断
    if bit == 4:
        assert symmetric is False, "If 'bit' is 4, 'symmetric' must be False."
        assert granularity in ['per_group', 'per_channel'], (
            "If 'bit' is 4, 'granularity' must be 'per_group' or 'per_channel'."
        )
    elif bit == 8:
        assert symmetric is True, "If 'bit' is 8, 'symmetric' must be False."
        assert granularity == 'per_channel', (
            "If 'bit' is 8, 'granularity' must be 'per_channel'."
        )
    else:
        raise ValueError(f"Unsupported 'bit' value: {bit}. Only 4 and 8 are supported.")

def add_cali_eval_config(quant_config, args):
    """
    在未给定 calib 和 eval 数据集的情况下给出默认的数据集
    
    条件：
    1. LLM 量化, Awq 默认 calib 采用 pileval,eval 采用 wikitext2;GPTQ 默认 calib 和 eval 都采用 wikitext2
    2. VLM 量化, Awq 默认 eval 采用 MME;calib 数据集符合文档规范即可
    """
    task_type = custom_data['run']['task_type']
    quant_method = custom_data['quant']['method']
    if "calib" not in custom_data:
        custom_data['calib'] = {}
        if task_type == "LLM":
            if quant_method == "Awq":
                custom_data['calib']['path'] = f"{args.llmc_tpu_path}/tpu/data/{task_type}/cali/pileval"
            elif quant_method == "GPTQ":
                custom_data['calib']['path'] = f"{args.llmc_tpu_path}/tpu/data/{task_type}/cali/wikitext2"
        elif task_type == "VLM":
            if quant_method == "Awq":
                custom_data['calib']['path'] = f"{args.llmc_tpu_path}/tpu/data/{task_type}/cali/general_custom_data"
    if "eval" not in custom_data:
        custom_data['eval'] = {}
        if task_type == "LLM":
            if quant_method =="Awq":
                custom_data['eval']['path'] = f"{args.llmc_tpu_path}/tpu/data/{task_type}/eval/ptb"
            elif quant_method == "GPTQ":
                custom_data['eval']['path'] = f"{args.llmc_tpu_path}/tpu/data/{task_type}/eval/wikitext2"
        elif task_type == "VLM":
            if quant_method == "Awq":
                custom_data['eval']['path'] = f"{args.llmc_tpu_path}/tpu/data/{task_type}/eval/MME"
    return custom_data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llmc_tpu_path', help='llmc-tpu path')
    parser.add_argument('--config_path',help='config_path')
    args = parser.parse_args()
    
    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    OrderedDumper.add_representer(OrderedDict, represent_ordered_dict)        
    os.chdir(args.llmc_tpu_path)
    
    with open(args.config_path, "r") as file:
        custom_data = yaml.load(file, Loader=OrderedLoader)
    #验证自定义config文件量化配置的正确性
    validate_quant_config(custom_data['quant']['weight'])
    
    #添加默认的数据集
    custom_data = add_cali_eval_config(custom_data, args)
        
    #打开对应config文件
    quant_method = custom_data['quant']['method']
    task_type = custom_data['run']['task_type']
    quant_path = f"tpu/config/{task_type}/{quant_method}.yml"
    task_name = custom_data['run']['task_name']
    with open(quant_path,"r") as file:
        data = yaml.load(file, Loader=OrderedLoader)
    
    #合并config文件
    merge_dicts(custom_data, data)

    #生成新的config文件并运行量化
    with open("tpu/w_only.yml", "w") as file:
        yaml.dump(data, file, Dumper=OrderedDumper, sort_keys=False)    
    subprocess.run(["bash", "tpu/run_llmc.sh", args.llmc_tpu_path, task_name])
