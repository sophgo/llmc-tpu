import torch
import argparse
from awq.utils.packing_utils import dequantize_gemm
from awq.modules.linear.gemm import WQLinear_GEMM
from bitsandbytes.nn.modules import Linear4bit
import importlib
from importlib.metadata import version
from typing import Dict, List, Type
import os


class ModelConfig:

    def __init__(
        self,
        model_class: Type,
        layers_attr: str = "layers",
        hidden_layers_attr: str = "num_hidden_layers",
        block_structure: List[str] = None,
    ):
        self.model_class = model_class
        self.layers_attr = layers_attr  # e.g. "model.layers"
        self.hidden_layers_attr = hidden_layers_attr  # e.g. "num_hidden_layers"
        self.block_structure = block_structure

    def get_model_class(self):
        module_path, class_name = self.model_class.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


QWEN2_5_VL_CONFIG = ModelConfig(
    model_class="transformers.Qwen2_5_VLForConditionalGeneration",
    layers_attr="model.layers",
    block_structure=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "input_layernorm",
        "post_attention_layernorm",
    ],
)

QWEN2_CONFIG = ModelConfig(
    model_class="transformers.AutoModelForCausalLM",
    layers_attr="model.layers",
    block_structure=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "input_layernorm",
        "post_attention_layernorm",
    ],
)

MINICPMV_CONFIG = ModelConfig(
    model_class="transformers.AutoModel",
    layers_attr="llm.model.layers",
    block_structure=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "input_layernorm",
        "post_attention_layernorm",
    ],
)

# register the configuration of different models
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "qwen2.5_vl": QWEN2_5_VL_CONFIG,
    "qwen2_vl": QWEN2_5_VL_CONFIG,
    "qwen2.5": QWEN2_CONFIG,
    "qwen2": QWEN2_CONFIG,
    "minicpmv": MINICPMV_CONFIG
}


def get_model_config(model_name: str) -> ModelConfig:
    """support different model configurations"""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    raise ValueError(f"Unsupported model type: {model_name}")


def get_nested_attr(obj, attr_path: str):
    """get nested attribute"""
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_path: str, value):
    """get nested attribute"""
    attrs = attr_path.split(".")
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)


# Global variable to store the contents of backed-up JSON files
_backup_json_contents = {}


def backup_json_file(pretrained_model_dir):
    """
    Backs up all JSON files in the specified directory except for 'model.safetensors.index.json'.

    Args:
        pretrained_model_dir (str): The directory path where the pretrained model is located.
    """
    global _backup_json_contents
    _backup_json_contents = {}

    # Iterate through all files in the directory
    for filename in os.listdir(pretrained_model_dir):
        # Only process JSON files and exclude 'model.safetensors.index.json'
        if filename.endswith('.json') and filename != 'model.safetensors.index.json':
            file_path = os.path.join(pretrained_model_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    _backup_json_contents[filename] = f.read()
            except Exception as e:
                print(f"Unable to read file {file_path}: {e}")


def restore_json_file(pretrained_model_dir):
    """
    Restores the previously backed-up JSON file contents to the specified directory.

    Args:
        pretrained_model_dir (str): The directory path where the pretrained model is located.
    """
    global _backup_json_contents

    # Write the backed-up contents back to the files
    for filename, content in _backup_json_contents.items():
        file_path = os.path.join(pretrained_model_dir, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Unable to write to file {file_path}: {e}")

    # Clear the backup contents
    _backup_json_contents = {}


def convert_weights(
    pretrained_model: str,
    quant_model: str,
    model_type: str,
    device_map: str = "cuda",
    torch_dtype="auto",
    use_cpu: bool = False,
):
    model_config = get_model_config(model_type)
    model_class = model_config.get_model_class()
    p_model_map = "cpu" if use_cpu else "cuda"
    print(f"Loading pretrained model from {pretrained_model}")
    p_model = model_class.from_pretrained(pretrained_model,
                                          torch_dtype=torch_dtype,
                                          device_map=p_model_map,
                                          trust_remote_code=True)
    print(f"Loading quantized model from {quant_model}")
    q_model = model_class.from_pretrained(quant_model,
                                          torch_dtype=torch_dtype,
                                          device_map=device_map,
                                          trust_remote_code=True)

    num_layers = getattr(p_model.config, model_config.hidden_layers_attr)
    assert num_layers == getattr(q_model.config, model_config.hidden_layers_attr)

    p_layers = get_nested_attr(p_model, model_config.layers_attr)
    q_layers = get_nested_attr(q_model, model_config.layers_attr)

    for layer_idx in range(num_layers):
        p_layer = p_layers[layer_idx]
        q_layer = q_layers[layer_idx]

        for layer_path in model_config.block_structure:
            try:
                q_module = get_nested_attr(q_layer, layer_path)
                p_module = get_nested_attr(p_layer, layer_path)
            except AttributeError:
                continue

            if isinstance(q_module, WQLinear_GEMM):
                iweight = dequantize_gemm(
                    q_module.qweight,
                    q_module.qzeros,
                    q_module.scales,
                    q_module.w_bit,
                    q_module.group_size,
                )
                iweight = iweight.T.to(device=p_module.weight.device, dtype=p_module.weight.dtype)
                if iweight.shape != p_module.weight.shape:
                    raise RuntimeError(
                        f"layer[{layer_idx}].{layer_path} shape mismatch: {iweight.shape} != {p_module.weight.shape}"
                    )

                p_module.weight.data.copy_(iweight)
                print(
                    f"Updated layer {layer_idx}.{layer_path}.weight, group: {q_module.group_size}, w_bit: {q_module.w_bit}"
                )

                if hasattr(q_module, 'bias') and q_module.bias is not None:
                    p_module.bias.data.copy_(
                        q_module.bias.to(device=p_module.bias.device, dtype=p_module.bias.dtype))
                    print(f"Updated layer {layer_idx}.{layer_path}.bias")
            elif isinstance(q_module, Linear4bit):
                raise NotImplementedError("Linear4bit is not supported yet")
            else:
                # copy the weights of the non-quantized layers
                if p_module.weight.shape != q_module.weight.shape:
                    raise RuntimeError(
                        f"layer[{layer_idx}].{layer_path} shape mismatch: {q_module.weight.shape} != {p_module.weight.shape}"
                    )
                p_module.weight.data.copy_(q_module.weight)
                print(f"Updated layer {layer_idx}.{layer_path}.weight")
                if hasattr(q_module, 'bias') and q_module.bias is not None:
                    p_module.bias.data.copy_(
                        q_module.bias.to(device=p_module.bias.device, dtype=p_module.bias.dtype))
                    print(f"Updated layer {layer_idx}.{layer_path}.bias")
    backup_json_file(pretrained_model)
    p_model.save_pretrained(pretrained_model)
    restore_json_file(pretrained_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--quant_model_path', type=str, required=True)
    parser.add_argument("--model_type",
                        type=str,
                        required=True,
                        choices=MODEL_CONFIGS.keys(),
                        help="Model type to process")
    parser.add_argument('--use_cpu',
                        action='store_true',
                        default=False,
                        help="Use CPU to load the pretrained model")
    args = parser.parse_args()

    print(f"Torch version: {version('torch')}")
    print(f"Transformers version: {version('transformers')}")

    convert_weights(args.pretrained_model_path,
                    args.quant_model_path,
                    args.model_type,
                    use_cpu=args.use_cpu)
