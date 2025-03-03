import torch
import argparse
from awq.utils.packing_utils import dequantize_gemm
from awq.modules.linear.gemm import WQLinear_GEMM

from importlib.metadata import version
from typing import Dict, List, Type
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM


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


# register the configuration of different models
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "qwen2.5_vl":
    ModelConfig(
        model_class=Qwen2_5_VLForConditionalGeneration,
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
    ),
    "qwen2.5":
    ModelConfig(
        model_class=AutoModelForCausalLM,
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
    ),
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


def convert_weights(
    pretrained_model: str,
    quant_model: str,
    model_type: str,
    device_map: str = "auto",
    torch_dtype="auto",
):
    model_config = get_model_config(model_type)

    p_model = model_config.model_class.from_pretrained(pretrained_model,
                                                       torch_dtype=torch_dtype,
                                                       device_map=device_map)
    q_model = model_config.model_class.from_pretrained(quant_model,
                                                       torch_dtype=torch_dtype,
                                                       device_map=device_map)

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

    p_model.save_pretrained(pretrained_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--awq_model_path', type=str)
    parser.add_argument("--model_type",
                        type=str,
                        required=True,
                        choices=MODEL_CONFIGS.keys(),
                        help="Model type to process")
    args = parser.parse_args()

    print(f"Torch version: {version('torch')}")
    print(f"Transformers version: {version('transformers')}")

    convert_weights(
        args.pretrained_model_path,
        args.awq_model_path,
        args.model_type,
    )
