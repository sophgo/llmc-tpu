import torch
import argparse
from awq.utils.packing_utils import dequantize_gemm
from awq.modules.linear.gemm import WQLinear_GEMM

from importlib.metadata import version

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# yapf: disable
Qwen2_5 = {
    "block": [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "input_layernorm",
        "post_attention_layernorm"
    ]
}
# yapf: enable


def func(pretrained_model, quant_model):
    model_config = AutoConfig.from_pretrained(pretrained_model, trust_remote_code=True)
    p_model = AutoModelForCausalLM.from_pretrained(pretrained_model,
                                                   torch_dtype="auto",
                                                   device_map="auto")
    q_model = AutoModelForCausalLM.from_pretrained(quant_model,
                                                   torch_dtype="auto",
                                                   device_map="auto")

    block_num = model_config.num_hidden_layers
    for i in range(block_num):
        for layer_name in Qwen2_5["block"]:
            if '.' in layer_name:
                parent_name, child_name = layer_name.split('.', 1)
            else:
                parent_name = None
                child_name = layer_name

            if parent_name:
                q_parent = getattr(q_model.model.layers[i], parent_name)
                p_parent = getattr(p_model.model.layers[i], parent_name)
                q_layer = getattr(q_parent, child_name)
                p_layer = getattr(p_parent, child_name)
            else:
                q_layer = getattr(q_model.model.layers[i], child_name)
                p_layer = getattr(p_model.model.layers[i], child_name)

            if isinstance(q_layer, WQLinear_GEMM):
                iweight = dequantize_gemm(q_layer.qweight, q_layer.qzeros, q_layer.scales,
                                          q_layer.w_bit, q_layer.group_size)
                iweight = iweight.T
                iweight = iweight.to(device=p_layer.weight.device, dtype=p_layer.weight.dtype)
                assert iweight.shape == p_layer.weight.shape, "Shape mismatch!"
                p_layer.weight.data.copy_(iweight)
                print(
                    f"Updated layer {i}.{layer_name}.weight"
                    f"group:{q_layer.group_size}, bit:{q_layer.w_bit}, scale:{q_layer.scales}, zp:{q_layer.qzeros}"
                )

                if hasattr(q_layer, 'bias') and q_layer.bias is not None:
                    q_layer.bias = q_layer.bias.to(device=p_layer.bias.device,
                                                   dtype=p_layer.bias.dtype)
                    assert q_layer.bias.shape == p_layer.bias.shape, "Shape mismatch!"
                    p_layer.bias.data.copy_(q_layer.bias)
                    print(f"Updated layer {i}.{layer_name}.bias")
            else:
                assert q_layer.weight.shape == p_layer.weight.shape, "Shape mismatch!"
                p_layer.weight.data.copy_(q_layer.weight)
                print(f"Updated layer {i}.{layer_name}")
    p_model.save_pretrained(pretrained_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--quant_model_path', type=str)
    args = parser.parse_args()

    print(f"torch : {version('torch')}")
    print(f"transformers : {version('transformers')}")
    print(f"tokenizers : {version('tokenizers')}")
    print(f"huggingface-hub : {version('huggingface-hub')}")
    print(f"datasets : {version('datasets')}")

    func(args.pretrained_model_path, args.quant_model_path)
