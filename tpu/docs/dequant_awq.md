# 如何反量化AWQ模型

本文以Qwen2.5VL举例，介绍如何将huggingface上的AWQ模型反量化成浮点模型

## 代码中添加llm模型支持

在`tpu/llm_dequant.py`中参考已有结构添加新模型，其中`block_structure`，可以通过读取模型结构得到。
比如qwen2.5vl的结构如下：
``` python
Qwen2_5_VLForConditionalGeneration(
  (visual): Qwen2_5_VisionTransformerPretrainedModel(
    (patch_embed): Qwen2_5_VisionPatchEmbed(
      (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
    )
    (rotary_pos_emb): Qwen2_5_VisionRotaryEmbedding()
    (blocks): ModuleList(
      (0-31): 32 x Qwen2_5_VLVisionBlock(
        (norm1): Qwen2RMSNorm((1280,), eps=1e-06)
        (norm2): Qwen2RMSNorm((1280,), eps=1e-06)
        (attn): Qwen2_5_VLVisionSdpaAttention(
          (qkv): Linear(in_features=1280, out_features=3840, bias=True)
          (proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (mlp): Qwen2_5_VLMLP(
          (gate_proj): Linear(in_features=1280, out_features=3420, bias=True)
          (up_proj): Linear(in_features=1280, out_features=3420, bias=True)
          (down_proj): Linear(in_features=3420, out_features=1280, bias=True)
          (act_fn): SiLU()
        )
      )
    )
    (merger): Qwen2_5_VLPatchMerger(
      (ln_q): Qwen2RMSNorm((1280,), eps=1e-06)
      (mlp): Sequential(
        (0): Linear(in_features=5120, out_features=5120, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=5120, out_features=2048, bias=True)
      )
    )
  )
  (model): Qwen2_5_VLModel(
    (embed_tokens): Embedding(151936, 2048)
    (layers): ModuleList(
      (0-35): 36 x Qwen2_5_VLDecoderLayer(
        (self_attn): Qwen2_5_VLSdpaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=2048, out_features=256, bias=True)
          (v_proj): Linear(in_features=2048, out_features=256, bias=True)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): Qwen2_5_VLRotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (up_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen2_5_VLRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)

```
其中我们只反量化llm部分，所以`layers_attr`对应的是`model.layers`，`block_structure`对应的是layers下面的结构。

## 下载原始权重和AWQ权重

``` shell
git clone git@hf.co:Qwen/Qwen2.5-VL-3B-Instruct
git clone git@hf.co:Qwen/Qwen2.5-VL-3B-Instruct-AWQ
```

## 执行模型进行反量化

``` shell
cd /workspace/llmc-tpu
python3 tpu/llm_dequant.py --pretrained_model_path /workspace/Qwen2.5-VL-3B-Instruct --quant_model_path /workspace/Qwen2.5-VL-3B-Instruct-AWQ --model_type qwen2.5_vl
```
反量化后的权重存储在`Qwen2.5-VL-3B-Instruct`中，可以验证其与`Qwen2.5-VL-3B-Instruct-AwQ`的结果是否高度一致。
另外注意配置文件还是保持原来的，只需要二进制替换。