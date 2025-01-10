# llmc_tpu

本项目支持了多种仅权重量化(weight_only)量化算法，用以支持大语言模型的量化，其最后会用经过量化算法调整后的权重替换原始输入模型权重，替换之后的权重更适合于TPU-MLIR的RTN group量化，相比于直接经过TPU-MLIR的RTN group量化会有更高的精度。

下面是具体的操作流程。

# llmc-tpu/tpu目录说明
```
.
├── README.md
├── data                                    
│   ├──LLM
│      ├──cali                              #校准数据集
│      ├──eval                              #推理数据集
│   ├──VLM
│      ├──cali
│      ├──eval
├── config
│   ├──LLM                                  #LLM量化config
│      ├── Awq.yml                              #Awq config
│      ├── GPTQ.yml                             #GPTQ config
│   ├──VLM                                  #VLM量化config
│      ├── Awq.yml                              #Awq config
├── config.yml                              #量化参数文件案例
├── llm_quant.py                            #量化主程序
├── run_llmc.sh                             #量化运行脚本
```
----------------------------

#  llmc-tpu量化流程



# 【阶段一】选择校准数据集

## 注意点
* 校准集可以是开源数据集或者业务数据集，如果模型经过下游业务数据集微调，则需要选用业务数据集做校准

### 一：开源数据集下载

如果不指定下游业务数据集，可以使用默认的开源数据集做校准。具体的数据集选取依赖于量化算法，如下所示：

|模型类型| 量化算法   | 校准数据集（开源） |
|:-----:|:---------:|:---------------:|
|  LLM  | Awq       | pileval         |
|  LLM  | GPTQ      | wikitext2       |
|  VLM  | Awq       | MME             |

校准数据集的选取与模型类型和量化算法相关，例如如果量化的是LLM模型，使用的是Awq算法，通常推荐使用pileval数据集。由于这些开源数据集比较大，本文档提供了专门的下载命令，可以下载对应的数据集。具体操作如下：可打开llmc-tpu/tools文件，里面对应有download_calib_dataset.py和download_eval_dataset.py两个python脚本，分别用于下载校准集和测试集。

下面以pileval为例，给出对应的python demo:
```
python3 tools/download_calib_dataset.py --dataset_name pileval --save_path llmc-tpu path/tpu/data/LLM/cali
```
其中save_path要输入上面目录中的LLM/cali目录路径，这主要是为了后续运行量化脚本时方便按照该路径直接调用校准集。

### 二：业务数据集校准

如果模型经过下游业务数据集微调，在选择校准集时，通常应该选择业务数据集。

如果是LLM，将业务数据集放置于上述LLM/cali目录下即可。至于数据集具体的格式，用户可以将一条一条数据文本，写到txt文件里面，每一行代表一条文本数据，使用上述的配置，可以实现自定义数据集的校准。

如果是VLM，将业务数据集放置于上述VLM/cali目录下即可。至于数据集具体的格式，可以参考VLM/cali/general_custom_data中的格式，选择符合需求的格式即可。这里一定需要注意，最后的json文件应该命名为samples.json。

# 【阶段二】选取测试数据集

## 注意点
* 测试数据集主要用来评估当前模型的精度表现，包括预训练（pretrain）模型或者量化（fake_quant）模型的精度

### 一：开源数据集下载

如果不指定下游业务数据集，可以使用默认的开源数据集做测试。具体的数据集选取依赖于量化算法，如下所示：

|模型类型| 量化算法    | 校准数据集(开源)  |测试数据集（开源）  | 
|:-----:|:---------:|:---------------:|:---------------:|
|  LLM  | Awq       | pileval         |     wikitext2   |
|  LLM  | GPTQ      | wikitext2       |     wikitext2   |
|  VLM  | Awq       | MME             |      MME        |

测试数据集的选取与模型类型、量化算法和校准数据集相关，例如如果量化的是LLM模型，使用的是Awq算法，并且使用pileval做校准数据集，这种情况下推荐用ptb做测试数据集。绝大多数情况下校准数据集和测试数据集应该保持一致。

由于这些开源数据集比较大，本文档提供了专门的下载命令，可以下载对应的数据集。具体操作如下：可打开llmc-tpu/tools文件，里面对应有download_eval_dataset.py python脚本，用于下载测试数据集。

下面以ptb为例，给出对应的python demo:
```
python3 tools/download_eval_dataset.py --dataset_name ptb --save_path llmc-tpu path/tpu/data/LLM/eval
```
其中save_path要输入上面目录中的LLM/eval目录路径，这主要是为了后续运行量化脚本时方便按照该路径直接调用测试集。

### 二：业务数据集测试

如果模型经过下游业务数据集校准，在选择测试集时，通常应该选择业务数据集测试。

如果是LLM，将业务数据集放置于上述LLM/eval目录下即可。至于数据集具体的格式，用户可以将一条一条数据文本，写到txt文件里面，每一行代表一条文本数据，使用上述的配置，可以实现自定义数据集的测试。

如果是VLM，将业务数据集放置于上述VLM/eval目录下即可。至于数据集具体的格式，可以参考VLM/cali/general_custom_data中的格式，选择符合需求的格式即可。这里需要注意一定，最后的json文件应该命名为samples.json。

# 【阶段三】配置量化config文件

## 注意点
* 量化config文件包括了量化过程中所需的量化配置，用户可按照需求进行选择，同时为了对齐TPU硬件的配置也会对某些参数做出限制，具体可看下文详细介绍。

### 一：量化config文件介绍

```yaml
base:
    seed: &seed 42
model:
    type: Qwen2 # 设置模型名,可支持Llama,Qwen2,Llava,Gemma2等模型
    path: # 设置模型权重路径
    torch_dtype: auto
calib:
    name: pileval
    download: False
    path: # 设置校准数据集路径
    n_samples: 128
    bs: 1
    seq_len: 512
    preproc: pileval_awq
    seed: *seed
eval:
    eval_pos: [pretrain, transformed, fake_quant]
    name: wikitext2
    download: False
    path: # 设置测试数据集路径
    bs: 1
    seq_len: 2048
quant:
    method: Awq
    weight:
        bit: 4
        symmetric: False
        granularity: per_group
        group_size: 64
    special:
        trans: True
        # The options for "trans_version" include "v1" and "v2".
        # But their results don't differ significantly.
        trans_version: v2
        weight_clip: True
        # For 2-bit quantization, setting "clip_sym: False" will yield better results.
        clip_sym: True
save:
    save_trans: True # 当设置为True，可以保存下调整之后的浮点权重
    save_path: ./save
run:
    task_name: awq_w_only
    task_type: LLM
```
上面是以Awq算法为例构建的一个完整的config文件。为了简便用户操作，用户无需关注全部参数，仅需关注特定的参数即可。

* model。在model类参数中，用户需要指定type和path,后者是模型当前存放的路径；前者对应当前需要被量化的模型的类型，目前可支持的模型类型有：

✅ [Bloom](https://huggingface.co/bigscience/bloom)

✅ [Llama](https://github.com/facebookresearch/llama)

✅ [Starcoder](https://github.com/bigcode-project/starcoder)

✅ [Opt](https://huggingface.co/docs/transformers/model_doc/opt)

✅ [Falcon](https://huggingface.co/docs/transformers/model_doc/falcon)

✅ [InternLM2](https://huggingface.co/internlm)

✅ [Mistral](https://huggingface.co/docs/transformers/model_doc/mistral)

✅ [Qwen](https://github.com/QwenLM/Qwen)

✅ [Qwen2](https://github.com/QwenLM/Qwen2)

✅ [Llava](https://github.com/haotian-liu/LLaVA)

✅ [StableLm](https://github.com/Stability-AI/StableLM)

✅ [Gemma2](https://huggingface.co/docs/transformers/main/en/model_doc/gemma2)

✅ [Phi](https://huggingface.co/microsoft/phi)

✅ [Phi3](https://huggingface.co/microsoft/phi-3)

✅ [MiniCPM](https://github.com/OpenBMB/MiniCPM)

✅ [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)

✅ [DeepseekV2](https://huggingface.co/deepseek-ai/DeepSeek-V2.5)

✅ [Qwen2Moe](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)

✅ [Qwen2VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)

✅ [InternVL2](https://huggingface.co/OpenGVLab/InternVL2-2B)

✅ [ChatGLM] 

用户也可自行添加模型type在`llmc/models/*.py`.

* calib。calib类参数主要指定了和校准集相关的参数，如果用户使用开源数据集校准，仅需根据上文步骤下载指定的开源数据集到指定位置，其余参数可不关心。如果使用业务数据集校准，则需要给出业务数据集路径,填写到上面config文件中calib类下的path。

* eval。eval类参数主要指定了和测试集相关的参数，如果用户使用开源数据集测试，仅需根据上文步骤下载指定的开源数据集到指定位置，其余参数可不关心。如果使用业务数据集做测试，则需要给出业务数据集路径,填写到上面config文件中eval类下的path。eval类参数中的eval_pos会分别指定不同的模型做精度测试，其中pretrain是预训练模型，transformed模型是权重经过调整的浮点模型，fake_quant是量化模型。

* quant。quant类参数主要关注method和weight类参数。method指定了所需的量化方法，上面config选择了Awq算法，该算法也是当前使用最为普遍的量化算法。weight类参数指定了weight量化的相关配置，由于TPU-MLIR仅支持weight only量化，因此这里只需要关注weight量化配置即可。在这些量化参数中，为了对齐TPU-MLIR量化配置，需要限制某些参数选取，具体如下所示：

| bit   | symmetric | granularity                    |   group_size                   | 
|:-----:|:---------:|:------------------------------:|:------------------------------:|
|  4    | False     | per_channel                    |      -1                        |
|  8    | True      | per_channel or per_group       |-1 or 任意（需对齐TPU-MLIR粒度）   |

* save。save_trans该参数表示是否需要保存量化调整之后的浮点权重，经过量化调整之后的权重相比于原始浮点权重更适合于RTN量化。save_path表示保存带有量化调整浮点权重的模型的路径。用户可以将新生成的浮点模型经过TPU-MLIR编译器weight_only RTN量化生成量化模型，最终部署在TPU硬件上。在同等量化配置下经过TPU-MLIR量化，llmc-tpu量化调整的浮点模型相比原始浮点模型，最终产生的量化模型精度更高。

* run。其中task_name可以由用户自行确定，task_type可分为LLM和VLM，依据自身模型类型选择即可。

### 二：config文件配置案例

```yaml
model:
    type: Llama
    path: .cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/dacdfcde31297e34b19ee0e7532f29586d2c17bc
quant:
    method: Awq
    weight:
        bit: 4
        symmetric: False
        granularity: per_group
        group_size: 64
save:
    save_path: ./save_awq_w4a16_llama
run:
    task_name: awq_w_only
    task_type: LLM
```

上述config文件（可见tpu/config.yml）描述了执行量化最基础的配置文件，此时校准数据集和测试数据集均采用默认开源数据集；

如果指定业务数据集为校准集和测试集，可使用如下config：

```yaml
model:
    type: Llama
    path: .cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/dacdfcde31297e34b19ee0e7532f29586d2c17bc
calib:
    path: # 设置校准数据集路径
eval:
    path: # 设置测试数据集路径
quant:
    method: Awq
    weight:
        bit: 4
        symmetric: False
        granularity: per_group
        group_size: 64
save:
    save_path: ./save_awq_w4a16_llama
run:
    task_name: awq_w_only
    task_type: LLM
```

# 【阶段四】执行量化算法

``` 
python3 tpu/llm_quant.py --llmc_tpu_path llmc-tpu path --config_path config path
```
* PS：其中llmc_tpu_path需指定当前llmc-tpu的路径；config_path则表示量化config文件对应的路径

