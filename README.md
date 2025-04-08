# LLMC-TPU

本项目源自[ModelTC/llmc](https://github.com/ModelTC/llmc)。`ModelTC/llmc`是非常优秀的项目，专为压缩LLM设计，利用最先进的压缩算法提高效率并减少模型体积，同时不影响预测精度。如果要深入了解llmc项目，请转到<https://github.com/ModelTC/llmc>

本项目是基于`ModelTC/llmc`进行一些定制化修改，用于支持Sophgo处理器。

现有已经支持的大模型在[支持列表](./llmc/models/__init__.py)

注意：如果该大模型已经有AWQ版本，则参考[如何反量化AWQ模型](tpu/docs/dequant_awq.md)即可。
如果没有AWQ版本，则按本文步骤进行AWQ量化。

## 环境准备

1) 下载本项目

``` shell
git clone git@github.com:sophgo/llmc-tpu.git
```

2) 准备您需要量化的LLM或者VLM模型，放到`llmc-tpu`的同级目录

比如huggingface上下载`Qwen2-VL-2B-Instruct`，如下：
``` shell
git lfs install
git clone git@hf.co:Qwen/Qwen2-VL-2B-Instruct
```

3) 下载Docker并建立Docker容器


``` shell
# pull docker images
docker pull registry.cn-hangzhou.aliyuncs.com/yongyang/llmcompression:pure-latest

# create container. llmc_test is just a name, and you can set your own name
docker run --privileged --name llmc_test -it --shm-size 64G --gpus all -v $PWD:/workspace  registry.cn-hangzhou.aliyuncs.com/yongyang/llmcompression:pure-latest
```
从这一步之后的所有环境都是在docker容器中。

4) 进入`llmc-tpu`，安装依赖包

``` shell
cd /workspace/llmc-tpu
pip3 install -r requirements.txt
```

5) huggingface token准备 (可选)

这一步是可选的，但是不执行可能会出现`huggingface_hub.errors.LocalTokenNotFoundError`错误。

``` shell
huggingface-cli login
#然后输入huggingface的token，参见https://huggingface.co/settings/tokens
```


## tpu目录说明

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
├── example.yml                             #量化参数参考例子
├── llm_quant.py                            #量化主程序
├── run_llmc.sh                             #量化运行脚本

```

##  操作步骤

### 【阶段一】准备校准数据集和测试数据集


* 注意点1：**校准数据集**可以是开源数据集或者业务数据集，如果模型经过下游业务数据集微调，则需要选用业务数据集做校准
* 注意点2：**测试数据集**主要用来评估当前模型的精度表现，包括预训练（pretrain）模型或者量化（fake_quant）模型的精度

可以选择用开源数据集，也可以选择用业务数据集。

#### 开源数据集
如果有业务数据集最好，没有的话可以用开源数据集，如下：

|模型类型| 量化算法    | 校准数据集(开源)  |测试数据集（开源）  | 
|:-----:|:---------:|:---------------:|:---------------:|
|  LLM  | Awq       | pileval         |     wikitext2   |
|  LLM  | GPTQ      | wikitext2       |     wikitext2   |
|  VLM  | Awq       | MME             |      MME        |

校准数据集的选取与模型类型和量化算法相关，例如如果量化的是LLM模型，使用的是Awq算法，通常推荐使用pileval数据集作为校准集。针对这些开源数据集本文档提供了对应的下载命令，可以运行下载相应的数据集。具体操作如下：可打开llmc-tpu/tools文件，里面对应有download_calib_dataset.py和download_eval_dataset.py两个python脚本，分别用于下载校准集和测试集。

如果是VLM模型，建议使用Awq算法，下载数据集命令如下:

``` shell
cd /workspace/llmc-tpu
# 校准数据集
python3 tools/download_calib_dataset.py --dataset_name MME --save_path tpu/data/VLM/cali
# 测试数据集
python3 tools/download_eval_dataset.py --dataset_name MME --save_path tpu/data/VLM/eval
# 对校准数据集转成校准需要的格式
python3 tpu/mme_extract.py --mme_path tpu/data/VLM/cali/MME --num_samples 128
```

如果是LLM模型，建议用Awq算法，下载数据集命令如下：

``` shell
cd /workspace/llmc-tpu
# 校准数据集
python3 tools/download_calib_dataset.py --dataset_name pileval --save_path tpu/data/LLM/cali
# 测试数据集
python3 tools/download_eval_dataset.py --dataset_name wikitext2 --save_path tpu/data/LLM/eval
```

#### 业务数据集

1) 业务校准数据集

如果模型经过下游业务数据集微调，在选择校准集时，通常应该选择业务数据集。
* 如果是LLM，将业务数据集放置于上述LLM/cali目录下即可。至于数据集具体的格式，可以参考LLM/cali/general_custom_data中的格式，选择符合需求的格式即可。这里一定需要注意，最后的json文件应该命名为samples.json。
* 如果是VLM，将业务数据集放置于上述VLM/cali目录下即可。至于数据集具体的格式，可以参考VLM/cali/general_custom_data中的格式，选择符合需求的格式即可。这里一定需要注意，最后的json文件应该命名为samples.json。

2) 业务测试数据集

如果模型经过下游业务数据集校准，在选择测试集时，通常应该选择业务数据集测试。
* 如果是LLM，将业务数据集放置于上述LLM/eval目录下即可。至于数据集具体的格式，可以参考LLM/cali/general_custom_data中的格式，选择符合需求的格式即可。这里需要注意一定，最后的json文件应该命名为samples.json。
* 如果是VLM，将业务数据集放置于上述VLM/eval目录下即可。至于数据集具体的格式，可以参考VLM/cali/general_custom_data中的格式，选择符合需求的格式即可。这里需要注意一定，最后的json文件应该命名为samples.json。



### 【阶段二】配置量化config文件

注意点：量化config文件包括了量化过程中所需的量化配置，用户可按照需求进行选择，同时为了对齐TPU硬件的配置也会对某些参数做出限制，具体可看下文详细介绍。

#### config文件参数说明

1) Qwen2VL

```yaml
base:
    seed: &seed 42
model:
    type: Qwen2VL # 设置模型名, 具体支持的模型参见llmc/models目录
    path: /workspace/Qwen2-VL-2B-Instruct    # 设置模型权重路径，请改成您需要的模型
    torch_dtype: auto
calib:
    name: custom_mm   # 设置成实际的校准数据集名称，mme，pileval等等
    download: False
    path: /workspace/llmc-tpu/tpu/data/VLM/cali/MME  # 设置校准数据集路径
    n_samples: 128 # 可以根据需要调整
    bs: 1
    seq_len: 512
    preproc: pileval_awq
    seed: *seed
eval:
    eval_pos: [pretrain, fake_quant]
    name: mme  # 设置成实际的测试数据集名称，mme,wikitext2等等
    download: False
    path: /workspace/llmc-tpu/tpu/data/VLM/eval/MME # 设置测试数据集路径
    bs: 1
    seq_len: 2048
quant:
    method: Awq
    quant_objects: [language] # 默认只量化LLM部分，如要量化VIT部分，则设置成[vision, language]
    skip_layers: True # LLM中skip_layer_name是否生效，如果生效则对应的layer不量化，一般对应lm_head
    weight:
        bit: 4 # 设置成想要的量化bit，可以支持4或8
        symmetric: False # 4bit填False；8bit填True
        granularity: per_group # 4bit填per_group；8bit，填per_channel
        group_size: 64 # 4bit填64(与TPU-MLIR对应)；8bit, 填-1
    special:
        trans: True
        trans_version: v2
        weight_clip: True
        clip_sym: True
save:
    save_trans: True       # 当设置为True，可以保存下调整之后的浮点权重
    save_path: /workspace/save_path # 设置保存权重的路径
run:
    task_name: awq_w_only # 配置任务名称
    task_type: VLM   # 设置成VLM或者LLM
```

2) Qwen2.5-0.5B

```yaml
base:
    seed: &seed 42
model:
    type: Qwen2 # 设置模型名, 具体支持的模型参见llmc/models目录
    path: /workspace/Qwen2.5-0.5B    # 设置模型权重路径，请改成您需要的模型
    torch_dtype: auto
calib:
    name: custom_txt   # 设置成实际的校准数据集名称，mme，pileval等等
    download: False
    apply_chat_template: True #调整system prompt 和 user prompt
    path: /workspace/llmc-tpu/tpu/data/LLM/cali/general_custom_data  # 设置校准数据集路径
    n_samples: 128 # 可以根据需要调整
    bs: 1
    seq_len: 512
    preproc: random_truncate_txt
    seed: *seed
eval:
    eval_pos: [pretrain, fake_quant]
    name: custom_ppl  # 设置成实际的测试数据集名称，mme,wikitext2等等
    download: False
    path: /workspace/llmc-tpu/tpu/data/LLM/eval/general_custom_data # 设置测试数据集路径
    bs: 1
    apply_chat_template: True
    seq_len: 2048
    inference_per_block: False
quant:
    method: Awq
    quant_objects: [language] # 默认只量化LLM部分，如要量化VIT部分，则设置成[vision, language]
    weight:
        bit: 4 # 设置成想要的量化bit，可以支持4或8
        symmetric: False # 4bit填False；8bit填True
        granularity: per_group # 4bit填per_group；8bit，填per_channel
        group_size: 64 # 4bit填64(与TPU-MLIR对应)；8bit, 填-1
    special:
        trans: True
        trans_version: v2
        weight_clip: True
        clip_sym: True
save:
    save_trans: True       # 当设置为True，可以保存下调整之后的浮点权重
    save_path: /workspace/save_path # 设置保存权重的路径
run:
    task_name: awq_w_only # 配置任务名称
    task_type: LLM   # 设置成VLM或者LLM
```

上面是以Awq算法为例构建的两个完整的config文件。为了简便用户操作，用户可以将上面直接拷贝到自己的config中，然后对有注解的部分参数进行修改。
下面对重要的一些参数做详细的说明：

| **参数**           | **描述**                                                  |
|:------------------|:--------------------------------------------------------|
| model              | 模型名称，支持的模型在`llmc/models/_init_.py`，可以自行支持新模型`llmc/models/xxxx.py` |
| calib              | calib类参数主要指定校准集相关的参数                       |
| eval               | eval类参数主要指定了和测试集相关的参数                     |
| quant              | 指定量化参数，一般建议用Awq算法，`quant_objects`一般选language，关于weight量化参数参考下表 |

为了与`TPU-MLIR`对齐，weight量化相关参数配置如下：

| bit   | symmetric | granularity                    |   group_size               | 
|:-----|:---------|:------------------------------|:------------------------------|
|  4    | False     | per_channel or per_group       |  -1 or 任意(与TPU-MLIR对齐) |
|  8    | True      | per_channel                    |        -1                  |


### 【阶段三】执行量化算法

``` shell
cd /workspace/llmc-tpu
python3 tpu/llm_quant.py --config_path ./tpu/example.yml
```
* config_path则表示量化config文件对应的路径
* 运行结束后会在`save_path`路径保存权重和日志

## QA

1. 如何支持新模型？

参考[添加新模型](tpu/docs/add_model.md)

2. 如果模型源码不在transformers里面，怎么运行?

如果模型源码不在transformers中，则按照模型源码的安装方法安装。比如Vila源码下载后进入Vila目录，执行如下命令：
``` shell
python -m pip install -e .
```
3. 如何添加prompt template?

在llmc-tpu/llmc/models目录下找到对应的模型类，替换其中的apply_chat_template函数即可。

