# 如何添加新模型

本文以Vila为例，介绍如何添加新模型。

## 第一步，下载模型源码和权重

此处假定下载到了与`llmc-tpu`同级目录

```shell
# weight from huggingface
git lfs install
git clone git@hf.co:Efficient-Large-Model/VILA1.5-3b

# code from github
git clone git@github.com:NVlabs/VILA.git
git reset --hard ec7fb2c264920bf004fd9fa37f1ec36ea0942db5
```

## 第二步，安装模型

如果模型已经在transformers中，则这一步可以跳过。

此步骤假定已经在docker容器中，VILA在`workspace/VILA`目录

``` shell
# 根据情况安装模型依赖，此处是vila的依赖
python -m pip install flash-attn --no-build-isolation
python -m pip install deepspeed

# 安装vila
cd /workspace/VILA
python -m pip install -e .
```

## 第三步，源码修改

1. `llmc/models/_init_.py`中增加模型定义

``` python
from .vila import Vila
```

2. 实现`llmc/models/vila.py`

实现两个类，`class Vila(BaseModel)`与`class VilaEval(lmms)`。

以下详细介绍两个类如何实现

### class Vila(BaseModel)

这个类继承自BaseModel(`llmc/models/base_model.py`)，大多数方法继承它即可。
有如下这些方法是需要专门实现的。下面一一介绍。

#### build_model()

这个方法是构建和加载模型，如何做可以在大模型的测试脚本中找到相关代码。
比如Vila用`load_pretrained_model`构建。
构建后这几个变量一定要赋值：
* self.mm_model: 代表整个VLM模型对象
* self.vlm_model: 代表整个VLM模型对象，等于mm_model
* self.eval_name: Eval类的名称，这里对应的是`VilaEval`
* self.vision_model: 代表VIT模型对象
* self.mm_projector: 代表projector对象，vit的输出结果会进入`mm_projector`
* self.model: 代表LLM模型对象
* self.model_config: 代表LLM模型的配置，这个必须是对象属性配置，而不是dict属性配置。可以`types.SimpleNamespace`方法对dict类型转换。

#### batch_process()

这个方法用于构建VLM模型输入，支持多batch构建。
建议如果模型支持`AutoProcessor.from_pretrained()`构建输入，则用它构建多batch；否则构建单batch即可。
它涉及到对文本的预处理和图片的预处理，可以在大模型的测试脚本中找到相关代码。
它返回的输出，用于VLM模型的generate的输入，所以要满足generate对输入的需要。
比如Vila的推理接口使用如下：
``` python
output_ids = model.generate(
    input_ids,
    images=images_input,
    do_sample=True if temperature > 0 else False,
    temperature=temperature,
    top_p=top_p,
    num_beams=num_beams,
    max_new_tokens=max_tokens,
    use_cache=use_cache,
    stopping_criteria=[stopping_criteria],
)
```
参考它`batch_process`返回结果如下(其他参数用默认值)：
``` python
inputs = {
    "input_ids": input_ids,
    "images": images,
    "attention_mask": attention_masks,
    "stopping_criteria": [stopping_criteria]
}
return inputs
```

#### get_extra_rot_module_besides_embed_layers()

这个用于获取`mm_projector`中最后一层，可以把self.mm_projector打印出来，看看对应哪一个。
比如Vila的`mm_projector`结构如下：
``` python
(mm_projector): MultimodalProjector(
(layers): Sequential(
    (0): DownSampleBlock()
    (1): LayerNorm((4608,), eps=1e-05, elementwise_affine=True)
    (2): Linear(in_features=4608, out_features=2560, bias=True)
    (3): GELU(approximate='none')
    (4): Linear(in_features=2560, out_features=2560, bias=True)
)
)
```

所以该接口实现如下：
``` python
def get_extra_rot_module_besides_embed_layers(self):
    return [self.mm_projector.layers[-1]]
```

#### skip_layer_name()

一般`return ['lm_head']`即可，在配置文件中由`skip_layers`决定是否生效

#### 其他方法

其他方法可以看LLM结构与哪个LLM近似，直接从`llmc/models/xxx.py`中拷贝过来即可

### class VilaEval(lmms)

这个类继承自[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)项目的[lmms](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/api/model.py)，用于做测试。

这里以Vila举例如何实现这个类，下面对类中的方法一一介绍。

#### __init__()

这个方法入参主要有以下几个，其他参数并不重要：
* llmc_model，它代表前一个类Vila中的mm_model;
* pretrained，它代表模型权重路径
* batch_size，它代表运行时的batch数

这个方法重要的代码是以下参数的初始化，其他代码基本可以拷贝以及看其他方法实现的需要：

``` python
self._tokenizer = llmc_model.tokenizer
self._model = llmc_model
self._max_length = 2048
self._config = llmc_model.config
self._model.eval().cuda()
```

#### generate_until()

该方法用于得到测试集的模型推理结果，基本格式如下：

``` python
def generate_until(self, requests) -> List[str]:
    res = []
    pbar = tqdm(total=len(requests),
                disable=(self.rank != 0),
                desc="Model Responding")

    for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
    ]:
        #重要部分
        res.append(outputs)
        pbar.update(1)
    return res
```

该重要部分可以参考VLM模型的测试代码填写，实现文本和图片的预处理，然后传给`self._model.generate`推理，得到输出结果。


#### 其他方法

剩下的方法直接基本上直接拷贝就可以了

## 第四步，执行整个流程

编写Vila配置文件，参考[config_vila.yml](tpu/docs/config_vila.yml)。
然后执行：
```shell
python3 tpu/llm_quant.py --config_path config_vila.yml
```

执行效果如下(省略中间细节)：
```log
......
......
2025-02-11 18:22:56.958 | INFO     | utils:mme_aggregate_results:124 - code_reasoning: 60.00
2025-02-11 18:22:56.958 | INFO     | utils:mme_aggregate_results:124 - numerical_calculation: 62.50
2025-02-11 18:22:56.958 | INFO     | utils:mme_aggregate_results:124 - text_translation: 50.00
2025-02-11 18:22:56.958 | INFO     | utils:mme_aggregate_results:124 - commonsense_reasoning: 110.00
2025-02-11 18:22:56.961 | INFO     | utils:mme_aggregate_results:124 - artwork: 122.50
2025-02-11 18:22:56.961 | INFO     | utils:mme_aggregate_results:124 - celebrity: 122.06
2025-02-11 18:22:56.961 | INFO     | utils:mme_aggregate_results:124 - count: 138.33
2025-02-11 18:22:56.961 | INFO     | utils:mme_aggregate_results:124 - color: 175.00
2025-02-11 18:22:56.961 | INFO     | utils:mme_aggregate_results:124 - position: 125.00
2025-02-11 18:22:56.961 | INFO     | utils:mme_aggregate_results:124 - OCR: 110.00
2025-02-11 18:22:56.961 | INFO     | utils:mme_aggregate_results:124 - landmark: 159.75
2025-02-11 18:22:56.961 | INFO     | utils:mme_aggregate_results:124 - scene: 164.00
2025-02-11 18:22:56.961 | INFO     | utils:mme_aggregate_results:124 - existence: 195.00
2025-02-11 18:22:56.961 | INFO     | utils:mme_aggregate_results:124 - posters: 119.39
2025-02-11 18:22:57.068 | INFO     | llmc.eval.utils:eval_model:75 - EVAL: vqa on mme is 
|Tasks|Version|Filter|n-shot|       Metric       |   |  Value  |   |Stderr|
|-----|-------|------|-----:|--------------------|---|--------:|---|------|
|mme  |Yaml   |none  |     0|mme_cognition_score |↑  | 282.5000|±  |   N/A|
|mme  |Yaml   |none  |     0|mme_perception_score|↑  |1431.0299|±  |   N/A|
......
......
2025-02-11 22:00:56.788 | INFO     | utils:mme_aggregate_results:124 - code_reasoning: 52.50
2025-02-11 22:00:56.788 | INFO     | utils:mme_aggregate_results:124 - numerical_calculation: 62.50
2025-02-11 22:00:56.788 | INFO     | utils:mme_aggregate_results:124 - text_translation: 50.00
2025-02-11 22:00:56.788 | INFO     | utils:mme_aggregate_results:124 - commonsense_reasoning: 110.00
2025-02-11 22:00:56.791 | INFO     | utils:mme_aggregate_results:124 - artwork: 124.50
2025-02-11 22:00:56.791 | INFO     | utils:mme_aggregate_results:124 - celebrity: 126.18
2025-02-11 22:00:56.791 | INFO     | utils:mme_aggregate_results:124 - count: 133.33
2025-02-11 22:00:56.791 | INFO     | utils:mme_aggregate_results:124 - color: 170.00
2025-02-11 22:00:56.791 | INFO     | utils:mme_aggregate_results:124 - position: 123.33
2025-02-11 22:00:56.791 | INFO     | utils:mme_aggregate_results:124 - OCR: 110.00
2025-02-11 22:00:56.791 | INFO     | utils:mme_aggregate_results:124 - landmark: 162.25
2025-02-11 22:00:56.791 | INFO     | utils:mme_aggregate_results:124 - scene: 163.00
2025-02-11 22:00:56.791 | INFO     | utils:mme_aggregate_results:124 - existence: 190.00
2025-02-11 22:00:56.791 | INFO     | utils:mme_aggregate_results:124 - posters: 141.50
2025-02-11 22:00:56.840 | INFO     | llmc.eval.utils:eval_model:75 - EVAL: vqa on mme is 
|Tasks|Version|Filter|n-shot|       Metric       |   |  Value  |   |Stderr|
|-----|-------|------|-----:|--------------------|---|--------:|---|------|
|mme  |Yaml   |none  |     0|mme_cognition_score |↑  | 275.0000|±  |   N/A|
|mme  |Yaml   |none  |     0|mme_perception_score|↑  |1444.0897|±  |   N/A|

2025-02-11 22:00:56.846 | INFO     | __main__:<module>:263 - llmc_duration_time: 4043.402092218399 s
2025-02-11 22:00:56.846 | INFO     | __main__:<module>:264 - --- llmc finished ---
```
可以看出W4A16的评分与原始模型评分比较接近。

最后用新的权重测试模型，查看效果是否正常。