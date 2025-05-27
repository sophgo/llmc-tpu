# llmc—opencompass

### 使用docker

* 创建docker：.69服务器上my_llmc_vllm:updated

* ```
  docker run --gpus all --privileged --name [自定义命名] -v /sea/zhangwanpeng:/workspace -v /sea/data/models:/dataset_local -p [8000:8000(自定义端口)] -it my_llmc_vllm:updated
  ```

* 进入docker激活环境：`conda activate torch231`

* llmc当前版本为9b89bad8f0bedbf738d3f8bd2dc05352f721e95a

* opencompass版本：0.4.1

* 创建容器后，需要在`llmc-tpu/gen_eval/opencompass`文件夹下配置文件：`run_opencompass.sh`及`eval_opencompass.sh`，详情见文件内注释

* 上述两文件依赖于`llmc_cross`文件，已注册到`docker`环境下的`/root/miniconda3/envs/torch231/lib/python3.10/site-packages/opencompass/models/llmc_cross.py`内，同时`__init__.py`内声明，可以详见docker内代码

* 在完成代码准备后，在所选位置执行`bash run_opencompass.sh`脚本，在当前文件夹下会生成`outputs`文件夹，中间文件和结果在`outputs/default/20250515_185623/`内：
  * `20250515_185623`：时间戳，如需修改评测后处理类或复现结果可执行`opencompass eval_base_ori.py --work-dir /workspace/llmc_vllm_models/opencompass/outputs/default/ --mode eval --reuse 20250515_185623`，其中mode选择eval，reuse选择具体时间戳

  * `infer`有具体进度，读取模型路径，数据集路径等信息打印

  * `eval`结果保存

  * `predictions`推理结果保存

* 最终结果会在`$task_log_path`最后保存

### 使用自己环境

* 安装`opencompass`：`pip install opencompass==0.4.1`
* 将`llmc-tpu/gen_eval/opencompass/llmc_cross.py`复制到conda环境`/your/conda/env/env_name/lib/pythonxx/site-package/opencompass/models/`内，如`/home/wanpeng.zhang/miniconda3/envs/llmc_vllm/lib/python3.10/site-packages/opencompass/models/`
* 修改`/your/conda/env/env_name/lib/pythonxx/site-package/opencompass/models/__init__.py`文件，路径如上将LLMC类注册进去，即末尾添加行：`from .llmc_cross import LLMC`
* 在`llmc-tpu/gen_eval/opencompass`文件夹下配置文件：`run_opencompass.sh`及`eval_opencompass.sh`，详情见文件内注释
* 在所选位置执行`bash run_opencompass.sh`脚本
