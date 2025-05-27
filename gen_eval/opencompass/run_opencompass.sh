#!/bin/bash

#选择计算卡号，当前只能单卡
export CUDA_VISIBLE_DEVICES=2
#选择评测配置文件
opencompass_config=/home/wanpeng.zhang/llmc-tpu/gen_eval/opencompass/eval_opencompass.py
#选择评测打印重定向文件位置
task_log_path=/dataset2/wanpeng.zhang/awq_model/opencompass/14_2_InternLM2.5_7b_wg4a16.txt
#配置llmc路径
llmc=/home/wanpeng.zhang/llmc-tpu
export PYTHONPATH=$llmc:$PYTHONPATH
#注：如选择huggingface内不支持或该模型有自定义结构发生时，需要把模型结构路径配置导出
#如llama，Qwen等通用模型不需要
InternLM=/dataset2/wanpeng.zhang/InternLM2.5-7B
export PYTHONPATH=$InternLM:$PYTHONPATH
#注：opencompass内数据路径固定为.chche/opencompass/，如需修改为自定义路径需要在opencompass/utils/datasets.py修改为
#DEFAULT_DATA_FOLDER = os.environ.get('DEFAULT_DATA_FOLDER', os.path.join(USER_HOME, '.cache/opencompass/'))
#先读取环境变量DEFAULT_DATA_FOLDER，没有配置再用默认路径
export DEFAULT_DATA_FOLDER="/dataset_local/datasets/opencompass/"



export MASTER_ADDR="127.0.0.1"
export RANK="0"
export WORLD_SIZE="1"
find_unused_port() {
    while true; do
        port=$(shuf -i 10000-60000 -n 1)
        if ! ss -tuln | grep -q ":$port "; then
            echo "$port"
            return 0
        fi
    done
}
#随机选择10000以上端口号
UNUSED_PORT=$(find_unused_port)
export MASTER_PORT=$UNUSED_PORT
#挂后台运行
nohup opencompass $opencompass_config > $task_log_path 2>&1 &

