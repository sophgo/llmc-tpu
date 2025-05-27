from mmengine.config import read_base
from opencompass.models import HuggingFaceBaseModel
from opencompass.models.llmc_cross import LLMC




models = [
    dict(
        type= LLMC,#类名
        abbr='internlm-7b',#自定义任务名
        path='/dataset2/wanpeng.zhang/awq_model/InternLM2_7B_w4a16/transformed_model',#llmc量化后的transformed模型路径
        batch_size=24,#batch大小对结果有一点的影响，要求同种模型在同一个数据集下batch不变
        generation_kwargs = {
            'config_path':'/home/wanpeng.zhang/llmc_yq/configs/quantization/methods/Awq/awq_w_a_chat_internlm_w4a16.yml',#llmc量化配置yml路径
            'quant_type' : 'fake_quant',#只有pretrain和fake_quant两种选择，原始模型用pretrain，llmc量化模型用fake_quant
            'temperature': 0,
            'max_new_tokens': 256,#最多生成多少新的token，受数据集影响，如果选择填空题可以限制128或更少，编程题可以设置512或1024等
            'do_sample': False,#确定性结果，禁止随机采样
        },
        run_cfg=dict(num_gpus=1),
    )
]



with read_base():
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_gen_742f0c import sanitized_mbpp_datasets
    from opencompass.configs.datasets.gpqa.gpqa_gen_4baadb import gpqa_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_3321a3 import ifeval_datasets
    from opencompass.configs.datasets.babilong.babilong_0k_gen import babiLong_0k_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_gen_a58960 import gsm8k_datasets
    from opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_20a989 import triviaqa_datasets
    from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets
    from opencompass.configs.datasets.winogrande.winogrande_gen import winogrande_datasets



#可以连续评测，也可以单独评测一个
datasets = [*humaneval_datasets,*sanitized_mbpp_datasets,*gpqa_datasets,*gsm8k_datasets,*ifeval_datasets,*babiLong_0k_datasets]
# datasets = [*triviaqa_datasets]
