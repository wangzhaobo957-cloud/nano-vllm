import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str#模型目录
    max_num_batched_tokens: int = 16384#单轮调度最多处理的token总数
    max_num_seqs: int = 512#最大序列数，每个时间步最多处理的序列数
    max_model_len: int = 4096#最大模型长度，每个序列的最大token数，不能超过模型的最大token数
    gpu_memory_utilization: float = 0.9#GPU内存利用率，默认0.9，即90%
    tensor_parallel_size: int = 1#张量并行度，默认1，即不并行
    enforce_eager: bool = False#是否强制使用Eager模式，False时可使用CUDA Graph
    hf_config: AutoConfig | None = None#HuggingFace模型配置
    eos: int = -1#结束符token id
    kvcache_block_size: int = 256#KV Cache 块大小
    num_kvcache_blocks: int = -1#KV Cache 块数，默认-1，即自动计算

    def __post_init__(self):
        assert os.path.isdir(self.model)#模型目录必须存在
        assert self.kvcache_block_size % 256 == 0#KV Cache 块大小必须是256的倍数
        assert 1 <= self.tensor_parallel_size <= 8#张量并行度必须在1到8之间
        self.hf_config = AutoConfig.from_pretrained(self.model)#从模型目录加载HuggingFace模型配置
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)#最大模型长度不能超过模型的最大token数
