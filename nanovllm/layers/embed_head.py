import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context

#词汇并行嵌入层
#将输入的token id转换为对应的向量表示
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,#词汇表大小
        embedding_dim: int,#嵌入维度
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()#当前是第几张卡
        self.tp_size = dist.get_world_size()#总卡数
        assert num_embeddings % self.tp_size == 0#词汇表大小必须是总卡数的整数倍，否则无法并行计算
        self.num_embeddings = num_embeddings#词汇表大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size#每张卡分多少行
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank#本卡负责的词汇表起始索引
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition#当前卡的词汇表结束索引
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))#本卡负责的词汇表嵌入权重
        self.weight.weight_loader = self.weight_loader#将权重加载到本卡负责的词汇表嵌入权重中

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y

#LMhead，拿最后一个token的隐藏状态，输出下一个token的logits，词表有3.2个词，就需要每个GPU并行计算3.2个词的logits
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
