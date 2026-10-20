import torch
from torch import nn
import torch.nn.functional as F

#SiluAndMul激活函数
#将输入的张量x拆分成两个张量x和y，对x应用SILU激活函数，对y应用乘法操作，返回结果
class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
