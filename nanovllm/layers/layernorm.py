import torch
from torch import nn

#层归一化层，稳定每个token隐藏向量的数值尺度，避免经过多层计算后过大或者过小
# 输入 x：[token数, hidden_size]
# 输出 y：[token数, hidden_size]
class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps#小常量，用于除0
        self.weight = nn.Parameter(torch.ones(hidden_size))#权重参数，用于缩放层归一化后的输出

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:#如果没有残差，直接进行层归一化
            return self.rms_forward(x)
        else:#如果有残差，先融合残差加法，再对结果进行层归一化
            return self.add_rms_forward(x, residual)
