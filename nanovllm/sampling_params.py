from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0#温度，用于控制采样的随机性
    max_tokens: int = 64#最多生成的completion token数，不包含prompt
    ignore_eos: bool = False#是否忽略结束符

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
