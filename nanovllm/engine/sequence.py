from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams

#保存一个请求的数据和状态
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    block_size = 256
    counter = count()
    #传入的参数：token_ids，sampling_params(采样参数，采样参数是温度，最大token数，是否忽略结束符)
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)#序列id

        self.status = SequenceStatus.WAITING#序列状态

        self.token_ids = copy(token_ids)#prompt与生成token的列表

        self.last_token = token_ids[-1]#最后一个token

        self.num_tokens = len(self.token_ids)#这个请求的完整token数

        self.num_prompt_tokens = len(token_ids)#提示token数

        self.num_cached_tokens = 0#已经计算的token数

        self.num_scheduled_tokens = 0#计划计算的token数

        self.is_prefill = True#是否是预填充

        self.block_table = []#持有的块列表

        self.temperature = sampling_params.temperature#温度

        self.max_tokens = sampling_params.max_tokens#这个请求最多允许模型生成几个token

        self.ignore_eos = sampling_params.ignore_eos#是否忽略结束符

    def __len__(self):#返回序列的总token数
        return self.num_tokens

    def __getitem__(self, key):#返回序列的第key个token
        return self.token_ids[key]

    @property#让一个普通成员函数像成员变量一样进行访问
    def is_finished(self):#是否完成
        return self.status == SequenceStatus.FINISHED

    @property#返回序列的完成token数
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens#这个是当前token数-提示词token数，得到的是完成token数
    @property#返回序列的提示token列表
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property#返回序列的完成token列表
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property#返回序列的块数
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property#返回序列的最后一个块的token数
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):#要返回序列的第i个块
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)#将token添加到序列token列表中
        self.last_token = token_id#将最后一个token设置为当前token
        self.num_tokens += 1#序列token数增加1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
