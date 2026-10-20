from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

#调度器，决定本轮让哪些Sequence使用GPU输入 token

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs#最多处理的序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens#本次调度最大批量token数

        self.eos = config.eos#结束符token id

        self.block_size = config.kvcache_block_size#KV Cache 块大小

        #传入参数，kvchache块数，块大小，block大小
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)#KV Cache 块管理器

        self.waiting: deque[Sequence] = deque()#等待序列队列
        self.running: deque[Sequence] = deque()#运行序列队列

    def is_finished(self):#是否完成判断的是等待序列队列和运行序列队列是否都为空
        return not self.waiting and not self.running#等待序列队列和运行序列队列都为空时，调度器完成

    def add(self, seq: Sequence):#添加序列到等待序列队列
        self.waiting.append(seq)#添加序列到等待序列队列

    #seqseq存的是token和采样器，不是string提示词
    def schedule(self) -> tuple[list[Sequence], bool]:#调度序列，返回本轮要使用的序列列表和是否是预填充

        scheduled_seqs = []#本轮准备调度的请求列表


        num_batched_tokens = 0#已使用的批量token数，初始值为0，本轮调度以及安排了多少   token

        # prefill
        #循环条件是等待序列队列不为空，且要使用的序列列表的长度小于等于最大处理的序列数
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:

            seq = self.waiting[0]#获取等待序列队列的第一个序列

            #用本轮最多使用的token数减去已经使用的放入token数，得到剩余的token数
            remaining = self.max_num_batched_tokens - num_batched_tokens#整个batch的token上线-已经使用放入
            if remaining == 0:# 如果已经没有剩余的token数了，跳出循环
                break
            if not seq.block_table:# 如果序列没有缓存的block
                num_cached_blocks = self.block_manager.can_allocate(seq)#判断是否可以分配block
                if num_cached_blocks == -1:# 如果没有足够的缓存block，跳出循环
                    break
               #num_tokens=这个请求的token数-已经缓存的token数
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size

            else:#如果序列有缓存的block
                #num_tokens=这个请求的token数-已经缓存的token数，等于需要计算的token
                num_tokens = seq.num_tokens - seq.num_cached_tokens


            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break

            if not seq.block_table:# 如果序列没有缓存的block
                self.block_manager.allocate(seq, num_cached_blocks)#分配block


            seq.num_scheduled_tokens = min(num_tokens, remaining)#计算本轮要使用的token数
            num_batched_tokens += seq.num_scheduled_tokens#已使用的批量token数增加本轮要使用的token数

            #nums_chached_tokens，已经完成模型计算的token数+本轮要使用的token数=序列的token数
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()#从等待序列队列中取出第一个序列
                self.running.append(seq)#添加序列到运行序列队列

            scheduled_seqs.append(seq)#添加序列到要使用的序列列表

        if scheduled_seqs:# 如果有序列要使用
            return scheduled_seqs, True#返回要使用的序列列表和是否是预填充，预填充为True

        # decode，循环条件是运行序列队列不为空，且要使用的序列列表的长度小于等于最大处理的序列数
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):#如果不能分配block
                if self.running:#如果运行序列队列不为空
                    self.preempt(self.running.pop())#抢占序列
                else:
                    self.preempt(seq)#抢占序列
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs#断言要使用的序列列表不为空，否则会报错
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False
    # 预填充，将序列从运行序列队列中取出，添加到等待序列队列中，抢占之后的序列block将被释放
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING#将序列状态设置为等待状态
        seq.is_prefill = True#将序列是否预填充设置为True
        self.block_manager.deallocate(seq)#将序列的缓存block释放
        self.waiting.appendleft(seq)#将序列添加到等待序列队列的最前面
    # 后处理序列，将序列的token添加到序列的completion_token_ids中
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)#将序列的缓存block哈希
            seq.num_cached_tokens += seq.num_scheduled_tokens#更新完成计算的token数
            seq.num_scheduled_tokens = 0#计划计算的token数设置为0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:#如果prompt有10个token，已经完成模型计算的token数小于序列的token数，说明序列还没有完成
                continue
            seq.append_token(token_id)#将token添加到序列的completion_token_ids中


            #如果生成了结束符并且没有忽略或者序列的token数等于最大token数，说明序列完成
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED#将序列状态设置为完成状态
                self.block_manager.deallocate(seq)#将序列的缓存block释放
                self.running.remove(seq)#从序列队列中取出序列
