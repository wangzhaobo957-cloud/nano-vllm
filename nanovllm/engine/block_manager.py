from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence
#KV Cache 块管理器，管理所有KV Cache 块
class Block:
#KV Cache 块，存储一个token id列表
#每个token id列表的长度为块大小
    def __init__(self, block_id):#构造函数，这玩意不需要声明自己有哪些资源
        self.block_id = block_id#传入的就是blockid这些东西
        self.ref_count = 0#引用计数
        self.hash = -1#哈希值，初始值为-1，表示没有被使用
        self.token_ids = []#空列表，存储块中的token id#每个token id列表的长度为块大小

    def update(self, hash: int, token_ids: list[int]):#建议传入整数，建议传入整数列表，->int代表了建议返回整数
        self.hash = hash#更新哈希值为传入的hash
        self.token_ids = token_ids#更新块中的token id列表

    def reset(self):
        self.ref_count = 1#重置引用计数为1，表示被一个序列引用
        self.hash = -1#重置哈希值为-1，表示没有被使用
        self.token_ids = []#重置块中的token id列表为空


class BlockManager:
#传入的参数：块数，块大小
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size#块大小

        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]#块列表，存储所有块

        self.hash_to_block_id: dict[int, int] = dict()#哈希值到块id的映射，字典，存储所有块的哈希值和对应的id，用于快速查找块

        self.free_block_ids: deque[int] = deque(range(num_blocks))#存储所有空闲块的id，初始值为0到num_blocks-1
        self.used_block_ids: set[int] = set()#存储所有被使用的块的id，初始值为空集

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()#创建一个xxh64哈希对象
        if prefix != -1:#如果前缀不是-1
            h.update(prefix.to_bytes(8, "little"))#将前缀转换为字节序列，更新哈希对象
        h.update(np.array(token_ids).tobytes())#将token id列表转换为字节序列，更新哈希对象
        return h.intdigest()#返回哈希对象的整数表示

    def _allocate_block(self) -> int:#分配block
        block_id = self.free_block_ids.popleft()#从空队列中取出第一个blockid
        block = self.blocks[block_id]#从块列表中取出对应的块
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:#判断是否可以分配block
        h = -1
        num_cached_blocks = 0#已经缓存的block数
        num_new_blocks = seq.num_blocks#
        for i in range(seq.num_blocks - 1):#遍历序列持有的块列表，判断是否可以分配block
            token_ids = seq.block(i)#从序列持有的块列表中取出第i个块的token id列表，第0个块
            h = self.compute_hash(token_ids, h)#计算token id列表的哈希值
            block_id = self.hash_to_block_id.get(h, -1)#从哈希值到块id的映射中取出对应的块id，如果不存在则返回-1
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:#如果块id不存在，或者块中的token id列表与当前token id列表不一致
                break
            num_cached_blocks += 1#已经缓存的block数加1，因为当前块可以被缓存
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks#返回已经缓存的block数

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:#判断是否可以分配block
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        #已经缓存的token数除以块大小，得到已经缓存的block数
        start = seq.num_cached_tokens // self.block_size

        #已经缓存的token数+本轮要使用的token数除以块大小，得到本轮要使用的block数
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
