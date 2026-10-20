import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

#真正在GPU上运行
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config#配置对象
        hf_config = config.hf_config#huggingface配置对象

        self.block_size = config.kvcache_block_size#块大小

        self.enforce_eager = config.enforce_eager#是否强制使用eager模式
        self.world_size = config.tensor_parallel_size#几张卡
        self.rank = rank#当前是第几张卡
        self.event = event#进程间通信事件
    #初始化分布式，告诉他一共有多少个进程，我是第几个
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)#rank=1 → 用 GPU 1
        default_dtype = torch.get_default_dtype()#默认精度
        torch.set_default_dtype(hf_config.dtype)#设置模型精度
        torch.set_default_device("cuda")#默认在GPU上建张量

        #加载模型
        self.model = Qwen3ForCausalLM(hf_config)#加载模型
        load_model(self.model, config.model)

        #初始化采样器
        self.sampler = Sampler()#初始化采样器
        self.warmup_model()#预热模型
        self.allocate_kv_cache()#分配kv缓存
        if not self.enforce_eager:
            self.capture_cudagraph()#捕获cuda graph
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:#如果有多个进程
            if rank == 0:#如果是第一个进程
                #创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()#等所有进程到齐
            else:#如果不是第一个进程
                dist.barrier()#等主进程创建好
                self.shm = SharedMemory(name="nanovllm")#链接共享内存
                self.loop()#子进程进入循环

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:#死循环
            method_name, args = self.read_shm()#从共享内存读取方法名和参数
            self.call(method_name, *args)#调用方法
            if method_name == "exit":#如果是退出方法
                break
#从共享内存读取方法名和参数
    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):#分配kv缓存
        config = self.config
        hf_config = config.hf_config#huggingface配置对象


        free, total = torch.cuda.mem_get_info()#获取当前GPU的空闲内存和总内存
        used = total - free#已用内存=总内存-空闲内存，used是物理占用

        #peak-current是模型预热阶段额外产生的峰值显存
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]#历史峰值，因为需要计算
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]#当前已用内存
        #整个模型的KV head数 /卡数=每个卡的KV head数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size#
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        #计算每个块的字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        #计算每个卡的kv缓存块数
        #目标显存-物理已用显存-(历史峰值-当前显存)=可用于KV Cache的显存
        #每个卡的可用内存/每个块的字节数=每个卡的kv缓存块数
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        #分配kv缓存，每个参数的含义
        #2，K和V两份
        #模型层数
        #block编号
        #self.block_size，每个block能放多少个token
        #num_kv_heads，每个卡的KV head数
        #head_dim，每个卡的KV head的维度
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0

        #将每个卡的kv缓存赋值给每个层的k_cache和v_cache
        for module in self.model.modules():
            #
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    #输入的参数，seqs，要执行的序列列表
    #返回的参数，input_ids，模型的输入token id列表
    #positions，模型的输入位置列表
    #cu_seqlens_q，模型的输入序列长度列表
    #cu_seqlens_k，模型的输入序列长度列表
    #max_seqlen_q，模型的输入序列长度
    #max_seqlen_k，模型的输入序列长度
    #slot_mapping，模型的输入slot映射列表
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []#输入token列表
        positions = []#输入位置列表
        cu_seqlens_q = [0]#保存每个token在seq中的位置
        cu_seqlens_k = [0]#保存每个token在seq中的位置
        max_seqlen_q = 0#模型的输入序列长度，初始值为0
        max_seqlen_k = 0#模型的输入序列长度，初始值为0
        slot_mapping = []#保存每个token在KV cache中的位置
        block_tables = None#Block Table 读取 Prefix Cache。
        for seq in seqs:
            start = seq.num_cached_tokens#已经完成模型计算的token数字

            seqlen_q = seq.num_scheduled_tokens#本轮要计算的token数
            end = start + seqlen_q#本轮结束位置

            seqlen_k = end#K是历史token+本轮token数


            input_ids.extend(seq[start:end])#将本轮要计算的token id添加到输入token id列表中
            positions.extend(range(start, end))#位置列表，0，1，2，3


            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)#累计Q序列长度
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)#累计K序列长度
            max_seqlen_q = max(seqlen_q, max_seqlen_q)#统计本轮要计算的token数和最大序列长度的较大值，作为模型的输入序列长度
            max_seqlen_k = max(seqlen_k, max_seqlen_k)#统计本轮要计算的token数和最大序列长度的较大值，作为模型的输入序列长度
            if not seq.block_table:    # warmup如果没有block_table，直接跳过
                continue

            #下面这一段都是把逻辑token位置映射到物理KV cache slot位置
            start_block = start // self.block_size#已经计算的token数除以块大小，得到已经计算的块数
            end_block = (end + self.block_size - 1) // self.block_size#本seq要计算的token数除以块大小，得到本seq要计算的块数


            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size#当前块的起始位置
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            hidden_states = self.model(input_ids, positions)#emdedding->transformer->hidden_states

            #计算logits，得到原始分数
            return self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    #输入的参数，seqs，要执行的序列列表，is_prefill，是否是预填充
    #返回的参数，token_ids，模型的输出token id列表，每个元素是一个序列的输出token id列表
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        #根据是否是预填充，调用不同的准备函数
        #如果是预填充，调用prepare_prefill函数
        #如果不是预填充，调用prepare_decode函数
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)#执行模型计算
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
