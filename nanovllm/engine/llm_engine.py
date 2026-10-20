import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

#总控制器，持有模型运行器，分词器，调度器
#负责将输入的字符串转成token_id,转化成sequence,将sequence放入调度器，循环调用step
class LLMEngine:


    #初始化LLMEngine，设置模型目录
    def __init__(self, model, **kwargs):#模型目录，接受所有带名字的额外参数
        #配置参数的过滤和集合获取
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)#创建配置对象

        #设置KV块大小
        Sequence.block_size = config.kvcache_block_size
        #准备子进程容器
        self.ps = []#process list
        self.events = []#event list
        ctx = mp.get_context("spawn")#多进程启动方式
        for i in range(1, config.tensor_parallel_size):#为每个GPU创建一个子进程
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()#启动子进程
            self.ps.append(process)#添加子进程到process list
            self.events.append(event)#添加事件到event list

        #主进程创建模型运行器，用于与子进程通信
        self.model_runner = ModelRunner(config, 0, self.events)#主进程创建模型运行器

        #创建分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)#创建分词器

        #设置结束符
        config.eos = self.tokenizer.eos_token_id#设置结束符

        #创建调度器
        self.scheduler = Scheduler(config)#创建调度器

        #注册退出时调用exit方法
        atexit.register(self.exit)#注册退出时调用exit方法

#退出时调用exit方法
    def exit(self):
        self.model_runner.call("exit")#通知模型运行器退出
        del self.model_runner#删除模型运行器
        for p in self.ps:
            p.join()#添加请求



    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        #预处理提示词(如果提示词是字符串，预处理)
        if isinstance(prompt, str):#如果是字符串，需要先编码
            prompt = self.tokenizer.encode(prompt)#，如果是字符串编码为整数列表

        #创建序列
        #将提示词和采集参数打包，创建序列
        seq = Sequence(prompt, sampling_params)#创建序列
        self.scheduler.add(seq)#添加序列到等待序列队列



#执行一步
    def step(self):


        #从调度器获取要执行的序列，是否是预填充
        seqs, is_prefill = self.scheduler.schedule()#调度序列，返回本轮要使用的序列列表和是否是预填充


        #计算要执行的序列的总token数，如果是预填充是本轮要使用的token数，如果不是预填充是负数
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)


        #调用模型运行器执行序列
        token_ids = self.model_runner.call("run", seqs, is_prefill)


        #后处理序列
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens


#判断是否完成
    def is_finished(self):
        return self.scheduler.is_finished()#看的是调度器的队列是否还有请求


#生成输出，传入参数，输入提示词，采集参数，是否使用tqdm进度条
    def generate(
        self,
        prompts: list[str] | list[list[int]],#接受字符串列表或整数列表，每个元素是一个提示词
        sampling_params: SamplingParams | list[SamplingParams],#接受采集参数或采集参数列表，每个元素是一个采集参数
        use_tqdm: bool = True,
    ) -> list[str]:#返回值注解，每个元素是一个包含文本和token id的字典

        #创建进度条，无所谓，这行不看
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)



        #将采集参数转换为列表，每个元素是一个采集参数
        if not isinstance(sampling_params, list):#判断采集参数是否为列表
            #如果不是列表，将采集参数重复len(prompts)次，如果是列表，直接使用
            sampling_params = [sampling_params] * len(prompts)#将采集参数重复len(prompts)次

        #将提示词和采集参数打包，遍历每个提示词和采集参数
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)#提示此和采集参数添加到调度器中，然后会把提示词和采样器格式化为序列，序列加到调度器的等待队列里等待调度



        outputs = {}#空字典

        #开始推理
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():#等到调度器的等待队列和运行队列为空，推理才会结束
            t = perf_counter()#记录当前时间



            output, num_tokens = self.step()#执行一轮推理



            if num_tokens > 0:#如果是预填充
                prefill_throughput = num_tokens / (perf_counter() - t)#计算预填充吞吐量
            else:#如果是解码
                decode_throughput = -num_tokens / (perf_counter() - t)#计算解码吞吐量


            #进度条，不关注这个带啊吗
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })


            #收集本轮产出的token id
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
