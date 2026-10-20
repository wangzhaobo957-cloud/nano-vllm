import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
#这个文件是演示怎么用的
#认识tokenizer,prompt,sampling_params
def main():
    # 模型路径
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 加载tokenizer，加载词表，根据模型自动选择分词器，把文本转换为token id列表
    tokenizer = AutoTokenizer.from_pretrained(path)
    # 加载模型，路径，是否立即加载模型，是否并行计算
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    #模型温度，0.6表示中等温度，0.2表示更确定的输出，1.0表示更随机的输出，采集参数，最大token数，最大生成token数
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [#输入提示词2个
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    result = []
    for prompt in prompts:
        result.append(tokenizer.apply_chat_template(#将提示词转换为模型输入的格式
            [{"role": "user", "content": prompt}],#用户角色，内容为提示词
            tokenize=False,#不进行tokenization
            add_generation_prompt=True,#添加生成提示词
        ))#将转换后的提示词添加到结果列表中
    prompts = result#将结果列表赋值给prompts
    outputs = llm.generate(result, sampling_params)#开始推理

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
