import os
import time
from typing import List
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
# from vllm import QuantizationConfig
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

VLM_MODELS = [
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
    'google/gemma-2-2b-it',
    'google/gemma-2-9b-it',
    'google/gemma-2-27b-it',
    'meta-llama/Llama-3.3-70B-Instruct',
    'lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit',
    'mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    'google/gemma-3-4b-it',
    'google/gemma-3-12b-it',
    'google/gemma-3-27b-it',  #
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-14B-Instruct',
    'Qwen/Qwen2.5-32B-Instruct',  #
    'Qwen/Qwen3-32B-AWQ',
    'Qwen/Qwen3-32B-FP8',
    'Qwen/Qwen2.5-72B-Instruct',
    "allenai/OLMo-2-1124-7B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Ministral-8B-Instruct-2410",
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8',
    'openai/gpt-oss-20b'  #
]

class GenerateOutput():
    """
    text: List[str]
        The generated text sequences from the model.
    """
    def __init__(self, text: List[str], n: int, input_token_cnt: int = None, output_token_cnt: List[int] = None):
        self.text = text
        self.n = n
        self.input_token_cnt = input_token_cnt
        self.output_token_cnt = output_token_cnt


class vlmModel():
    def __init__(self, model, max_tokens, temperature, tensor_parallel_size, gpu_memory_utilization, max_model_len=5000):
        self.model = LLM(
            model=model, 
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=4,
            task="generate",
            trust_remote_code=True, 
            enforce_eager=True
        )
        # Retrieve the tokenizer from the model
        self.tokenizer = self.model.get_tokenizer()
        self.max_tokens = max_tokens
        self.temperature = temperature
        # self.stop_token_ids = [self.tokenizer.eos_token_id]

    def generate(
        self, 
        prompt: str, 
        num_return_sequences: int = 1, 
    ) -> GenerateOutput:
        
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.95, 
            max_tokens=self.max_tokens, 
            n=num_return_sequences,
        )
        message = [{"role": "user", "content": prompt}]
        prompt_token_ids = self.tokenizer.apply_chat_template([message], add_generation_prompt=True)
        num_prompt_tokens = len(prompt_token_ids)
        outputs = self.model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)[0].outputs

        texto = [completion.text for completion in outputs]
        gen_tokens = [len(c.token_ids) for c in outputs]


        return GenerateOutput(text=texto, n=len(texto), input_token_cnt=num_prompt_tokens, output_token_cnt=gen_tokens)
    


if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()

    import json
    fp = "data-v5/movie/benchmark/estimation/2025-07-01_2025-09-30/en/level0/sampled_100/P_s_cond_t.jsonl"
    data = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    q = data[1]["question"]


    model = vlmModel(
        model='Qwen/Qwen3-32B-FP8',
        max_tokens=10000,
        temperature=0.6,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        max_model_len=32768
    )

    # model = vlmModel(
    #     model='lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit',
    #     max_tokens=8000,
    #     temperature=0.6,
    #     tensor_parallel_size=2,
    #     gpu_memory_utilization=0.95,
    #     max_model_len=15000
    # )
    print(q)
    gen = model.generate(prompt=q, num_return_sequences=1)
    print(gen.text)
