import os
import time
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from typing import List, Dict, Union, Optional
from gen_structs import GenerateOutput
import warnings
warnings.filterwarnings("ignore")

load_dotenv()



class VLLMModel:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int = 8000,
        top_p: float = 0.95,
        max_num_seqs: int = 8,
    ):

        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.model_name = model


    def _format_prompt(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )


    def generate(
        self,
        prompt: str,
        num_return_sequences: int = 1,
    ) -> GenerateOutput:

        start_time = time.time()

        formatted_prompt = self._format_prompt(prompt)

        # correct token counting (after chat template)
        num_prompt_tokens = len(self.tokenizer.encode(formatted_prompt))

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=num_return_sequences,
        )

        outputs = self.llm.generate([formatted_prompt], sampling_params)

        all_results = []
        gen_tokens = []

        for request_output in outputs:
            for seq in request_output.outputs:
                all_results.append(seq.text)
                gen_tokens.append(len(seq.token_ids))

        time_cost = time.time() - start_time
        print(f"Generation time: {time_cost:.2f} seconds")

        return GenerateOutput(
            input_text=prompt,
            output_text=all_results,
            reason_text="NaN",
            gen_time=time_cost,
            input_token_cnt=num_prompt_tokens,
            output_token_cnt=gen_tokens,
            thoughts_token_cnt="NaN",
        )