# grok_client.py
import os, time
from typing import List, Union
from openai import OpenAI
from dotenv import load_dotenv
from gen_structs import GenerateOutput, AnnotGenerateOutput
load_dotenv()




class GrokModel:
    def __init__(self, model="grok-4-0709"):
        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROK_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        self.model = model

    def eval_generate(self, prompt: str, temperature = 0.5, retry=2):
        for trial in range(retry + 1):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    n=1,
                    # temperature=self.temperature,
                    # max_tokens=self.max_tokens,
                )
                end_time = time.time()
                time_cost = end_time - start_time
                output_text = response.choices[0].message.content
                input_token_cnt = response.usage.prompt_tokens
                output_token_cnt = response.usage.completion_tokens
                thoughts_token_cnt = response.usage.completion_tokens_details.reasoning_tokens

                return GenerateOutput(
                    input_text=prompt,
                    output_text=output_text,
                    reason_text="encrypted",
                    gen_time=time_cost,
                    input_token_cnt=input_token_cnt,
                    output_token_cnt=output_token_cnt,
                    thoughts_token_cnt=thoughts_token_cnt
                )
            except Exception as e:
                print(f"Error during generation trial_No.{trial}): {e}")


    def annot_generate(self, prompt: Union[str, List[str]], temperature = 0.5, num_return_sequences=1, retry=2):
        def _one(p):
            last = None
            for a in range(retry + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": p}],
                        temperature=temperature,
                        n=num_return_sequences,
                        # max_tokens=max_tokens,
                    )
                    # ipdb.set_trace()
                    
                    return AnnotGenerateOutput(text=[c.message.content for c in response.choices])
                except Exception as e:
                    last = e
                    if a < retry:
                        time.sleep(0.8 ** a)
            raise last
        if isinstance(prompt, str):
            return _one(prompt)
        else:
            return [_one(p) for p in prompt]
