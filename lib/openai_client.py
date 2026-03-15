import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from gen_structs import GenerateOutput, AnnotGenerateOutput
load_dotenv()




class OpenAIModel:
    def __init__(self, model: str, temperature: float=0.7, max_tokens: int = 10000):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        self.temperature = temperature
        self.max_tokens = max_tokens


    def generate(self, prompt: str, retry: int = 3) -> GenerateOutput:
        for attempt in range(retry + 1):
            try:
                start_time = time.time()
                kwargs = {
                    "model": self.model,
                    "input": prompt,
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                response = self.client.responses.create(**kwargs)
                # ipdb.set_trace()
                # print(response)
                end_time = time.time()
                time_cost = end_time - start_time
                print(f"Generation time: {time_cost:.2f} seconds")
                input_token_cnt = response.usage.input_tokens
                output_token_cnt = response.usage.output_tokens
                output_text = response.output[0].content[0].text
                
                return GenerateOutput(
                    input_text=prompt,
                    output_text=output_text,
                    reason_text="NaN",
                    gen_time=time_cost,
                    input_token_cnt=input_token_cnt,
                    output_token_cnt=output_token_cnt,
                    thoughts_token_cnt="NaN"
                )
            except Exception as e:
                print(f"Error during generation trial_No.{attempt}: {e}")


    def annot_generate(
        self, prompt: str,
        num_return_sequences: int = 1,
        response_format: dict = None,
    ) -> AnnotGenerateOutput:
        
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "n": num_return_sequences
        }
        if response_format:
            kwargs["response_format"] = response_format
        response = self.client.chat.completions.create(**kwargs)
        texto = [choice.message.content for choice in response.choices]
        
        return AnnotGenerateOutput(text=texto)
    



class OpenAIReasoningModel:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


    def generate(self, prompt: str, retry: int = 3, effort="low") -> GenerateOutput:
        for attempt in range(retry + 1):
            try:
                start_time = time.time()
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    reasoning={
                        "effort": effort,
                        "summary": "auto"
                    },
                )
                # ipdb.set_trace()
                # print(response)
                end_time = time.time()
                time_cost = end_time - start_time
                print(f"Generation time: {time_cost:.2f} seconds")
                input_token_cnt = response.usage.input_tokens
                thoughts_token_cnt = response.usage.output_tokens_details.reasoning_tokens
                output_token_cnt = response.usage.output_tokens - thoughts_token_cnt
                print(response.output)
                reason_text = "NaN"
                output_text = "NaN"
                for block in response.output:
                    # 取 reasoning summary
                    if block.type == "reasoning" and block.summary:
                        reason_text = "\n".join(s.text for s in block.summary)

                    # 取 final output
                    if block.type == "message" and block.content:
                        for c in block.content:
                            if c.type == "output_text":
                                output_text = c.text
                
                return GenerateOutput(
                    input_text=prompt,
                    output_text=output_text,
                    reason_text=reason_text,
                    gen_time=time_cost,
                    input_token_cnt=input_token_cnt,
                    output_token_cnt=output_token_cnt,
                    thoughts_token_cnt=thoughts_token_cnt
                )
            except Exception as e:
                print(f"Error during generation trial_No.{attempt}): {e}")

