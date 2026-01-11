import os
import threading
import time
from queue import Queue
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import ipdb
from gen_structs import GenerateOutput
load_dotenv()



class OpenAIWorker(threading.Thread):
    def __init__(self, queue: Queue, model: str, api_key: str, max_tokens: int, rate_limit_per_min: int):
        threading.Thread.__init__(self)
        self.queue = queue
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
        self.rate_limit_per_min = rate_limit_per_min

    def run(self):
        while True:
            prompt, num_return_sequences, temperature, retry, result_queue, response_format = self.queue.get()
            for i in range(1, retry + 1):
                try:
                    if self.rate_limit_per_min is not None:
                        time.sleep(60 / self.rate_limit_per_min)
 
                    messages = [{"role": "user", "content": prompt}]
                    kwargs = {
                        "model": self.model,
                        "messages": messages,
                        "max_completion_tokens": self.max_tokens,
                        "temperature": temperature,
                        "n": num_return_sequences
                    }

                    if response_format:
                        kwargs["response_format"] = response_format

                    response = self.client.chat.completions.create(**kwargs)
                    texto = [choice.message.content for choice in response.choices]
                    result_queue.put(GenerateOutput(text=texto))
                    self.queue.task_done()
                    break
 
                except Exception as e:
                    print(f"An Error Occured: {e}, sleeping for {i} seconds")
                    time.sleep(i)
            else:
                result_queue.put(RuntimeError(f"GPTCompletionModel failed to generate output, even after {retry} tries"))
                self.queue.task_done()


class OpenAIModel_parallel():
    def __init__(self, model: str, temperature:float, max_tokens: int = 2048, num_workers: int = 2):
        self.model = model
        self.max_tokens = max_tokens
        self.queue = Queue()
        self.num_workers = num_workers
        self.workers = []
        self.temperature = temperature

        for _ in range(num_workers):
            api_key = os.getenv("OPENAI_API_KEYS", "").split(',')[_%num_workers]
            worker = OpenAIWorker(self.queue, model, api_key, max_tokens, rate_limit_per_min=9999999)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def generate(
        self,
        prompt: str,
        num_return_sequences: int = 1,
        retry: int = 10,
        response_format: dict = None,
    ) -> GenerateOutput:
        
        result_queue = Queue()
        self.queue.put(
            (
                prompt, 
                num_return_sequences, 
                self.temperature, 
                retry, result_queue, response_format
            )
        )
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
        return result



class OpenAIModel:
    def __init__(self, model: str, temperature: float=0.7, max_tokens: int = 10000):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY2", ""))
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


# class OpenAIReasoningModel():
#     def __init__(self, model: str):
#         self.model = model
#         self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY2", ""))

#     def generate(self, prompt, num_return_sequences: int = 1):
#         response = self.client.responses.create(
#             model=self.model,
#             input=[
#                 {
#                     "role": "user", 
#                     "content": prompt
#                 }
#             ],
#             reasoning={
#                 "effort": "low",
#                 # "summary": "auto" 
#             }
#         )
#         # ipdb.set_trace()
#         return GenerateOutput(text=[response.output[1].content[0].text])

class OpenAIReasoningModel:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY2", ""))


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


    def gssgenerate(self, prompt, num_return_sequences: int = 1, retry: int = 5):
        all_texts = []
        for i in range(num_return_sequences):
            last_err = None
            for attempt in range(retry + 1):
                try:
                    response = self.client.responses.create(
                        model=self.model,
                        input=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        reasoning={
                            "effort": "medium",
                            "summary": "auto"
                        },
                    )
                    ipdb.set_trace()
                    all_texts.append(response.output[1].content[0].text)
                    break  # success → go to next generation
                except Exception as e:
                    print(f"Attempt {attempt+1} failed for sequence {i+1}: {e}")
                    last_err = e
            else:
                # all retries failed
                raise last_err

        return GenerateOutput(text=all_texts)

if __name__ == "__main__":
    import json
    fp = "data-v5/movie/benchmark/prior/2025-07-01_2025-09-30/en/sampled_100/P_t.jsonl"
    data = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    # model = OpenAIModel_parallel('gpt-5-mini-2025-08-07', temperature=1, max_tokens=9999, num_workers=2)
    model = OpenAIReasoningModel('gpt-5.1')
    # model = OpenAIModel('gpt-4.1', temperature=0.7, max_tokens=10000)
    q = data[0]["question"]
    print(q)
    # output = model.generate(
    #     prompt=q,
    #     retry=3,
    # )
    # print(output.text)
    output = model.generate(
        prompt=q, effort="high"
    )
    output.printout()