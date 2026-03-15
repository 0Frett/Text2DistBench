from google import genai
from google.genai import types
import os
from typing import List
from dotenv import load_dotenv
import time
from gen_structs import GenerateOutput, AnnotGenerateOutput

load_dotenv()

class GeminiModel:
    def __init__(self, model="gemini-2.5-pro"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
        self.model = model

    def eval_generate(self, prompt, temperature=1, retry=3):
        for trial in range(retry + 1):
            try:
                start_time = time.time()
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        candidate_count=1,
                        thinking_config=types.ThinkingConfig(include_thoughts=True)
                    ),
                )
                end_time = time.time()
                time_cost = end_time - start_time
                print(f"Generation time: {time_cost:.2f} seconds")
                input_token_cnt = resp.usage_metadata.prompt_token_count
                output_token_cnt = resp.usage_metadata.candidates_token_count
                thoughts_token_cnt = resp.usage_metadata.thoughts_token_count
                for part in resp.candidates[0].content.parts:
                    if part.thought:
                        reason_text = part.text
                    else:
                        output_text = part.text
                
                # ipdb.set_trace()
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
                print(f"Error during generation trial_No.{trial}): {e}")
                
 
    def annot_generate(self, prompt, temperature=0.5, num_return_sequences=1, retry=3):
        max_per_call = 5
        all_texts = []
        remaining = num_return_sequences
        last_err = None

        while remaining > 0:
            batch_size = min(remaining, max_per_call)
            for _ in range(retry + 1):
                try:
                    resp = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                        
                            candidate_count=batch_size,  # up to 8 per call
                            thinking_config=types.ThinkingConfig(include_thoughts=True)
                        ),
                    )
                    # ipdb.set_trace()
                    all_texts.extend([c.content.parts[0].text for c in resp.candidates])
                    remaining -= batch_size
                    break  # success, go to next batch
                except Exception as e:
                    print(f"Error during generation (batch_size={batch_size}): {e}")
                    last_err = e
            else:
                # if retry exhausted
                raise last_err

        return AnnotGenerateOutput(text=all_texts)



 
