import os
import json
import argparse
from tqdm import tqdm
from google_client import GeminiModel
from io_utils import load_jsonl, save_jsonl



def run_inference(model, input_data):
    """Run inference on a single model checkpoint."""

    results = []
    for item in tqdm(input_data):
        prompt = item["question"]
        # print(prompt)
        gen = model.eval_generate(prompt=prompt)
        pred = '<think>' + gen.reason_text + '<think>' + gen.output_text
        total_token_cnt = gen.output_token_cnt + gen.thoughts_token_cnt
        item["response"] = pred
        item["input_token_cnt"] = gen.input_token_cnt
        item["output_token_cnt"] = gen.output_token_cnt
        item["thoughts_token_cnt"] = gen.thoughts_token_cnt
        item["out_token_cnt"] = total_token_cnt
        item['gen_time'] = gen.gen_time
        results.append(item)
        print("------------------------------", flush=True)
        print(f"QID: {item['qid']}", flush=True)
        print(f"GT: {item['answer']}", flush=True)
        print(f"Pred: {pred}", flush=True)

    return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", 
        type=str, 
        default=None,
    )
    parser.add_argument(
        "--test_fp", 
        type=str, 
        default="data/movie/benchmark/2025-07-01_2025-09-30/en/sampled_50/stance_dist.jsonl", 
        help="Path to JSONL file with validation prompts."
    )
    parser.add_argument(
        "--output_fp", 
        type=str, 
        default="inference_output/movie/2025-07-01_2025-09-30/en/sampled_50/stance_dist/gemma-3-27b-it.jsonl",
        help="Directory to save generated responses."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_fp), exist_ok=True)
    if os.path.exists(args.output_fp):
        print(f"Pass Exist File : {args.output_fp}")
        return
    
    
    print(f"[INFO] Run OpBench on Model: {args.model_id}")

    model = GeminiModel(model=args.model_id)

    input_data = load_jsonl(args.test_fp)
    outputs = run_inference(model, input_data)    
    save_jsonl(outputs, args.output_fp)
    print(f"[✓] Saved Output: {args.output_fp}")

    return



if __name__ == "__main__":
    main()

