import os
import json
import argparse
from tqdm import tqdm
import torch


def load_jsonl(path):
    """Load a JSONL file as a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data



def run_inference(model, input_data, n):
    """Run inference on a single model checkpoint."""

    results = []
    for item in tqdm(input_data):
        prompt = item["question"]
        gen = model.generate(prompt=prompt, num_return_sequences=n)
        pred = gen.text[0] if gen.n == 1 else gen.text
        input_token_cnt = gen.input_token_cnt
        output_token_cnt = gen.output_token_cnt[0] if gen.n == 1 else gen.output_token_cnt
        
        item["response"] = pred
        item["input_token_cnt"] = input_token_cnt
        item["output_token_cnt"] = output_token_cnt
        results.append(item)
        print("------------------------------", flush=True)
        print(f"QID: {item['qid']}", flush=True)
        print(f"GT: {item['answer']}", flush=True)
        print(f"Pred: {pred}", flush=True)
    return results



def save_jsonl(data, path):
    """Save list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", 
        type=str, 
        default=None,
        help="Hugging Face model ID or local path for a pretrained model. If specified, uses pretrained mode."
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
    parser.add_argument(
        "--n_return", type=int, default=1
    )
    parser.add_argument(
        "--max_model_len", type=int, default=6000
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=4
    )
    parser.add_argument(
        "--max_output_tokens", type=int, default=1024
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.95
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_fp), exist_ok=True)
    if os.path.exists(args.output_fp):
        print(f"Pass Exist File : {args.output_fp}")
        return
    
    print(f"[INFO] Run OpBench on Model: {args.model_id}")
    from local_llms import vlmModel
    model = vlmModel(
        model=args.model_id,
        max_tokens=args.max_output_tokens, 
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )
    input_data = load_jsonl(args.test_fp)
    outputs = run_inference(model, input_data, args.n_return)    
    save_jsonl(outputs, args.output_fp)
    print(f"[✓] Saved Output: {args.output_fp}")

    return



if __name__ == "__main__":
    main()

