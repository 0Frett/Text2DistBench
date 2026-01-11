
import os
import json
import argparse
from tqdm import tqdm
from claude_client import ClaudeBatchModel


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def run_inference_batch(
    model: ClaudeBatchModel,
    input_data,
    batch_size: int = 5000,
    num_return_sequences: int = 1,
):
    """
    以 Batch API 推論整份資料。
    - 自動依 batch_size 切批（每批一次 API 呼叫，便宜 50% token）
    - num_return_sequences > 1 時，會對每個樣本重複提交 N 份請求（在 class 內處理）
    """
    n = len(input_data)
    results = []

    for start in tqdm(range(0, n, batch_size), desc="Batch inference"):
        end = min(start + batch_size, n)
        chunk = input_data[start:end]
        prompts = [x["question"] for x in chunk]

        # 一次丟一批 prompts 進 Batch API
        outs = model.generate(
            prompt=prompts,
            num_return_sequences=num_return_sequences,
        )  # -> List[GenerateOutput]（順序與輸入對齊）

        # 組裝輸出到原本結構
        for item, gen in zip(chunk, outs):
            # gen.text 是 list；如果只要 1 個候選就回傳字串，否則保留 list
            pred = gen.text[0] if gen.n == 1 else gen.text
            usage = gen.usage[0] if gen.n == 1 else gen.usage

            item["response"] = pred
            item["input_token_cnt"] = usage.get("input_tokens", 0)
            item["output_token_cnt"] = usage.get("output_tokens", 0)
            item["out_token_cnt"] = usage.get("output_tokens", 0)
            item['gen_time'] = "NaN"

            # rec = {
            #     "qtype": item["qtype"],
            #     "qid": item["qid"],
            #     "source": item["source"],
            #     "attribute": item["attribute"],
            #     "answer": item["answer"],
            #     "pred": pred,
            #     "input_token_cnt": usage.get("input_tokens", 0),
            #     "output_token_cnt": usage.get("output_tokens", 0),
            # }
            results.append(item)

            # 可選：即時列印（大型批次可關掉以加速 I/O）
            # print("------------------------------", flush=True)
            # print(f"QID: {item['qid']}", flush=True)
            # print(f"GT: {item['answer']}", flush=True)
            # print(f"Pred: {pred}", flush=True)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="claude-3-5-haiku-20241022")
    parser.add_argument("--test_fp", type=str,
                        default="data/movie/benchmark/2025-07-01_2025-09-30/en/sampled_50/stance_dist.jsonl",
                        help="Path to JSONL file with validation prompts.")
    parser.add_argument("--output_fp", type=str,
                        default="inference_output/movie/2025-07-01_2025-09-30/en/sampled_50/stance_dist/claude-haiku.jsonl",
                        help="Path to save generated responses.")
    parser.add_argument("--batch_size", type=int, default=5000,
                        help="Number of prompts per Message Batch create() call.")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Per prompt, number of variants (duplicates requests in a batch).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_fp), exist_ok=True)
    if os.path.exists(args.output_fp):
        print(f"Pass Exist File : {args.output_fp}")
        return

    print(f"[INFO] Run OpBench (Claude Batch) on Model: {args.model_id}")

    model = ClaudeBatchModel(
        model=args.model_id,
        poll_sec=10,  # 可調整輪詢間隔
    )

    input_data = load_jsonl(args.test_fp)
    outputs = run_inference_batch(
        model=model,
        input_data=input_data,
        batch_size=args.batch_size,
        num_return_sequences=args.num_return_sequences,
    )
    save_jsonl(outputs, args.output_fp)
    print(f"[✓] Saved Output: {args.output_fp}")

if __name__ == "__main__":
    main()
