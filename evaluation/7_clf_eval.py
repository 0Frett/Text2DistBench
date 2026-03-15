import os
import json
import re
import argparse
from typing import List, Dict, Any
from io_utils import load_jsonl, save_jsonl
from eval_utils import top1_mass, top1_minus_top2_mass, js_between_uniform



def extract_json_from_text(text: str) -> str | None:
    """
    Attempts to extract the FIRST {...} block from the text that looks like JSON.
    Returns the JSON substring if found, else None.
    """
    stack = []
    start = None
    candidates = []

    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text[start:i + 1])
                    start = None

    # Try parsing each candidate as JSON, return the first valid one
    for block in candidates:
        try:
            json.loads(block)
            return block
        except json.JSONDecodeError:
            continue

    return None



def _get_pred(text: str) -> Dict[str, Any]:
    """
    Normalize quotes/control chars, extract first JSON block, and parse it.
    Returns a dict (possibly empty if parsing fails).
    """
    # normalize smart quotes and remove control chars
    text_fixed = (
        text.replace("“", '"').replace("”", '"')
            .replace("’", "'").replace("`", "'")
            .replace("{{", "{").replace("}}", "}")
    )
    text_fixed = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text_fixed)
    json_block = extract_json_from_text(text_fixed)

    if json_block:
        try:
            pred_json = json.loads(json_block)
        except json.JSONDecodeError:
            pred_json = {}
    else:
        pred_json = {}

    return pred_json



def QA_eval(item: Dict[str, Any], task: str) -> Dict[str, Any]:
    # parse prediction
    pred_json = _get_pred(item["response"])
    gt = item["answer"]

    # evaluate correctness
    if task in ["P_s", "P_s_cond_t"]:
        pred_label = pred_json.get("stance")
        is_correct = (pred_label in gt)

    elif task in ["P_t", "P_t_cond_s"]:
        pred_label = pred_json.get("aspect")
        is_correct = (pred_label in gt)

    elif task == "P_ts":
        pred_label = pred_json.get("combination")
        if pred_label:
            pred_label = pred_label.replace("(", "").replace(")", "").replace(" ", "")
        is_correct = (f"({pred_label})" in gt)

    else:
        raise ValueError(f"Unknown task: {task}")

    # write back
    print(f"GT: {gt} | Pred: {pred_json} | Correct: {is_correct}")
    ref_dist = item['ref_dist']
    new = {
        "pred": pred_json, 
        "correctness": is_correct,
        # "random_correctness": (len(gt) / len(ref_dist)),
        "top1_mass": top1_mass(list(ref_dist.values())),
        "top1_minus_top2_mass": top1_minus_top2_mass(list(ref_dist.values())),
        "js_between_uniform": js_between_uniform(list(ref_dist.values())),
        "support_size": int(len(list(ref_dist.values()))),
    }
    return new


def main(pred_fp: str, output_fp: str, task: str, domain: str) -> None:
    pred_data = load_jsonl(pred_fp)
    evals = []
    for item in pred_data:
        eval_output = QA_eval(item, task)
        evals.append(eval_output)

    save_jsonl(evals, output_fp)
    print(f"Eval results saved to {output_fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process answer and prediction files.")
    parser.add_argument(
        "--pred_fp",
        type=str,
        required=True,
        help="Path to prediction JSONL file.",
    )
    parser.add_argument(
        "--output_fp",
        type=str,
        required=True,
        help="Path to eval result JSONL file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["P_s", "P_t", "P_s_cond_t", "P_t_cond_s", "P_ts"],
        required=True,
        help="Type of QA evaluation.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["movie", "music"],
        help="Domain",
    )

    args = parser.parse_args()
    pred_fp = args.pred_fp
    output_fp = args.output_fp
    task = args.task
    domain = args.domain

    out_dir = os.path.dirname(output_fp)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    main(pred_fp, output_fp, task, domain)
