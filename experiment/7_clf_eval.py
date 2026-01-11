import os
import json
import re
import argparse
from typing import List, Dict, Any
import numpy as np

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file as a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """Save list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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


def top1_mass(probs):
    p = np.asarray(probs, dtype=np.float64)
    return np.max(p)

def top1_minus_top2_mass(probs):
    p = np.asarray(probs, dtype=np.float64)
    p_sorted = np.sort(p)[::-1]
    if len(p_sorted) < 2:
        return np.nan
    return p_sorted[0] - p_sorted[1]

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


def normalized_entropy(probs, eps=1e-12):
    p = np.asarray(probs, dtype=np.float64)
    total = p.sum()

    # empty or invalid distribution
    if total <= 0:
        return np.nan

    p = p / total
    p = np.clip(p, eps, 1.0)

    K = len(p)
    entropy = -np.sum(p * np.log(p))
    return entropy / np.log(K)


def majority_support_size(probs, threshold=0.5):
    p = np.asarray(probs, dtype=np.float64)
    total = p.sum()

    if total <= 0:
        return np.nan

    p = p / total
    p_sorted = np.sort(p)[::-1]
    cumulative = np.cumsum(p_sorted)

    return int(np.searchsorted(cumulative, threshold) + 1)


def normalized_majority_support_size(probs, threshold=0.5):
    K = len(probs)
    size = majority_support_size(probs, threshold)
    if np.isnan(size):
        return np.nan
    return size / K



def js_between_uniform(probs, eps=1e-12, log_base=np.e):
    """
    Jensen–Shannon divergence between a categorical distribution and uniform.

    Args:
        probs: iterable of non-negative numbers
        eps: small constant for numerical stability
        log_base: np.e (nats) or 2 (bits)

    Returns:
        JSD(P || Uniform)
    """
    p = np.asarray(probs, dtype=np.float64)
    total = p.sum()

    if total <= 0:
        return np.nan

    # normalize
    p = p / total
    K = len(p)
    u = np.full(K, 1.0 / K)

    # mixture
    m = 0.5 * (p + u)

    # avoid log(0)
    p = np.clip(p, eps, 1.0)
    m = np.clip(m, eps, 1.0)

    # KL(P || M)
    kl_pm = np.sum(p * np.log(p / m))

    # KL(U || M)
    kl_um = np.sum(u * np.log(u / m))

    js = 0.5 * (kl_pm + kl_um)

    # change log base if needed
    if log_base != np.e:
        js /= np.log(log_base)

    return js



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
        "random_correctness": (len(gt) / len(ref_dist)),
        "normalized_entropy": normalized_entropy(list(ref_dist.values())),
        "normalized_majority_support_size": normalized_majority_support_size(list(ref_dist.values())),
        "top1_mass": top1_mass(list(ref_dist.values())),
        "top1_minus_top2_mass": top1_minus_top2_mass(list(ref_dist.values())),
        "js_between_uniform": js_between_uniform(list(ref_dist.values())),
        **item
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
        required=True,
        help="Domain.",
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
