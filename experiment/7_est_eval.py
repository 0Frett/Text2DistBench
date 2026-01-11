import os
import json
import re
import argparse
import numpy as np
from typing import List, Dict, Any

def load_jsonl(path):
    """Load a JSONL file as a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    """Save list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")




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


def top1_mass(probs):
    p = np.asarray(probs, dtype=np.float64)
    return np.max(p)

def top1_minus_top2_mass(probs):
    p = np.asarray(probs, dtype=np.float64)
    p_sorted = np.sort(p)[::-1]
    if len(p_sorted) < 2:
        return np.nan
    return p_sorted[0] - p_sorted[1]



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



def extract_json_from_text(text):
    """
    Attempts to extract the FIRST valid JSON object from the text.
    Returns a Python dict if success, else None.
    """
    text_fixed = (
        text.replace("“", '"').replace("”", '"')
            .replace("’", "'").replace("`", "'")
            .replace("{{", "{").replace("}}", "}")
    )
    text_fixed = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text_fixed)
    # 2. Find all {...} blocks using a bracket-matching approach
    stack = []
    start = None
    candidates = []

    for i, ch in enumerate(text_fixed):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text_fixed[start:i+1])
                    start = None

    # 3. Try parsing each candidate
    for block in candidates:
        return block
    # print(text)
    return None


#  "P_s" "P_t" "P_s_cond_t" "P_t_cond_s" "P_ts"

def _parse_P_t(text, domain):
    # normalize smart quotes and remove control chars
    text_fixed = extract_json_from_text(text)

    # Regex fallback: look anywhere in text for "positive ... 70" patterns
    # captures patterns like: positive: 70, positive 70%, "positive": "70", etc.

    if domain == "movie":
        Targets = ["Actor", "Storyline", "Visual", "Audio"]
    if domain == "music":
        Targets = ["Song", "Singer", "Lyrics", "Visual"]

    if text_fixed:

        if domain == "movie":
            fallback_matches = re.findall(
                r'(Actor|Storyline|Visual|Audio)[^0-9\-]*?([-+]?\d{1,3}(?:\.\d+)?)',
                text_fixed, flags=re.I
            )
        if domain == "music":
            fallback_matches = re.findall(
                r'(Song|Singer|Visual|Lyrics)[^0-9\-]*?([-+]?\d{1,3}(?:\.\d+)?)',
                text_fixed, flags=re.I
            )

    if text_fixed and fallback_matches:
        prob_map = {}
        used_stances = set()
        for kname, num in fallback_matches:
            key = kname.capitalize()
            if key not in used_stances:
                prob_map[key] = float(num)
                used_stances.add(key)

        for t in Targets:
            if t not in prob_map:
                prob_map[t] = 0.0

        # 1. Any value > 100
        if any(v > 100 for v in prob_map.values()):
            print(
                "⚠️ Warning: some probabilities > 100"
                f"text : {text_fixed}"
                f"Fallback matches: {fallback_matches}"
            )

        # 2. Total sum check (allow small rounding error)
        total = sum(prob_map.values())
        if abs(total - 100) > 0.1:
            print(
                f"Sum of probabilities = {total:.2f}, expected ~100. "
                f"text : {text_fixed}"
                f"Raw matches: {fallback_matches}"
            )
        return prob_map

    # 6) ultimate fallback: uniform distribution
    prob_map = {s:100/len(Targets) for s in Targets}
    print("-----------")
    print(f"text : {text_fixed}")
    print("P_t fallback uniform", prob_map)
    print("-----------")
    return prob_map


def _parse_P_s(text):
    # normalize smart quotes and remove control chars
    text_fixed = extract_json_from_text(text)

    # Regex fallback: look anywhere in text for "positive ... 70" patterns
    # captures patterns like: positive: 70, positive 70%, "positive": "70", etc.
    STANCES = ['positive', 'negative']
    if text_fixed:
        fallback_matches = re.findall(
            r'(positive|negative)[^0-9\-]*?([-+]?\d{1,3}(?:\.\d+)?)',
            text_fixed, flags=re.I
        )
    if text_fixed and fallback_matches:
        prob_map = {}
        # 只用第一個出現的值
        used_stances = set()
        for kname, num in fallback_matches:
            key = kname.lower()
            if key not in used_stances:
                prob_map[key] = float(num)
                used_stances.add(key)

        # 補上沒抓到的 stance 為 0
        for s in STANCES:
            if s not in prob_map:
                prob_map[s] = 0.0

        # 1. Any value > 100
        if any(v > 100 for v in prob_map.values()):
            print(
                "⚠️ Warning: some probabilities > 100"
                f"text : {text_fixed}"
                f"Fallback matches: {fallback_matches}"
            )

        # 2. Total sum check (allow small rounding error)
        total = sum(prob_map.values())
        if abs(total - 100) > 0.1:
            print(
                f"Sum of probabilities = {total:.2f}, expected ~100. "
                f"text : {text_fixed}"
                f"Raw matches: {fallback_matches}"
            )

        return prob_map


    # 6) ultimate fallback: uniform distribution

    prob_map = {s:100/len(STANCES) for s in STANCES}
    print("-----------")
    print(f"text : {text_fixed}")
    print("P_s fallback uniform", prob_map)
    print("-----------")
    return prob_map



def _parse_P_ts(text, domain):
    # normalize smart quotes and remove control chars
    text_fixed = extract_json_from_text(text)

    # Regex fallback: look anywhere in text for "positive ... 70" patterns
    # captures patterns like: positive: 70, positive 70%, "positive": "70", etc.

    if domain == "movie":
        Targets = ["Actor", "Storyline", "Visual", "Audio"]
    if domain == "music":
        Targets = ["Song", "Singer", "Lyrics", "Visual"]
    
    if text_fixed:
        pattern = rf'["\(]\s*({ "|".join(Targets) })\s*,\s*(positive|negative)\s*["\)]\s*["\']?\s*[:=]\s*["\']?([-+]?\d+(?:\.\d+)?)%?["\']?'
        fallback_matches = re.findall(pattern, text_fixed, flags=re.I)

    if text_fixed and fallback_matches:
        prob_map = {
            f"({t},{s})": float(v) 
            for t, s, v in fallback_matches
        }
        prob_map = {}
        used_keys = set()
        for t, s, v in fallback_matches:
            key = f"({t},{s})"
            if key not in used_keys:
                prob_map[key] = float(v) 
                used_keys.add(key)

        # Fill missing (Target, positive/negative) pairs with 0.0
        for t in Targets:
            for stance in ("positive", "negative"):
                key = f"({t},{stance})"
                if key not in prob_map:
                    prob_map[key] = 0.0

        # Sanity check for values > 100
        # 1. Any value > 100
        if any(v > 100 for v in prob_map.values()):
            print(
                "⚠️ Warning: some probabilities > 100\n"
                f"text : {text_fixed}\n"
                f"Fallback matches: {fallback_matches}"
            )

        # 2. Total sum check (allow small rounding error)
        total = sum(prob_map.values())
        if abs(total - 100) > 0.1:
            print(
                f"Sum of probabilities = {total:.2f}, expected ~100.\n"
                f"text : {text_fixed}\n"
                f"Raw matches: {fallback_matches}"
            )
        return prob_map
    
    # 6) ultimate fallback: uniform distribution
    prob_map = {f"({t},{s})": 100 / (len(Targets) * 2)
                for t in Targets for s in ("positive", "negative")}
    print("-----------")
    print(f"text : {text_fixed}")
    print("P_ts fallback uniform", prob_map)
    print("-----------")
    return prob_map

def QA_eval(item, task, domain):
    # parse prediction
    if task in ["P_s", "P_s_cond_t"]:
        pred_dist = _parse_P_s(item["response"])
    if task in ["P_t", "P_t_cond_s"]:
        pred_dist = _parse_P_t(item["response"], domain)
    if task == "P_ts":
        pred_dist = _parse_P_ts(item["response"], domain)
    
    # normalize
    total = sum(pred_dist.values())
    if total > 0:
        pred_dist = {k: v / total for k, v in pred_dist.items()}
    else:
        n = len(pred_dist)
        pred_dist = {k: 0 for k in pred_dist} if n > 0 else {}
    
    # ground truth normalization
    gt_dist = item["answer"]  # {"positive": 50.0, "negative": 50.0}
    gt_dist = {k: float(v)/100 for k, v in gt_dist.items()}

    all_keys = gt_dist.keys()

    # MAE + uniform MAE
    mae = 0
    mae_uniform = 0
    
    # --- NEW: TVD ---
    tvd = 0
    tvd_uniform = 0

    for k in all_keys:
        pv = pred_dist.get(k, 0.0)
        gv = gt_dist.get(k, 0.0)

        upv = 1 / len(all_keys)

        mae += abs(pv - gv)
        mae_uniform += abs(upv - gv)

        # TVD terms
        tvd += abs(pv - gv)
        tvd_uniform += abs(upv - gv)

    # normalize
    mae /= len(all_keys)
    mae_uniform /= len(all_keys)

    tvd /= 2
    tvd_uniform /= 2

    # write back
    item["extract_pred"] = pred_dist
    item["extract_answer"] = gt_dist
    item["mae"] = mae
    item["uniform_mae"] = mae_uniform

    ref_dist = item['answer']

    # --- NEW fields ---
    item["tvd"] = tvd
    item["uniform_tvd"] = tvd_uniform
    new = {
        "pred_dist": pred_dist, 
        "gt_dist": gt_dist,
        "mae": mae,
        "uniform_mae": mae_uniform,
        "tvd": tvd,
        "uniform_tvd": tvd_uniform,
        "normalized_entropy": normalized_entropy(list(ref_dist.values())),
        "normalized_majority_support_size": normalized_majority_support_size(list(ref_dist.values())),
        "top1_mass": top1_mass(list(ref_dist.values())),
        "top1_minus_top2_mass": top1_minus_top2_mass(list(ref_dist.values())),
        "js_between_uniform": js_between_uniform(list(ref_dist.values())),
        **item
    }

    return new



def main(pred_fp, output_fp, task, domain):
    pred_data = load_jsonl(pred_fp)
    evals = []
    for item in pred_data:
        eval_output = QA_eval(item, task, domain)
        evals.append(eval_output)

    save_jsonl(evals, output_fp)
    print(f"Eval results saved to {output_fp}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process answer and prediction files.")
    parser.add_argument(
        "--pred_fp", 
        type=str, 
        help="Directory of prediction JSONL file."
    )
    parser.add_argument(
        "--output_fp", 
        type=str, 
        help="Directory of eval result"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["P_s", "P_t", "P_s_cond_t", "P_t_cond_s", "P_ts"],
        help="Type of QA evaluation."
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["movie", "music"],
        help="Domain"
    )

    args = parser.parse_args()
    pred_fp = args.pred_fp
    output_fp = args.output_fp
    task = args.task
    domain = args.domain

    os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    main(pred_fp, output_fp, task, domain)
