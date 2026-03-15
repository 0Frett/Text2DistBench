#!/usr/bin/env python3
# hierarchical_stratified_sample.py
# 先按 target 比例分配，再在每個 target 內按 stance 比例分配
import json, math, random, argparse, os
from collections import defaultdict

def load_blocks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def hierarchy_map(blocks):
    """
    回傳:
      tgt2stance2comments: { target: { stance: [comments...] } }
    也回傳總數以備後續使用。
    """
    m = defaultdict(lambda: defaultdict(list))
    for blk in blocks:
        t = blk.get("target")
        s = blk.get("stance")
        for c in blk.get("comments", []):
            m[t][s].append(c)
    return m

def proportional_rounddown_then_fill(counts_dict, total_budget):
    """
    通用配額器：
      counts_dict: {key: original_count}
      total_budget: 這一層要分配的總額度
    規則：
      - 依比例 * total_budget 得到浮點數
      - 先 floor
      - 將剩餘名額依小數部分由大到小補齊
    回傳 {key: allocated_int}
    """
    grand = sum(counts_dict.values())
    if grand == 0 or total_budget <= 0:
        return {k: 0 for k in counts_dict}
    want_float = {k: counts_dict[k] / grand * total_budget for k in counts_dict}
    alloc = {k: math.floor(want_float[k]) for k in counts_dict}
    rem = total_budget - sum(alloc.values())
    if rem > 0:
        # 依小數部分排序補齊
        order = sorted(counts_dict.keys(),
                       key=lambda k: (want_float[k] - math.floor(want_float[k])),
                       reverse=True)
        for k in order[:rem]:
            alloc[k] += 1
    return alloc

def cap_by_availability(alloc, counts_dict):
    """避免超抽：若 alloc 超過該桶可用數量，設為該桶上限；釋出的名額回到呼叫端處理。"""
    new_alloc = {}
    overflow = 0
    for k, a in alloc.items():
        cap = counts_dict[k]
        if a > cap:
            new_alloc[k] = cap
            overflow += (a - cap)
        else:
            new_alloc[k] = a
    return new_alloc, overflow

def redistribute_overflow(alloc, counts_dict, overflow):
    """
    把溢出的名額分配給仍有空間的桶（依剩餘空間與比例潛力）。
    """
    if overflow <= 0:
        return alloc
    # 迭代補發直到耗盡或沒有可放的桶
    while overflow > 0:
        # 可接受的桶
        candidates = [k for k in counts_dict if alloc[k] < counts_dict[k]]
        if not candidates:
            break
        # 用剩餘空間排序（大的優先）
        candidates.sort(key=lambda k: (counts_dict[k] - alloc[k]), reverse=True)
        for k in candidates:
            if overflow == 0:
                break
            if alloc[k] < counts_dict[k]:
                alloc[k] += 1
                overflow -= 1
    return alloc

def hierarchical_sample(tgt2stance2comments, total_out, seed=42):
    random.seed(seed)

    # 原始總數
    tgt_counts = {t: sum(len(v) for v in tgt2stance2comments[t].values())
                  for t in tgt2stance2comments}
    orig_total = sum(tgt_counts.values())
    if total_out >= orig_total:
        # 不下采樣
        return tgt2stance2comments

    # 第一層：按 target 分配
    tgt_alloc = proportional_rounddown_then_fill(tgt_counts, total_out)
    # 如果某 target 本身可用數量少於分配額度，做上限截斷並回收剩餘名額
    capped, overflow = cap_by_availability(tgt_alloc, tgt_counts)
    tgt_alloc = redistribute_overflow(capped, tgt_counts, overflow)

    # 第二層：對每個 target，按 stance 分配
    result = defaultdict(lambda: defaultdict(list))
    for t in tgt2stance2comments:
        stance_counts = {s: len(tgt2stance2comments[t][s]) for s in tgt2stance2comments[t]}
        quota_t = tgt_alloc.get(t, 0)
        if quota_t == 0 or sum(stance_counts.values()) == 0:
            continue

        stance_alloc = proportional_rounddown_then_fill(stance_counts, quota_t)
        # 上限截斷 + 回收在該 target 內重分配
        stance_capped, s_overflow = cap_by_availability(stance_alloc, stance_counts)
        stance_alloc = redistribute_overflow(stance_capped, stance_counts, s_overflow)

        # 逐 stance 抽樣
        for s in stance_alloc:
            k = stance_alloc[s]
            if k <= 0:
                continue
            pool = tgt2stance2comments[t][s]
            if not pool:
                continue
            # 固定 seed 的無放回抽樣
            sampled = random.sample(pool, k) if k < len(pool) else list(pool)
            result[t][s].extend(sampled)

    return result

def to_blocks(tgt2stance2comments):
    out = []
    for t in sorted(tgt2stance2comments.keys()):
        for s in sorted(tgt2stance2comments[t].keys()):
            comments = tgt2stance2comments[t][s]
            if comments:
                out.append({"target": t, "stance": s, "comments": comments})
    return out

def main():
    ap = argparse.ArgumentParser(description="Hierarchical stratified downsampling (target -> stance).")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--total_comments", required=True, type=int)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for fn in os.listdir(args.input_dir):
        if not fn.endswith(".json"):
            continue
        input_file = os.path.join(args.input_dir, fn)
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, fn)

        blocks = load_blocks(input_file)
        tgt2stance2comments = hierarchy_map(blocks)

        sampled_hier = hierarchical_sample(tgt2stance2comments, total_out=args.total_comments, seed=args.seed)
        out_blocks = to_blocks(sampled_hier)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(out_blocks, f, ensure_ascii=False, indent=2)

        orig_total = sum(len(c) for t in tgt2stance2comments.values() for c in t.values())
        new_total  = sum(len(c) for t in sampled_hier.values()        for c in t.values())
        print(f"Input total: {orig_total} -> Output total: {new_total} (target={args.total_comments})")

        def show_dist(title, mapping):
            print(f"\n{title}")
            tgt_counts = {t: sum(len(v) for v in mapping[t].values()) for t in mapping}
            grand = sum(tgt_counts.values()) or 1
            for t in sorted(mapping.keys()):
                t_sum = sum(len(v) for v in mapping[t].values()) or 1
                tgt_pct = t_sum / grand
                stance_pct = {s: (len(mapping[t][s]) / t_sum) for s in mapping[t]}
                stance_str = ", ".join([f"{s}:{stance_pct[s]:.3f}" for s in sorted(mapping[t].keys())])
                print(f"- target={t:20s}  share={tgt_pct:.3f}  |  {stance_str}")

        show_dist("Original distribution (target share & stance share within target):", tgt2stance2comments)
        show_dist("Sampled distribution:", sampled_hier)

if __name__ == "__main__":
    main()
