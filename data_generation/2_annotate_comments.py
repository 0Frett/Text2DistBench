import json
import argparse
import os
import re
from typing import Any, Dict, List, Tuple
from collections import Counter
from collections import defaultdict
from openai_client import OpenAIModel
from google_client import GeminiModel
from grok_client import GrokModel
from prompts import movie_datagen_prompts, music_datagen_prompts


# STANCE_KEYS = ["support","neutral","oppose"]
STANCE_KEYS = ["support","oppose"]


def _strip_to_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    m = list(re.finditer(r"\{.*\}", text, flags=re.DOTALL))
    return m[-1].group(0) if m else text

def _parse_json(text: str) -> Any:
    try:
        return json.loads(_strip_to_json(text))
    except Exception:
        return None

def _block_with_local_indices(comments: List[str], start_global: int) -> Tuple[str, Dict[int,int]]:
    lines, local2global = [], {}
    for i, c in enumerate(comments):
        local2global[i] = start_global + i
        lines.append(f"{i}. {c}")
    return "\n".join(lines), local2global

def _call_models_once(models: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    out = {}
    for name, mdl in models.items():
        try:
            print("-"*50)
            # print(prompt)
            resp = mdl.annot_generate(prompt=prompt, num_return_sequences=1)
            text = resp.text[0] if hasattr(resp, "text") and isinstance(resp.text, list) else str(resp)
            out[name] = _parse_json(text)
            print(name)
            print(out[name])
            print("-"*50)
        except Exception as e:
            print(f"[WARN] {name} inference error: {e}")
            out[name] = None
    return out


def _majority_single_topic(per_model_json: Dict[str, Dict[str, List[int]]],
                           local_idx: int,
                           attrs_topic: List[str],
                           allow_other: bool = False,
                           k: int = 2) -> str:

    votes = []
    for _, js in per_model_json.items():
        if not isinstance(js, dict):
            votes.append(None); continue
        hits = [a for a in attrs_topic if local_idx in (js.get(a) or [])]
        if len(hits) == 1:
            lab = hits[0]
            if (lab == "Other") and not allow_other:
                votes.append(None)
            else:
                votes.append(lab)
        else:
            votes.append(None)  # 多標或未標
    c = Counter(votes)  # 包含 None
    # 去除 None
    if None in c: del c[None]
    if not c: return ''
    top_lab, top_cnt = c.most_common(1)[0]
    return top_lab if top_cnt >= k else ''

def _majority_stance(per_model_json: Dict[str, Dict[str, List[int]]],
                     local_idx: int,
                     k: int = 2) -> str:

    candidates = ["support","oppose"]
    votes = []
    for _, js in per_model_json.items():
        if not isinstance(js, dict):
            votes.append(None); continue
        hits = [lab for lab in candidates if local_idx in (js.get(lab) or [])]
        votes.append(hits[0] if len(hits) == 1 else None)
    c = Counter(votes)
    if None in c: del c[None]
    if not c: return ''
    top_lab, top_cnt = c.most_common(1)[0]
    return top_lab if top_cnt >= k else ''

# ---------- process() using PROMPT_TEMPLATE.ATTRS_TOPIC ----------


def process(train_docs, models, PROMPT_TEMPLATE, batch_size=100):
    meta = train_docs["meta_data"]
    comments = train_docs["comments"]

    TOPIC_TMPL  = PROMPT_TEMPLATE.TOPIC_CLF_TEMPLATE
    STANCE_TMPL = PROMPT_TEMPLATE.STANCE_CLF_TEMPLATE
    attrs_topic = PROMPT_TEMPLATE.ATTRS_TOPIC

    # ---- Stage 1: topics ----
    kept_by_target_global: Dict[str, List[int]] = {a: [] for a in attrs_topic if a != "Other"}

    for start in range(0, len(comments), batch_size):
        batch = comments[start:start + batch_size]
        block, l2g = _block_with_local_indices(batch, start_global=start)
        topic_prompt = TOPIC_TMPL.format(meta_data=meta, comments=block)

        print(f"\n[STAGE 1] Batch {start}-{start+len(batch)} | Sending to {len(models)} models")
        topic_jsons = _call_models_once(models, topic_prompt)

        # quick summary of per-model results
        for name, js in topic_jsons.items():
            if isinstance(js, dict):
                summary = {k: len(v) if isinstance(v, list) else 0 for k, v in js.items()}
                print(f"  ↳ {name} topic summary: {summary}")
            else:
                print(f"  ↳ {name} returned invalid JSON.")

        for li in l2g.keys():
            majority = _majority_single_topic(topic_jsons, li, attrs_topic, allow_other=False, k=2)
            if majority and majority in kept_by_target_global:
                kept_by_target_global[majority].append(l2g[li])

    print("\n[STAGE 1 DONE] Comments kept by topic:")
    for target, ids in kept_by_target_global.items():
        print(f"  {target:<20} → {len(ids)} comments")

    if not any(kept_by_target_global[t] for t in kept_by_target_global):
        print("[WARN] No comments survived Stage 1. Returning empty list.")
        return []

    # ---- Stage 2: stance ----
    results: List[Dict[str, Any]] = []
    print("\n[STAGE 2] Beginning stance classification...")

    for target, gidxs in kept_by_target_global.items():
        if not gidxs:
            continue
        print(f"\n  Target = '{target}' | {len(gidxs)} comments to process")

        for s in range(0, len(gidxs), batch_size):
            chunk_g = gidxs[s:s + batch_size]
            chunk_comments = [comments[g] for g in chunk_g]
            stance_block, l2g_chunk = _block_with_local_indices(chunk_comments, start_global=0)

            stance_prompt = STANCE_TMPL.format(meta_data=meta, comments=stance_block)
            stance_jsons = _call_models_once(models, stance_prompt)

            # quick summary
            for name, js in stance_jsons.items():
                if isinstance(js, dict):
                    summary = {k: len(v) if isinstance(v, list) else 0 for k, v in js.items()}
                    print(f"    ↳ {name} stance summary: {summary}")
                else:
                    print(f"    ↳ {name} returned invalid JSON.")

            by_stance_locals = {k: [] for k in STANCE_KEYS}
            for li in l2g_chunk.keys():
                us = _majority_stance(stance_jsons, li, k=2)
                if us in STANCE_KEYS:
                    by_stance_locals[us].append(li)

            for stance_label in STANCE_KEYS:
                locs = by_stance_locals[stance_label]
                if not locs:
                    continue
                results.append({
                    "target": target,
                    "stance": stance_label,
                    "comments": [chunk_comments[li] for li in locs]
                })
                print(f"    ✅ {target} / {stance_label}: {len(locs)} unanimous comments")
    # ---- Merge duplicate (target, stance) entries ----
    merged = defaultdict(lambda: {"target": None, "stance": None, "comments": []})
    for r in results:
        key = (r["target"], r["stance"])
        merged[key]["target"] = r["target"]
        merged[key]["stance"] = r["stance"]
        merged[key]["comments"].extend(r["comments"])

    results = list(merged.values())
    
    print(f"\n[STAGE 2 DONE] Total opinion units formed: {len(results)}\n")
    return results



# ---------- main ----------

def main(source_docs_dir, output_dir, domain):
    # Load domain-specific prompts
    if domain == "movie":
        PROMPT_TEMPLATE = movie_datagen_prompts
    elif domain == "music":
        PROMPT_TEMPLATE = music_datagen_prompts
    else:
        raise ValueError("domain must be 'movie' or 'music'")

    models = {
        'gpt':    OpenAIModel('gpt-4o-mini', temperature=0.5, max_tokens=10000),
        'gemini': GeminiModel(model="gemini-2.5-flash-lite"),
        'grok':   GrokModel(model="grok-4-fast-non-reasoning")
    }

    time_stamp = os.path.basename(source_docs_dir)
    save_dir = os.path.join(output_dir, time_stamp)
    os.makedirs(save_dir, exist_ok=True)


    for doc_fn in os.listdir(source_docs_dir):
        with open(os.path.join(source_docs_dir, doc_fn), "r", encoding="utf-8") as f:
            train_docs = json.load(f)

        output_path = os.path.join(save_dir, doc_fn)

        if os.path.exists(output_path):
            print(f"Skipping: {output_path}. Already processed.")
            continue
        print(f"\nProcessing: {output_path}")

        video_opinions = process(train_docs, models, PROMPT_TEMPLATE)

        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(video_opinions, out_f, ensure_ascii=False, indent=2)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="movie", choices=["movie", "music"])
    parser.add_argument("--source_docs_dir", type=str, default="data/movie/source_docs/2025-07-01_2025-09-30")
    parser.add_argument("--output_dir", type=str, default="data/movie/opinion_units")
    args = parser.parse_args()

    main(
        source_docs_dir=args.source_docs_dir,
        output_dir=args.output_dir,
        domain=args.domain
    )
