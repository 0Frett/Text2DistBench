import os
import json
import argparse
import random
from typing import Dict, List, Any, Tuple
import pandas as pd
from collections import defaultdict

from prompts import movie_posteriorQ_prompts as movie_postQ
from prompts import movie_priorQ_prompts as movie_priorQ
from prompts import music_posteriorQ_prompts as music_postQ
from prompts import music_priorQ_prompts as music_priorQ

STANCES = ["support", "oppose"]

STANCE_MAP = {
    "support": "positive",
    "oppose": "negative",
}


def _make_prompt(system_prompt: str, question: str) -> str:
    return system_prompt.rstrip() + "\n\n" + question.lstrip()


def make_qa(qtype: str, question: str, answer: Dict[str, Any], ref_dist: Dict[str, Any],
            attribute: str, video_title: str, idx: int) -> Dict[str, Any]:
    return {
        "qid": f"most_{video_title}_{qtype}_{idx}",
        # "attribute": attribute,  # "stance"/"target"/specific target/stance/"joint"
        "answer": answer,
        "ref_dist": ref_dist,
        "qtype": qtype,          # P_s | P_t | P_s_cond_t | P_t_cond_s | P_ts
        "source": video_title,
        "question": question,
    }


# ------------------------
# ingest stats (counts-only OR p_* CSV)
# ------------------------
def _load_counts(op_units:List[Dict[str, Any]], target_set) -> pd.DataFrame:
    """
    Return a (target, stance, count) dataframe and total N.
    """
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for op in op_units:
        target = op.get("target")
        stance = op.get("stance")
        n = len(op.get("comments"))
        if target not in target_set:
            print(f"Skipping target '{target}' not in target_set")
            continue
        if stance not in STANCES:
            print(f"Skipping stance '{stance}' not in STANCES")
            continue
        counts[(target, stance)] += n

    for t in target_set:
        for s in STANCES:
            counts[(t, s)] += 0
    
    rows = []
    for (t, s), cnt in counts.items():
        rows.append({"target":t, "stance":s, "count":cnt})

    return pd.DataFrame(rows)



def _marginals_and_conditionals(counts_df: pd.DataFrame,
                                target_set: List[str]) -> Dict[str, Any]:
    """
    Compute P(S), P(T), P(S|T), P(T|S), P(S,T) from counts.
    Returns dict with keys: pS, pT, pS_given_T, pT_given_S, pST
    """
    df = counts_df.copy()
    N = float(df["count"].sum())

    # P(S)
    pS = (df.groupby("stance")["count"].sum() / N).to_dict()
    # P(T)
    pT = (df.groupby("target")["count"].sum() / N).to_dict()

    # P(S|T)
    pS_given_T = {}
    for t, g in df.groupby("target"):
        denom = float(g["count"].sum())
        if denom > 0:
            d = (g.set_index("stance")["count"] / denom).to_dict()
        else:
            d = {s: 0.0 for s in STANCES}
        pS_given_T[t] = d

    # P(T|S)
    pT_given_S = {}
    for s, g in df.groupby("stance"):
        denom = float(g["count"].sum())
        if denom > 0:
            d = (g.set_index("target")["count"] / denom).to_dict()
        else:
            d = {t: 0.0 for t in target_set}
        pT_given_S[s] = d

    # P(S,T)
    pST = {}
    for (t, s), c in df.groupby(["target", "stance"])["count"].sum().items():
        pST.setdefault(t, {})[s] = float(c) / N

    return dict(pS=pS, pT=pT, pS_given_T=pS_given_T, pT_given_S=pT_given_S, pST=pST)


# ------------------------
# QA generators (use stats)
# ------------------------
def gen_pred_dist_question(
    video_title: str,
    meta_data: str,
    comments_block: str,
    op_units: List[Dict[str, Any]],
    post_template,
    prior_template,
    qa_type: str,
    is_prior: bool) -> List[Dict[str, Any]]:
    
    """
    Produces QA items matching the 5 templates.
    """
    if is_prior:
        sys_prompt = prior_template.SYS_TEMPLATE.format(meta_data=meta_data)
    else:
        sys_prompt = post_template.SYS_TEMPLATE.format(meta_data=meta_data, comments=comments_block)


    # Targets ordering: prefer template.TARGETS if present
    target_set = getattr(post_template, "TARGETS", None)
    if not target_set:
        raise ValueError("template.TARGETS must be defined.")
    
    # Get counts 
    counts_df = _load_counts(op_units, target_set)

    # Compute distributions
    dists = _marginals_and_conditionals(counts_df, target_set)
    pS = dists["pS"]
    pT = dists["pT"]
    pS_given_T = dists["pS_given_T"]
    pT_given_S = dists["pT_given_S"]
    pST = dists["pST"]

    qa_sets: List[Dict[str, Any]] = []

    # --- P(S): single QA item ---
    if qa_type == "P_s":
        if is_prior:
            question = prior_template.MOSTFREQ_S_TEMPLATE
        else:
            question = post_template.MOSTFREQ_S_TEMPLATE
        ref_dist = {STANCE_MAP[s]: 100*pS.get(s, 0.0) for s in STANCES}
        max_value = max(ref_dist.values())
        answer = [k for k, v in ref_dist.items() if v == max_value]
        if max_value != 0:
            qa_sets.append(make_qa("P_s", _make_prompt(sys_prompt, question), answer, ref_dist, "margin_s", video_title, 0))

    # --- P(T): single QA item ---
    elif qa_type == "P_t":
        if is_prior:
            question = prior_template.MOSTFREQ_T_TEMPLATE
        else:
            question = post_template.MOSTFREQ_T_TEMPLATE
        ref_dist = {t: 100*pT.get(t, 0.0) for t in target_set}
        max_value = max(ref_dist.values())
        answer = [k for k, v in ref_dist.items() if v == max_value]
        if max_value != 0:
            qa_sets.append(make_qa("P_t", _make_prompt(sys_prompt, question), answer, ref_dist, "margin_t", video_title, 0))

    # --- P(S|T): one item per target ---
    elif qa_type == "P_s_cond_t":
        for i, tgt in enumerate(target_set):
            if is_prior:
                question = prior_template.MOSTFREQ_S_cond_T_TEMPLATE.format(topic=tgt)
            else:
                question = post_template.MOSTFREQ_S_cond_T_TEMPLATE.format(topic=tgt)
            ref_dist = {STANCE_MAP[s]: 100*pS_given_T.get(tgt, {}).get(s, 0.0) for s in STANCES}
            max_value = max(ref_dist.values())
            answer = [k for k, v in ref_dist.items() if v == max_value]
            if max_value != 0:
                qa_sets.append(make_qa("P_s_cond_t", _make_prompt(sys_prompt, question), answer, ref_dist, tgt, video_title, i))

    # --- P(T|S): one item per stance ---
    elif qa_type == "P_t_cond_s":
        for i, stance in enumerate(STANCES):
            if is_prior:
                question = prior_template.MOSTFREQ_T_cond_S_TEMPLATE.format(stance_label=STANCE_MAP[stance])
            else:
                question = post_template.MOSTFREQ_T_cond_S_TEMPLATE.format(stance_label=STANCE_MAP[stance])
            ref_dist = {t: 100*pT_given_S.get(stance, {}).get(t, 0.0) for t in target_set}
            max_value = max(ref_dist.values())
            answer = [k for k, v in ref_dist.items() if v == max_value]
            if max_value != 0:
                qa_sets.append(make_qa("P_t_cond_s", _make_prompt(sys_prompt, question), answer, ref_dist, stance, video_title, i))

    # --- P(S,T): single joint item ---
    elif qa_type == "P_ts":
        if is_prior:
            question = prior_template.MOSTFREQ_T_S_TEMPLATE
        else:
            question = post_template.MOSTFREQ_T_S_TEMPLATE
        ref_dist = {f"({t},{STANCE_MAP[s]})": 100*pST.get(t, {}).get(s, 0.0) for t in target_set for s in STANCES}
        max_value = max(ref_dist.values())
        answer = [k for k, v in ref_dist.items() if v == max_value]
        if max_value != 0:
            qa_sets.append(make_qa("P_ts", _make_prompt(sys_prompt, question), answer, ref_dist, "joint", video_title, 0))

    else:
        raise ValueError("Wrong QA Type")

    return qa_sets


# ------------------------
# IO + main
# ------------------------
def save_jsonl(data, save_dir, base_name):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{base_name}.jsonl")
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} items to: {file_path}")


def main(
    domain: str,
    source_docs_dir: str,
    op_units_dir: str,   # to pull raw comments for the prompt context
    output_dir: str,
    qa_type: str,
    seed: int,
    is_prior: bool,
):
    # Select domain-specific template
    if domain == "movie":
        post_template = movie_postQ
        prior_template = movie_priorQ
    elif domain == "music":
        post_template = music_postQ
        prior_template = music_priorQ
    else:
        raise ValueError("domain must be 'movie' or 'music'")

    test_qs: List[Dict[str, Any]] = []

    for unit_fp in os.listdir(op_units_dir):
        if not unit_fp.endswith(".json"):
            continue

        unit_name = os.path.splitext(unit_fp)[0]
        print(f"Processing {unit_name}...")

        # Load metadata + raw comments (for system context)
        with open(os.path.join(source_docs_dir, f"{unit_name}.json"), "r", encoding="utf-8") as f:
            doc_data = json.load(f)
        with open(os.path.join(op_units_dir, f"{unit_name}.json"), "r", encoding="utf-8") as f:
            op_units = json.load(f)

        # Load stats (counts-only or p_joint)
        meta_data = doc_data["meta_data"]
        # Flatten comments (0-based index) and shuffle for context variety
        comments = [c for op in op_units for c in op.get("comments", [])]
        random.seed(seed)
        random.shuffle(comments)
        comments_str = "\n".join([f"{idx+1}. {c}" for idx, c in enumerate(comments)])

        qas = gen_pred_dist_question(
            unit_name, meta_data, comments_str, op_units, 
            post_template, prior_template, qa_type, is_prior)
        test_qs.extend(qas)

    print(f"[INFO] {len(test_qs)} test items generated.")
    save_jsonl(test_qs, output_dir, qa_type)
    print(f"[END] Finish Generating {qa_type} Most-Freq Benchmark to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, choices=["movie", "music"])
    parser.add_argument("--op_units_dir", type=str,
                        default="data/movie/opinion_units/2025-07-01_2025-09-30/en/sampled_250")
    parser.add_argument("--output_dir", type=str,
                        default="data/movie/benchmark/2025-07-01_2025-09-30/en/sampled_250")
    parser.add_argument("--source_docs_dir", type=str,
                        default="data/movie/source_docs/2025-01-01_2025-08-31/en")
    parser.add_argument("--qa_type", type=str,
                        choices=["P_s", "P_t", "P_s_cond_t", "P_t_cond_s", "P_ts"])
    parser.add_argument("--seed", type=int, default=204, help="Random seed for shuffling comments in context.")
    parser.add_argument("--prior", action="store_true")
    args = parser.parse_args()

    main(
        args.domain,
        args.source_docs_dir,
        args.op_units_dir,
        args.output_dir,
        args.qa_type,
        args.seed,
        args.prior
    )
