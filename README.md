# Text2DistBench
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/frett/Text2DistBench)

Reference code for **Text2DistBench**, a fully automated reading comprehension benchmark for evaluating whether LLMs can infer distributional knowledge from natural language evidence.
Given metadata and a set of user comments about an entity (e.g., a movie or song), models must estimate statistics such as: stance distribution, topic distribution, and most/second-most frequent labels. The benchmark dataset is publicly available on Hugging Face and will be continuously updated.

## Overview

This repository provides two main workflows:

1. **Benchmark generation**
   - collect valid entities
   - download related documents (metadata, user comments)
   - annotate comments with topic / stance labels
   - generate QA instances (Tasks: Estimation, Most-Frequent, Second-Frequent)

2. **Model evaluation**
   - run model inference on generated benchmarks
   - evaluate question predictions



## Repository layout

```text
Text2DistBench/
├── data_generation/          # Benchmark construction pipeline
│   ├── entity_collection/    # Collect valid entities (movie / music)
│   ├── 1_collect_entity_docs.py
│   ├── 2_annotate_comments.py
│   ├── 3_stratify_sample.py
│   └── 4_gen_*_bench.py
├── evaluation/               # Inference + evaluation
│   ├── 6_*_inference.py
│   ├── 7_clf_eval.py         # Most- & Second- Frequent tasks
│   └── 7_est_eval.py         # Estimation task
├── lib/                      # API clients, prompts, utils
├── scripts/                  # End-to-end pipeline scripts
│   ├── 1_preprocess.sh
│   ├── 2_qa_generation.sh
│   ├── 3_model_inference/
│   └── 4_evaluation.sh
├── requirements.txt
└── .env.example
```



## Dataset Format
The generated benchmark instances are stored in JSONL format.
Each instance corresponds to a distributional reading comprehension question.
```
{
  "qid": < question id >,
  "qtype": < distribution type >,
  "answer": < answer (mode label or distribution depending on task) >,
  "ref_dist": < Ground-truth distribution over labels >,
  "question": < Full prompt shown to model (instruction + evidence + query) >,
  "source": "< entity name >",
  "meta_data": "< text evidence >",
  "comments": "< text evidence >",
  "condition": "< conditioning variable for P(s|t), P(t|s) >",
}
```



## Quick start

### 1. Install dependencies & configure API keys
Install dependencies (`pip install -r requirements.txt`), then copy `.env.example` to `.env` and add the API keys for the providers you plan to use.


### 2. Generate Text2DistBench
The `scripts/` directory contains example scripts for running the full benchmark generation pipeline.

```
bash scripts/1_preprocess.sh
bash scripts/2_qa_generation.sh
```
Before running the scripts, you may need to adjust benchmark configuration inside the script files (e.g., date, task, domain, sample size, ...).


### 3. Run model inference (optional)
Example scripts for model inference are provided in `scripts/3_model_inference/`:
```
bash scripts/3_model_inference/<model_script>.sh
```
You may also run your own inference pipeline as long as the output file follows the expected evaluation format.


### 4. Run evaluation
```
bash scripts/4_evaluation.sh
```
The evaluation scripts expect JSONL input, where each line corresponds to one benchmark instance with the following Minimal required format:
```
{
  "qid": "< question id >",
  "answer": "< ground truth answer >",
  "ref_dist": "< ground truth distribution >",
  "response": "< raw model response text >"
}
```
The evaluation scripts output a JSONL file containing instances of:
parsed model prediction, evaluation metrics (`1-tvd`, `correctness`), and statistics of the target distribution (e.g., `top1_mass`, `top2margin`, `js_between_uniform`, `support_size`).
