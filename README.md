# Text2DistBench

Reference code for **Text2DistBench**, a fully automated benchmark for evaluating whether LLMs can infer distributional knowledge from natural language evidence.

---

## Repository layout

```text
Text2DistBench/
├── data_generation/          # Benchmark construction pipeline
│   ├── entity_collection/    # Collect valid entities (movie / music)
│   ├── 1_collect_entity_docs.py
│   ├── 2_annotate_comments.py
│   ├── 3_stratify_sample.py
│   └── 4_gen_*_bench.py
├── evaluation/               # Inference + evaluation scripts
│   ├── 6_*_inference.py
│   ├── 7_clf_eval.py         # Most- & Second- Frequent tasks
│   └── 7_est_eval.py         # Estimation task
├── lib/                      # API clients, prompts, utils
├── scripts/                  # Example commands
├── requirements.txt
└── .env.example
```

---

## Quick start

### Install dependencies & configure API keys
Install dependencies (`pip install -r requirements.txt`), then copy `.env.example` to `.env` and add the API keys for the providers you plan to use.


This repo supports two main workflows:
1. **Benchmark generation**: collect valid entities, download related documents (e.g., metadata, user comments), annotate comments (topic/stance), and generate QA instances.
2. **Model evaluation**: run model inference on generated benchmarks and score predictions.


