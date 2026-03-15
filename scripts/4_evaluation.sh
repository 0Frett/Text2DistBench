

#!/usr/bin/env bash
set -euo pipefail

models=(
    "gpt-5.1"
    "gemini-2.5-pro"
    "grok-4-fast-reasoning"
    "claude-sonnet-4-5"
    # "Qwen/Qwen2.5-1.5B-Instruct"
)
domains=("movie")
task_types=("P_s")
sample_sizes=(50)
qa_types=("estimation")
ps=("posterior" "prior")

DATE_RANGE="2025-12-01_2026-03-01"

for model in "${models[@]}"; do
  for domain in "${domains[@]}"; do
    for task in "${task_types[@]}"; do
      for size in "${sample_sizes[@]}"; do
        for p in "${ps[@]}"; do
          for qa_type in "${qa_types[@]}"; do
            echo "Evaluating: $model | $domain | $qa_type | $p | $task | $size"

            out_fp="eval/${domain}/${p}/${qa_type}/${DATE_RANGE}/sampled_${size}/${task}/${model}.jsonl"
            pred_fp="output/${domain}/${p}/${qa_type}/${DATE_RANGE}/sampled_${size}/${task}/${model}.jsonl"

            if [[ "$p" == "prior" ]]; then
              out_fp="eval/${domain}/${p}/${qa_type}/${DATE_RANGE}/${task}/${model}.jsonl"
              pred_fp="output/${domain}/${p}/${qa_type}/${DATE_RANGE}/${task}/${model}.jsonl"
            fi

            if [[ "$qa_type" == "estimation" ]]; then
              PYTHONPATH=lib python3 evaluation/7_est_eval.py \
                --task "$task" \
                --output_fp "$out_fp" \
                --pred_fp "$pred_fp" \
                --domain "$domain"

            elif [[ "$qa_type" == "most_freq" || "$qa_type" == "second_freq" ]]; then
              PYTHONPATH=lib python3 evaluation/7_clf_eval.py \
                --task "$task" \
                --output_fp "$out_fp" \
                --pred_fp "$pred_fp" \
                --domain "$domain"
                
            else
              echo "Unknown qa_type: $qa_type"
              exit 1

            fi
          done
        done
      done
    done
  done
done

