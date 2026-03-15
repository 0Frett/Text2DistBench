
#!/usr/bin/env bash
models=(
  "claude-sonnet-4-5"
#   "claude-3-haiku-20240307"
)
domains=('movie')
# task_types=("P_s" "P_t" "P_s_cond_t" "P_t_cond_s" "P_ts")
task_types=("P_s")
sample_sizes=(50)
qa_types=("estimation" "most_freq")
ps=("posterior")
date="2025-12-01_2026-03-01"

export date

parallel -j 10 --colsep ' ' '
  echo "Inference: {1} | {2} | {5} | {4} | {3} | sampled_{6}"
  PYTHONPATH=lib python3 evaluation/6_claude_inference.py \
    --model_id "{1}" \
    --test_fp "data/{2}/benchmark/{5}/{4}/${date}/sampled_{6}/{3}.jsonl" \
    --output_fp "output/{2}/{5}/{4}/${date}/sampled_{6}/{3}/{1}.jsonl"
' ::: "${models[@]}" ::: "${domains[@]}" ::: "${task_types[@]}" ::: "${qa_types[@]}" ::: "${ps[@]}" ::: "${sample_sizes[@]}"

