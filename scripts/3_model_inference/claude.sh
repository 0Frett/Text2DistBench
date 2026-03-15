#!/usr/bin/env bash

models=(
  "claude-sonnet-4-5"
)

domains=("movie")
task_types=("P_s")
sample_sizes=(50)
qa_types=("estimation" "most_freq")
ps=("posterior")
date="2025-12-01_2026-03-01"

export date

parallel -j 10 --colsep ' ' '
model={1}
domain={2}
task={3}
qa_type={4}
p={5}
size={6}

if [[ "$p" == "prior" ]]; then
    test_fp="data/${domain}/benchmark/${p}/${qa_type}/${date}/${task}.jsonl"
    out_fp="output/${domain}/${p}/${qa_type}/${date}/${task}/${model}.jsonl"
else
    test_fp="data/${domain}/benchmark/${p}/${qa_type}/${date}/sampled_${size}/${task}.jsonl"
    out_fp="output/${domain}/${p}/${qa_type}/${date}/sampled_${size}/${task}/${model}.jsonl"
fi


echo "Inference: $model | $domain | $qa_type | $p | $task | sampled_$size"

PYTHONPATH=lib python3 evaluation/6_claude_inference.py \
    --model_id "$model" \
    --test_fp "$test_fp" \
    --output_fp "$out_fp"
' ::: "${models[@]}" ::: "${domains[@]}" ::: "${task_types[@]}" ::: "${qa_types[@]}" ::: "${ps[@]}" ::: "${sample_sizes[@]}"