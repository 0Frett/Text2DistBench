
models=("grok-4-fast-reasoning")
domains=('movie')
# task_types=("P_s" "P_t" "P_s_cond_t" "P_t_cond_s" "P_ts")
task_types=("P_s")
sample_sizes=(50)
qa_types=("most_freq")
ps=("posterior")
date="2025-12-01_2026-03-01"

for model in "${models[@]}"; do
for domain in "${domains[@]}"; do
for task in "${task_types[@]}"; do
for size in "${sample_sizes[@]}"; do
for p in "${ps[@]}"; do
for qa_type in "${qa_types[@]}"; do
    echo "Running: $model | $domain | $qa_type | $p | $task | $size"

    PYTHONPATH=lib python3 evaluation/6_grok_inference.py \
        --model_id "$model" \
        --test_fp "data/${domain}/benchmark/${p}/${qa_type}/${date}/sampled_${size}/${task}.jsonl" \
        --output_fp "output/${domain}/${p}/${qa_type}/${date}/sampled_${size}/${task}/${model}.jsonl"

done
done
done
done
done
done
