

domains=('movie')
# task_types=("P_s" "P_t" "P_s_cond_t" "P_t_cond_s" "P_ts")
task_types=("P_s")
sample_sizes=(50)
qa_types=("most_freq" "estimation")
ps=("posterior")
date="2025-12-01_2026-03-01"

for domain in "${domains[@]}"; do
for task in "${task_types[@]}"; do
for size in "${sample_sizes[@]}"; do
for p in "${ps[@]}"; do
for qa_type in "${qa_types[@]}"; do
    echo "Running: gpt-4.1 | $domain | $qa_type | $p | $task | $size"

    test_fp="data/${domain}/benchmark/${p}/${qa_type}/${date}/sampled_${size}/${task}.jsonl"
    out_dir="output/${domain}/${p}/${qa_type}/${date}/sampled_${size}/${task}"

    if [[ "$p" == "prior" ]]; then
        test_fp="data/${domain}/benchmark/${p}/${qa_type}/${date}/${task}.jsonl"
        out_dir="output/${domain}/${p}/${qa_type}/${date}/${task}"
    fi

    echo "Running: gpt-4.1 | $domain | $qa_type | $p | $task | $size"
    PYTHONPATH=lib python3 evaluation/6_gpt_inference.py \
        --model_id "gpt-4.1" \
        --test_fp "$test_fp" \
        --output_fp "${out_dir}/gpt-4.1.jsonl"

    echo "Running: gpt-5.1 | $domain | $qa_type | $p | $task | $size"
    PYTHONPATH=lib python3 evaluation/6_gpt_inference.py \
        --model_id "gpt-5.1" \
        --test_fp "$test_fp" \
        --output_fp "${out_dir}/gpt-5.1.jsonl" \
        --effort "medium" \
        --reason "True"


done
done
done
done
done
