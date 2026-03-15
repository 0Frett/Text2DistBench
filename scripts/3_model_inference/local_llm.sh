export HF_HOME="/data1/frett/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=0

models=(
    # 'Qwen/Qwen3-32B-FP8'
    # 'lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit'
    "Qwen/Qwen2.5-1.5B-Instruct"
)

domains=('movie')
# task_types=("P_s" "P_t" "P_s_cond_t" "P_t_cond_s" "P_ts")
task_types=("P_s")
qa_types=("estimation" "most_freq")
sample_sizes=(50)
ps=("posterior")
date="2025-12-01_2026-03-01"

for domain in "${domains[@]}"; do
for task in "${task_types[@]}"; do
for size in "${sample_sizes[@]}"; do
for model in "${models[@]}"; do
for p in "${ps[@]}"; do
for qa_type in "${qa_types[@]}"; do
    echo "Running: $model | $domain | $qa_type | $p | $task | $size"

    test_fp="data/${domain}/benchmark/${p}/${qa_type}/${date}/sampled_${size}/${task}.jsonl"
    out_fp="output/${domain}/${p}/${qa_type}/${date}/sampled_${size}/${task}/${model}.jsonl"

    if [[ "$p" == "prior" ]]; then
        out_fp="output/${domain}/${p}/${qa_type}/${date}/${task}/${model}.jsonl"
        test_fp="data/${domain}/benchmark/${p}/${qa_type}/${date}/${task}.jsonl"
    fi

    PYTHONPATH=lib python3 evaluation/6_local_inference.py \
        --model_id "$model" \
        --test_fp "$test_fp" \
        --output_fp "$out_fp" \
        --temperature 0.6 \
        --max_model_len 13000 \
        --tensor_parallel_size 1 \
        --max_output_tokens 8000 \
        --gpu_memory_utilization 0.95
done
done
done
done
done
done
