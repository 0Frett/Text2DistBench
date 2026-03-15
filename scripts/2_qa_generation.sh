
set -euo pipefail

DOMAIN="movie"
DATE_RANGE="2025-12-01_2026-03-01"
SAMPLE_TAG="sampled_50"

QA_TYPES=("P_s" "P_t" "P_s_cond_t" "P_t_cond_s" "P_ts")

OP_UNITS_DIR="data/${DOMAIN}/opinion_units/${DATE_RANGE}/${SAMPLE_TAG}"
SOURCE_DOCS_DIR="data/${DOMAIN}/source_docs/${DATE_RANGE}"

SCRIPTS=(
  "4_gen_estimation_bench.py:estimation"
  "4_gen_most_freq_bench.py:most_freq"
  "4_gen_second_freq_bench.py:second_freq"
)

# posterior (metadata + comments)
for entry in "${SCRIPTS[@]}"; do
    IFS=":" read -r script_name task_name <<< "$entry"

    for qa_type in "${QA_TYPES[@]}"; do
        PYTHONPATH=lib python3 "data_generation/${script_name}" \
            --domain "${DOMAIN}" \
            --op_units_dir "${OP_UNITS_DIR}" \
            --source_docs_dir "${SOURCE_DOCS_DIR}" \
            --output_dir "data/${DOMAIN}/benchmark/posterior/${task_name}/${DATE_RANGE}/${SAMPLE_TAG}" \
            --qa_type "${qa_type}"
    done
done

# prior (metadata only)
for entry in "${SCRIPTS[@]}"; do
    IFS=":" read -r script_name task_name <<< "$entry"

    for qa_type in "${QA_TYPES[@]}"; do
        PYTHONPATH=lib python3 "data_generation/${script_name}" \
            --domain "${DOMAIN}" \
            --op_units_dir "${OP_UNITS_DIR}" \
            --source_docs_dir "${SOURCE_DOCS_DIR}" \
            --output_dir "data/${DOMAIN}/benchmark/prior/${task_name}/${DATE_RANGE}/${SAMPLE_TAG}" \
            --qa_type "${qa_type}" \
            --prior
    done
done