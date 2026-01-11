# Text2DistBench 

## Installation
Clone the repository and install the required dependencies:
```
    pip install -r requirements.txt
    uv pip install vllm --torch-backend=auto
```


## Benchmark Generation
Text2DistBench is a fully automated, continuously-updated reading comprehension benchmark for evaluating LLMs’ ability to infer distributional knowledge from natural language. 


## Step 1. Collect Opinion Entities
```
PYTHONPATH=lib python3 data_generation/entity_collection/movie/get_movie_pool.py \
    --start_str 2025-07-01 \
    --end_str 2025-11-30 \
    --output_dir data/movie/movie_pool \
    --max_entity 10



PYTHONPATH=lib python3 data_generation/entity_collection/movie/get_valid_movie.py \
    --entity_pool data/movie/movie_pool/2025-07-01_2025-10-31 \
    --output_dir data/movie/opinion_entity \
    --min_comments 1000 \
    --max_entity 20 \
    --comment_lang en
```



## Step 2. Get Metadata and Viewer Comments
```
PYTHONPATH=lib python3 data_generation/1_gen_source_docs.py \
    --domain movie \
    --opinion_entity_dir data/movie/opinion_entity/2025-07-01_2025-10-31 \
    --output_dir data/movie/source_docs 
```



## Step 3. Comment Topic/Sentiment Annotation
```
PYTHONPATH=lib python3 data_generation/2_gen_op_units.py \
    --domain movie \
    --source_docs_dir data/movie/source_docs/2025-07-01_2025-10-31 \
    --output_dir data/movie/opinion_units

# stratify sample comments (if needed)
PYTHONPATH=lib python3 data_generation/3_stratify_sample.py \
  --input_dir data/music/opinion_units/2025-07-01_2025-10-31/en \
  --output_dir data/music/opinion_units/2025-07-01_2025-10-31/en/sampled_50 \
  --total 50 \
  --seed 42
```


## Step 4. QA Generation
qa_type : "P_s" "P_t" "P_s_cond_t" "P_t_cond_s" "P_ts"
```
# most-frequent task
PYTHONPATH=lib python3 data_generation/4_gen_most_bench.py \
    --domain movie \
    --op_units_dir data/movie/opinion_units/2025-07-01_2025-10-31/en/sampled_50 \
    --source_docs_dir data/movie/source_docs/2025-07-01_2025-10-31/en \
    --output_dir data/movie/benchmark/posterior/most/2025-07-01_2025-10-31/en/sampled_50 \
    --qa_type P_s

# second-frequent task
PYTHONPATH=lib python3 data_generation/4_gen_second_most_bench.py \
    --domain movie \
    --op_units_dir data/movie/opinion_units/2025-07-01_2025-10-31/en/sampled_50 \
    --source_docs_dir data/movie/source_docs/2025-07-01_2025-10-31/en \
    --output_dir data/movie/benchmark/posterior/most/2025-07-01_2025-10-31/en/sampled_50 \
    --qa_type P_s

# estimation task
PYTHONPATH=lib python3 data_generation/4_gen_estimation_bench.py \
    --domain movie \
    --op_units_dir data/movie/opinion_units/2025-07-01_2025-10-31/en/sampled_50 \
    --source_docs_dir data/movie/source_docs/2025-07-01_2025-10-31/en \
    --output_dir data/movie/benchmark/posterior/most/2025-07-01_2025-10-31/en/sampled_50 \
    --qa_type P_s
```

For the prior experiment in analysis section, run like the following example:
```
# most-frequent task (only metadata)
PYTHONPATH=lib python3 data_generation/4_gen_most_bench.py \
    --domain movie \
    --op_units_dir data/movie/opinion_units/2025-07-01_2025-10-31/en/sampled_50 \
    --source_docs_dir data/movie/source_docs/2025-07-01_2025-10-31/en \
    --output_dir data/movie/benchmark/prior/most/2025-07-01_2025-10-31/en/sampled_50 \
    --qa_type P_s \
    --prior
```


## Evaluation 
task_types : "P_s" "P_t" "P_s_cond_t" "P_t_cond_s" "P_ts"
```
# most-frequent & second-frequent
PYTHONPATH=lib python3 experiment/7_clf_eval.py \
    --task "$task" \
    --output_fp "$eval_output_fp" \
    --pred_fp "$model_pred_fp" \
    --domain "$domain"

# estimation
PYTHONPATH=lib python3 experiment/7_est_eval.py \
    --task "$task" \
    --output_fp "$eval_output_fp" \
    --pred_fp "$model_pred_fp" \
    --domain "$domain"
```



