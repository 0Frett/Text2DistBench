
domain="movie"  # music

### Step 1. Collect valid entities
PYTHONPATH=lib python3 data_generation/entity_collection/${domain}/get_${domain}_pool.py \
  --start_str 2026-01-01 \
  --end_str 2026-04-01 \
  --output_dir data/${domain}/${domain}_pool \
  --max_entity 50

PYTHONPATH=lib python3 data_generation/entity_collection/${domain}/get_valid_${domain}.py \
  --entity_pool_file data/${domain}/${domain}_pool/2026-01-01_2026-04-01.json \
  --output_file data/${domain}/opinion_entity/2026-01-01_2026-04-01.json \
  --min_comments 500 \
  --max_entity 20


### Step 2. Download source documents
PYTHONPATH=lib python3 data_generation/1_collect_entity_docs.py \
  --domain ${domain} \
  --opinion_entity_file data/${domain}/opinion_entity/2026-01-01_2026-04-01.json \
  --output_dir data/${domain}/source_docs


### Step 3. Annotate comments
PYTHONPATH=lib python3 data_generation/2_annotate_comments.py \
  --domain ${domain} \
  --source_docs_dir data/${domain}/source_docs/2026-01-01_2026-04-01 \
  --output_dir data/${domain}/opinion_units


### Step 4. Stratified sampling comments (Optional):
PYTHONPATH=lib python3 data_generation/3_stratify_sample.py \
  --input_dir data/${domain}/opinion_units/2026-01-01_2026-04-01 \
  --output_dir data/${domain}/opinion_units/2026-01-01_2026-04-01/sampled_100 \
  --total_comments 100 \
  --seed 204

PYTHONPATH=lib python3 data_generation/3_stratify_sample.py \
  --input_dir data/${domain}/opinion_units/2026-01-01_2026-04-01 \
  --output_dir data/${domain}/opinion_units/2026-01-01_2026-04-01/sampled_50 \
  --total_comments 50 \
  --seed 204
