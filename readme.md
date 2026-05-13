## Download

1. CORE4D and/or InterHuman motion files
2. local instruction LLM (only if not using openai API)

## Install

```bash
conda env create -f environment.yml
conda activate collage-motion
pip install -e .
```

Manual install:

```bash
conda create -n collage-motion python=3.10 -y
conda activate collage-motion
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

Optional 4-bit LLM loading:

```bash
pip install bitsandbytes
```

## Download motion data

Downloads may not work (rate limits etc..), in that case download from official dources manually and place in the diretories manually.

```bash
python tools/download_official_data.py --dataset links
```

Download CORE4D:

```bash
python tools/download_official_data.py --dataset core4d --out-root /data
```

Download InterHuman:

```bash
python tools/download_official_data.py --dataset interhuman --out-root /data
```


## OpenAI API setup for GPT planning cues

The OpenAI endpoint is configured through environment variables. edit it:

```bash
cp configs/openai_api.example.env .env
nano .env
```

Example:

```bash
OPENAI_API_KEY=replace_with_your_key
OPENAI_BASE_URL=https://api.openai.com/v1
COLLAGE_LLM_MODEL=gpt-4o
```

Load it before generating cues:

```bash
set -a
source .env
set +a
```

Generate six-level cues with the API backend:

```bash
python tools/generate_llm_cues.py \
  --backend openai \
  --input /data/collage_captions.csv \
  --output data/plans/core4d_llm_cues.jsonl \
  --openai-model "$COLLAGE_LLM_MODEL" \
  --api-key-env OPENAI_API_KEY \
  --base-url-env OPENAI_BASE_URL \
  --levels 6 \
  --resume
```



Full download notes are in `download.md` if doing non API based generation.

#### Download LLM

```bash
python tools/download_llm.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --out-dir llm_models/qwen2_5_7b_instruct \
  --verify-tokenizer
```

Shell command:

```bash
bash scripts/download_llm_qwen.sh
```

## Generate collage LLM cues from caption 

Caption CSV from CORE4D:

```csv
motion_number,Description
2763,"Two humans are picking up the object together."
```

Generate cues:

```bash
python tools/generate_llm_cues.py \
  --input /data/collage_captions.csv \
  --output data/plans/core4d_llm_cues.jsonl \
  --model-path llm_models/qwen2_5_7b_instruct \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --levels 6 \
  --resume
```

4-bit cue generation:

```bash
python tools/generate_llm_cues.py \
  --input /data/collage_captions.csv \
  --output data/plans/core4d_llm_cues.jsonl \
  --model-path llm_models/qwen2_5_7b_instruct \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --levels 6 \
  --load-in-4bit \
  --resume
```

## CORE4D collage pipeline

```bash
python tools/preprocess_core4d.py \
  --config configs/core4d_preprocess.yaml \
  --raw-root /data/CORE4D \
  --caption-csv /data/collage_captions.csv \
  --out-dir /data/collage_processed/core4d

python tools/compute_stats.py \
  --data-root /data/collage_processed/core4d \
  --out-path /data/collage_processed/core4d/stats.npz

python tools/train_vqvae.py \
  --config configs/train_vqvae_core4d.yaml \
  --data-root /data/collage_processed/core4d \
  --stats-path /data/collage_processed/core4d/stats.npz \
  --plan-cache data/plans/core4d_llm_cues.jsonl

python tools/train_association.py \
  --config configs/train_assoc_core4d.yaml \
  --data-root /data/collage_processed/core4d \
  --stats-path /data/collage_processed/core4d/stats.npz \
  --vqvae-ckpt runs/core4d_vqvae/checkpoints/best.pt \
  --plan-cache data/plans/core4d_llm_cues.jsonl

python tools/train_diffusion.py \
  --config configs/train_diffusion_core4d.yaml \
  --data-root /data/collage_processed/core4d \
  --stats-path /data/collage_processed/core4d/stats.npz \
  --vqvae-ckpt runs/core4d_vqvae/checkpoints/best.pt \
  --assoc-ckpt runs/core4d_assoc/checkpoints/best.pt \
  --plan-cache data/plans/core4d_llm_cues.jsonl

python tools/infer.py \
  --config configs/infer_core4d.yaml \
  --caption "Two humans pick up a chair together and carry it forward." \
  --out-dir results/collage_core4d \
  --plan-cache data/plans/core4d_llm_cues.jsonl
```

End-to-end CORE4D command:

```bash
CORE4D_ROOT=/data/CORE4D \
COLLAGE_CAPTIONS=/data/collage_captions.csv \
PROCESSED_ROOT=/data/collage_processed/core4d \
PLAN_CACHE=data/plans/core4d_llm_cues.jsonl \
bash scripts/run_core4d_full.sh
```

## InterHuman collage pipeline

```bash
python tools/preprocess_interhuman.py \
  --config configs/interhuman_preprocess.yaml \
  --raw-root /data/InterHuman \
  --out-dir /data/collage_processed/interhuman

python tools/generate_llm_cues.py \
  --input /data/collage_processed/interhuman/metadata.jsonl \
  --output data/plans/interhuman_llm_cues.jsonl \
  --model-path llm_models/qwen2_5_7b_instruct \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --levels 6 \
  --resume

python tools/compute_stats.py \
  --data-root /data/collage_processed/interhuman \
  --out-path /data/collage_processed/interhuman/stats.npz

python tools/train_vqvae.py \
  --config configs/train_vqvae_interhuman.yaml \
  --data-root /data/collage_processed/interhuman \
  --stats-path /data/collage_processed/interhuman/stats.npz \
  --plan-cache data/plans/interhuman_llm_cues.jsonl

python tools/train_association.py \
  --config configs/train_assoc_interhuman.yaml \
  --data-root /data/collage_processed/interhuman \
  --stats-path /data/collage_processed/interhuman/stats.npz \
  --vqvae-ckpt runs/interhuman_vqvae/checkpoints/best.pt \
  --plan-cache data/plans/interhuman_llm_cues.jsonl

python tools/train_diffusion.py \
  --config configs/train_diffusion_interhuman.yaml \
  --data-root /data/collage_processed/interhuman \
  --stats-path /data/collage_processed/interhuman/stats.npz \
  --vqvae-ckpt runs/interhuman_vqvae/checkpoints/best.pt \
  --assoc-ckpt runs/interhuman_assoc/checkpoints/best.pt \
  --plan-cache data/plans/interhuman_llm_cues.jsonl

python tools/infer.py \
  --config configs/infer_interhuman.yaml \
  --caption "Two people are dancing together." \
  --out-dir results/collage_interhuman \
  --plan-cache data/plans/interhuman_llm_cues.jsonl
```

End-to-end InterHuman command:

```bash
INTERHUMAN_ROOT=/data/InterHuman \
PROCESSED_ROOT=/data/collage_processed/interhuman \
PLAN_CACHE=data/plans/interhuman_llm_cues.jsonl \
bash scripts/run_interhuman_full.sh
```

## Inference cue cache for new prompts(local non-api)

```bash
cat > prompts.csv <<'CSV'
motion_number,Description
new_0001,"Two humans lift a heavy box together, walk forward, and place it on a table."
CSV

python tools/generate_llm_cues.py \
  --input prompts.csv \
  --output data/plans/inference_llm_cues.jsonl \
  --model-path llm_models/qwen2_5_7b_instruct \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --levels 6 \
  --resume

python tools/infer.py \
  --config configs/infer_core4d.yaml \
  --caption "Two humans lift a heavy box together, walk forward, and place it on a table." \
  --out-dir results/collage_box_lift \
  --plan-cache data/plans/inference_llm_cues.jsonl
```

## Evaluation and rendering

```bash
python tools/evaluate.py \
  --generated-dir results/collage_core4d \
  --reference-root /data/collage_processed/core4d \
  --out-json results/collage_eval_core4d.json \
  --out-csv results/collage_eval_core4d.csv

python tools/render_npz.py \
  --input results/collage_core4d \
  --out-dir results/collage_renders
```

---
