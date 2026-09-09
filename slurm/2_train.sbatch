#!/bin/bash
#SBATCH --job-name=segment-train
#SBATCH --gres=gpu:h100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#
# Qwen3.6-27B + rank-8 LoRA on the 128-frame SEGMENT mix.
# 8x H100 80GB, ~11.5 h for 2 epochs (1,466 steps at 23,454 rows / effective batch 32).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${PYTHON:?set PYTHON to the interpreter (see README)}"
: "${TORCHRUN:=$(dirname "$PYTHON")/torchrun}"
: "${HF_HOME:?set HF_HOME to the cache holding Qwen/Qwen3.6-27B}"
: "${EXPORT:=$ROOT/sft_export_128}"
: "${RUN_NAME:=segment-27b-8r-128f-2ep}"
: "${N_FRAMES:=128}"
: "${NPROC:=8}"
export HF_HOME HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p logs checkpoints

TRAIN="$EXPORT/train_${N_FRAMES}.jsonl"
EVAL="$EXPORT/eval.jsonl"
[[ -f "$TRAIN" ]] || { echo "FATAL: $TRAIN missing -- run slurm/1_prepare_data.sbatch"; exit 1; }
echo "training rows: $(wc -l < "$TRAIN")"
$PYTHON - "$TRAIN" "$N_FRAMES" <<'PY'
import json, sys
want = int(sys.argv[2])
n = {len(json.loads(l)["frames_indices"]) for l in open(sys.argv[1])}
if n != {want}:
    raise SystemExit(f"FATAL: clip lengths {sorted(n)}, expected exactly [{want}]")
print(f"verified: every row carries {want} frames")
PY

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PYTHON segment_track/preflight.py

$TORCHRUN --standalone --nproc_per_node="$NPROC" \
    segment_track/sft_train_qwen_segment_ddp.py \
    --train-file "$TRAIN" --eval-file "$EVAL" \
    --output-dir "$ROOT/checkpoints/$RUN_NAME" \
    --run-name "$RUN_NAME" \
    --epochs 2 --lr 1e-4 --seed 42 \
    --batch-size 1 --grad-accum 4 \
    --lora-r 8 --lora-alpha 16 \
    --eval-steps 366 --save-steps 366 --save-total-limit 10 \
    --early-stopping-patience 999
