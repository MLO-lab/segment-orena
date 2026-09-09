#!/bin/bash
#SBATCH --job-name=segment-prep
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#
# Build the SEGMENT training set: 23,454 rows at 128 frames per clip.
# CPU only. Writes $EXPORT/train_128.jsonl.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${PYTHON:?set PYTHON to the interpreter (see README)}"
: "${ORENA_DATA_ROOT:?set ORENA_DATA_ROOT to the directory holding heico/ lapchole/ hernia_yt/}"
: "${ORENA_ANNOT_DIR:=$ROOT/annotations/private/raw}"
: "${LAPCHOLE_CSV_DIR:=$ROOT/annotations/private/raw/lapchole_annotations}"
: "${MESH_CSV_DIR:=$ROOT/annotations/public/raw/hernia_mesh_annotations}"
: "${ORENA_ANNOT_FRAMES:=$ROOT/annot_frames}"
: "${EXPORT:=$ROOT/sft_export_128}"
: "${N_FRAMES:=128}"
export ORENA_DATA_ROOT ORENA_ANNOT_DIR ORENA_ANNOT_FRAMES LAPCHOLE_CSV_DIR MESH_CSV_DIR
export SEGMENT_EXPORT_DIR="$EXPORT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p "$EXPORT" logs

echo "############ 1/5  official export at ${N_FRAMES} frames ############"
$PYTHON segment_track/build_segment_sft_dataset.py \
    --datasets heico lapchole \
    --root-dir "$ORENA_DATA_ROOT" \
    --n-frames "$N_FRAMES" \
    --out-dir "$EXPORT"
mv "$EXPORT/all.jsonl" "$EXPORT/all_gold.jsonl"
wc -l "$EXPORT/all_gold.jsonl"

echo; echo "############ 2/5  derived questions from labelled tracks ############"
$PYTHON -m segment_data.derive_segment --out "$EXPORT/segment_derived_all.jsonl"
wc -l "$EXPORT/segment_derived_all.jsonl"

echo; echo "############ 3/5  lapchole gallstone / AHA questions ############"
$PYTHON -m frame_data.lapchole_segment_questions \
    --csv-dir "$LAPCHOLE_CSV_DIR" \
    --n-frames "$N_FRAMES" --max-lastvis 250 --max-cooc 300 \
    --in-domain-negatives 300 --extract \
    --seg-all "$EXPORT/all_gold.jsonl" \
    --out "$EXPORT/lapchole_segment.jsonl"
wc -l "$EXPORT/lapchole_segment.jsonl"

echo; echo "############ 4/5  hernia / mesh questions ############"
$PYTHON -m frame_data.mesh_segment_questions \
    --csv-dir "$MESH_CSV_DIR" \
    --n-frames "$N_FRAMES" \
    --seg-all "$EXPORT/all_gold.jsonl" \
    --out "$EXPORT/mesh_segment.jsonl"
wc -l "$EXPORT/mesh_segment.jsonl"

echo; echo "############ 5/5  assemble train_128.jsonl ############"
cp "$EXPORT/all_gold.jsonl" "$EXPORT/all.jsonl"
$PYTHON segment_track/merge_generated_segment.py \
    --export "$EXPORT" --gen "$EXPORT/segment_derived_all.jsonl"
$PYTHON - "$EXPORT" "$N_FRAMES" <<'PY'
import json, random, sys
from pathlib import Path
d, want = Path(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(l) for l in (d / "all.jsonl").open()]
for extra in ("mesh_segment.jsonl", "lapchole_segment.jsonl"):
    rows += [json.loads(l) for l in (d / extra).open()]
seen, uniq = set(), []
for r in rows:
    k = (r["source_dataset"], r["videoID"], int(r["start_time"]), int(r["end_time"]),
         r["messages"][0]["content"][-1]["text"].strip())
    if k in seen:
        continue
    seen.add(k); uniq.append(r)
random.Random(42).shuffle(uniq)
out = d / f"train_{want}.jsonl"
with out.open("w") as f:
    for r in uniq:
        f.write(json.dumps(r) + "\n")
n = {len(r["frames_indices"]) for r in uniq}
print(f"{len(uniq)} rows -> {out}")
if n != {want}:
    raise SystemExit(f"FATAL: clip lengths {sorted(n)}, expected exactly [{want}]")
print(f"verified: every row carries {want} frames")
PY
echo "=== prep done ==="
