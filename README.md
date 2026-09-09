# MLO-Lab — ORena FOCUS 2026, SEGMENT track

Training code for the SEGMENT submission: **Qwen3.6-27B with a rank-8 LoRA adapter**,
trained on 23,454 examples at 128 frames per clip.

## Layout

```
annotations/       our frame-level annotations + the VQA pairs generated from them
frame_data/        question generation from foreign-object annotations
segment_data/      question derivation from already-labelled object tracks
orena_sft/         shared modules (system prompt, frame-track dataset builder)
segment_track/     SEGMENT dataset builder, collator, trainer, preflight
slurm/             the two jobs that run the pipeline
```

The annotation data is not committed. The public half is a Hugging Face dataset:

**https://huggingface.co/datasets/Machine-Learning-Oncology/orena-segment-annotations**

```bash
python annotations/fetch.py     # HeiCo + hernia questions, sheets, and hernia frames
```

The LapChole half is **not hosted**: its data usage agreement forbids publishing any
derivative until the organisers release LapChole-FOCUS. It goes directly to the organisers
and is **available on request** thereafter. Unpack it into `annotations/private/` and the
pipeline picks it up automatically.

| you have | rows you can rebuild |
|---|---:|
| public only | 21,440 |
| + private annotations | 23,294 |
| + LapChole frame extraction (`--extra-frames`) | **23,454** (exact) |

The official 20,000 are always present; the differences are the 1,440 public and 2,014
LapChole questions we added, of which 160 also need frames extracted from LapChole videos
because their windows are absent from the official export.

See `annotations/README.md` for the schemas and the reasoning behind the split.

## Requirements

* Python 3.12, 8× H100 80 GB (single node) for training; CPU only for data preparation.
* `pip install -r requirements.txt` — install `torch` from the CUDA 12.8 index first.
* `Qwen/Qwen3.6-27B` cached in `$HF_HOME` (~56 GB). Training runs with `HF_HUB_OFFLINE=1`.
* The FOCUS dataset root, containing `heico/`, `lapchole/` and `hernia_yt/`.
* Foreign-object annotation CSVs (`lapchole_annotations/`, hernia sheets).

## Configuration

Both jobs are configured through environment variables. No paths are hardcoded.

| variable | required | meaning |
|---|---|---|
| `PYTHON` | yes | interpreter of the environment built from `requirements.txt` |
| `ORENA_DATA_ROOT` | yes | directory holding `heico/`, `lapchole/`, `hernia_yt/` |
| `HF_HOME` | training | cache containing `Qwen/Qwen3.6-27B` |
| `ORENA_ANNOT_DIR` | no | directory holding `crop_boxes.json` (default `annotations/private/raw`) |
| `LAPCHOLE_CSV_DIR` | no | LapChole sheets (default `annotations/private/raw/lapchole_annotations`) |
| `MESH_CSV_DIR` | no | hernia sheets (default `annotations/public/raw/hernia_mesh_annotations`) |
| `ORENA_ANNOT_FRAMES` | prep | where extracted annotation frames are written |
| `EXPORT` | no | dataset output directory (default `./sft_export_128`) |
| `N_FRAMES` | no | frames per clip (default `128`, must be even) |
| `RUN_NAME` | no | checkpoint directory name |
| `NPROC` | no | GPUs per node (default `8`) |

## 1. Build the dataset

There are two routes. **Use route A unless you are regenerating the annotations from raw
sheets** — it needs no annotation sheets, no LapChole videos and no question generators.

### Route A — integrate the released annotations (recommended)

```bash
export PYTHON=/path/to/venv/bin/python
export ORENA_DATA_ROOT=/path/to/orena

python annotations/fetch.py --private        # or without --private

# 1. build the official export at 128 frames from YOUR copy of FOCUS
$PYTHON segment_track/build_segment_sft_dataset.py \
    --datasets heico lapchole --root-dir "$ORENA_DATA_ROOT" \
    --n-frames 128 --out-dir sft_export_128
mv sft_export_128/all.jsonl sft_export_128/all_gold.jsonl

# 2. merge the released questions into it
$PYTHON segment_track/integrate_released_annotations.py \
    --export sft_export_128 --annotations annotations --n-frames 128
```

HeiCo and LapChole questions inherit their clip from the matching window in *your* export,
which is why those frames are never redistributed. Hernia questions compute indices from the
window and use the frames that ship with the release. 160 LapChole questions annotate windows
absent from the official export; they carry explicit `frames_indices` and need
`--extra-frames <dir>` pointing at frames extracted from the LapChole videos, or they are
skipped with a warning.

### Route B — regenerate from the raw annotation sheets

```bash
export PYTHON=/path/to/venv/bin/python
export ORENA_DATA_ROOT=/path/to/orena

sbatch slurm/1_prepare_data.sbatch
```

Runs five steps and writes `$EXPORT/train_128.jsonl`:

| step | output | rows |
|---|---|---:|
| 1 | official export rebuilt at 128 frames → `all_gold.jsonl` | 20,000 |
| 2 | derived from labelled tracks → `segment_derived_all.jsonl` | 4,202 |
| 3 | lapchole gallstone / AHA → `lapchole_segment.jsonl` | 627 |
| 4 | hernia / mesh → `mesh_segment.jsonl` | 440 |
| 5 | merge, deduplicate, shuffle → **`train_128.jsonl`** | **23,454** |

Step 5 asserts that every row carries exactly `N_FRAMES` frames and fails otherwise.

Individual generators can also be run directly:

```bash
$PYTHON -m segment_data.derive_segment --out segment_derived_all.jsonl
$PYTHON -m frame_data.lapchole_segment_questions --n-frames 128 --extract
$PYTHON -m frame_data.mesh_segment_questions --n-frames 128
$PYTHON segment_track/build_segment_sft_dataset.py --datasets heico lapchole --n-frames 128
```

## 2. Train

```bash
export PYTHON=/path/to/venv/bin/python
export HF_HOME=/path/to/hf_cache

sbatch slurm/2_train.sbatch
```

| | |
|---|---|
| base model | `Qwen/Qwen3.6-27B` |
| adapter | LoRA r=8, α=16, dropout 0.05 |
| target modules | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` (vision tower frozen) |
| batch | per-device 1 × grad-accum 4 × 8 GPUs = effective 32 |
| schedule | lr 1e-4, bf16, seed 42, 2 epochs = 1,466 steps |
| checkpoints | every 366 steps → `checkpoints/$RUN_NAME/` |
| runtime | ~11 h 30 m; 73.8 GB peak allocated per GPU |

The job verifies the training file exists and that every clip has `N_FRAMES` frames, then
runs `segment_track/preflight.py`, which gates `triton>=3.7.1` and `flash-linear-attention`.

The submitted model is `checkpoint-1466` (epoch 2).

## 3. Package for submission

Merge the adapter into the base model, then build the container. `N_FRAMES` in the
container's frame sampler must equal the value used for training.

## Notes

* `N_FRAMES` must be even — the model fuses frames pairwise (`temporal_patch_size=2`).
* Frames are sampled on a fixed 15 fps grid, so changing `N_FRAMES` needs no re-extraction.
* `segment_track/build_segment_sft_dataset.py` defaults `--root-dir` to a path that may not
  exist on your cluster; set `ORENA_DATA_ROOT` or pass `--root-dir` explicitly.
