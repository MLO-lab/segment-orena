"""Pre-flight check before committing a multi-day training run.

Everything here is cheap and CPU-only. It exists because the expensive failures on
this pipeline are all silent: a missing kernel makes training 4x slower, an old
Triton computes wrong gradients, a stale export mismatches the sampler, and a
misconfigured save/eval cadence disables early stopping. None of those announce
themselves -- you find out hours in, or not at all.

    .venv/bin/python        segment_track/preflight.py   # expect kernel FAILs
    venv3.12/bin/python     segment_track/preflight.py   # the real check
"""

from __future__ import annotations

import json
import shutil
import sys
from importlib.util import find_spec
from pathlib import Path

SEG = Path(__file__).resolve().parent
sys.path.insert(0, str(SEG))
sys.path.insert(0, str(SEG.parent / "orena_sft"))

FAIL, WARN = [], []


def check(label: str, ok: bool, detail: str = "", warn_only: bool = False) -> bool:
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{tag}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        (WARN if warn_only else FAIL).append(label)
    return ok


print("1. environment / fused kernels")
import torch  # noqa: E402

print(f"      python {sys.version.split()[0]} | torch {torch.__version__}")
check("flash-linear-attention installed", find_spec("fla") is not None,
      "without it the 48 linear-attention layers run a Python loop (4x slower)")
try:
    import triton

    v = tuple(int(x) for x in triton.__version__.split(".")[:3])
    check("triton >= 3.7.1", v >= (3, 7, 1),
          f"{triton.__version__} — below 3.7.1 computes WRONG gradients on Hopper (fla #640)")
except Exception as e:  # noqa: BLE001
    check("triton importable", False, str(e))
import transformers  # noqa: E402

check("transformers >= 5.9", tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (5, 9),
      transformers.__version__)
for mod in ("trl", "peft", "focus", "wandb"):
    check(f"{mod} importable", find_spec(mod) is not None)
import importlib.metadata as md  # noqa: E402

check("orena-focus version recorded", True, md.version("orena-focus"))

print("\n2. model weights present locally")
from huggingface_hub import snapshot_download  # noqa: E402

try:
    p = snapshot_download("Qwen/Qwen3.6-27B", local_files_only=True)
    n_shards = len(list(Path(p).glob("*.safetensors")))
    size = sum(f.stat().st_size for f in Path(p).glob("*.safetensors")) / 1e9
    check("Qwen3.6-27B weights cached", n_shards > 0 and size > 40,
          f"{n_shards} shards, {size:.0f} GB")
except Exception as e:  # noqa: BLE001
    check("Qwen3.6-27B weights cached", False, f"not downloaded ({type(e).__name__})")

print("\n3. dataset export")
from clip_sampling import DEFAULT_N_FRAMES, frame_file  # noqa: E402

splits = {}
for name in ("train", "eval", "test"):
    p = SEG / "sft_export" / f"{name}.jsonl"
    if not p.exists():
        check(f"{name}.jsonl exists", False, str(p))
        continue
    splits[name] = [json.loads(l) for l in p.open()]
    check(f"{name}.jsonl exists", True, f"{len(splits[name])} rows")

if len(splits) == 3:
    n_frames = {len(r["frames_indices"]) for s in splits.values() for r in s}
    check("frame count is uniform", len(n_frames) == 1, str(n_frames))
    check(f"export matches DEFAULT_N_FRAMES ({DEFAULT_N_FRAMES})",
          n_frames == {DEFAULT_N_FRAMES},
          f"export has {n_frames}; re-export or the sampler and data disagree")
    check("frame count is even (temporal_patch_size=2)", all(n % 2 == 0 for n in n_frames))
    vtr = {r["videoID"] for r in splits["train"]}
    check("train/eval video-disjoint", not (vtr & {r["videoID"] for r in splits["eval"]}))
    check("train/test video-disjoint", not (vtr & {r["videoID"] for r in splits["test"]}))
    sample = splits["train"][0]
    check("frames resolve on disk",
          frame_file(sample["frame_dir"], sample["frames_indices"][0]).exists(),
          sample["frame_dir"])

print("\n4. training configuration")
from sft_train_qwen_segment_ddp import build_parser  # noqa: E402

a = vars(build_parser().parse_args([]))
es, ss = a["eval_steps"], a["save_steps"]
check("save_steps is a multiple of eval_steps", ss % es == 0,
      f"save={ss} eval={es} — required for load_best_model_at_end")
check("early stopping enabled", a["early_stopping_patience"] is not None,
      f"patience={a['early_stopping_patience']} evals "
      f"(= {a['early_stopping_patience'] * es} steps without eval_loss improvement)")
check("checkpoint cadence", ss == 50, f"every {ss} steps")
check("gradient checkpointing on", not a["no_gradient_checkpointing"],
      "required — it OOMs without (measured)")
check("LoRA r/alpha", (a["lora_r"], a["lora_alpha"]) == (8, 16),
      f"r={a['lora_r']} alpha={a['lora_alpha']}")
check("effective batch 32 on 2 GPUs", a["batch_size"] * a["grad_accum"] * 2 == 32,
      f"{a['batch_size']}x{a['grad_accum']}x2")
check("prompt style is 'direct'", a["prompt_style"] == "direct")

print("\n5. cost projection")
if "train" in splits:
    steps_per_epoch = len(splits["train"]) / 32
    step_s = 80.0            # measured at N=80, batch 1, 2xH200
    eval_n = a["eval_subset"] or len(splits.get("eval", []))
    eval_s = eval_n / 2 * 2.9
    n_evals = steps_per_epoch * a["epochs"] / es
    train_h = steps_per_epoch * a["epochs"] * step_s / 3600
    eval_h = n_evals * eval_s / 3600
    print(f"      {steps_per_epoch:.0f} steps/epoch x {a['epochs']} epochs @ {step_s:.0f}s")
    print(f"      eval: {eval_n} clips x {n_evals:.0f} evals @ {eval_s/60:.0f} min")
    check("fits the 2-day SLURM limit", train_h + eval_h < 48,
          f"train {train_h:.1f}h + eval {eval_h:.1f}h = {train_h+eval_h:.1f}h")
    check("eval overhead is reasonable", eval_h / max(train_h, 1e-9) < 0.25,
          f"{100*eval_h/max(train_h,1e-9):.0f}% of training time", warn_only=True)
    written = steps_per_epoch * a["epochs"] / ss
    limit = a["save_total_limit"]
    n_ckpt = written if limit is None else min(limit, written)   # None = keep all
    gb = n_ckpt * 0.464                                          # measured size per checkpoint
    free = shutil.disk_usage(SEG).free / 1e9
    check("disk space for checkpoints", free > gb + 20,
          f"{free:.0f} GB free, need ~{gb:.0f} GB for {n_ckpt:.0f} checkpoints "
          f"({'keeping all' if limit is None else f'limit {limit}'})")

print("\n6. prompt / sampler agreement")
from prompts import build_system_prompt  # noqa: E402

sp = build_system_prompt(style="direct", track="segment")
check("segment prompt builds", len(sp) > 1000, f"{len(sp)} chars")
check("prompt documents the timestamp convention", "<SECONDS seconds>" in sp)
check("prompt covers time + percentage", "hh:mm:ss" in sp and "percentage" in sp)

print("\n" + "=" * 70)
if FAIL:
    print(f"NOT READY — {len(FAIL)} blocking issue(s):")
    for f in FAIL:
        print(f"   - {f}")
elif WARN:
    print(f"READY (with {len(WARN)} warning(s)):")
    for w in WARN:
        print(f"   - {w}")
else:
    print("READY — all checks passed.")
sys.exit(1 if FAIL else 0)
