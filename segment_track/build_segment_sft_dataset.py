"""Builds the chat-formatted SFT dataset (train/eval/test) for the FOCUS SEGMENT track.

Mirrors `orena_sft/build_frame_sft_dataset.py`, with one structural difference: a
segment sample is a *clip*, so each record carries the frame indices spanning
`[start_time, end_time]` instead of a single image path.

Frames are referenced by directory + index list, never by expanded paths and never
as pixels -- N absolute paths per row (80 by default) would make the export ~10x
larger for information that `clip_sampling.frame_file()` reconstructs exactly.

`eval` is carved out of `train` at the *video* level, stratified by procedure_type.
Segment clips overlap heavily (consecutive rows are the same question over sliding
windows of one video), so a row-level split would leak almost completely.

Usage:
    .venv/bin/python segment_track/build_segment_sft_dataset.py --datasets heico lapchole
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orena_sft"))

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config  # noqa: E402
from focus.config import DATASET_BASE_FPS  # noqa: E402

from build_frame_sft_dataset import make_eval_video_split  # noqa: E402
from clip_sampling import (  # noqa: E402
    DEFAULT_N_FRAMES, frame_count, frame_dir, marker_times, sample_frame_indices,
)

DEFAULT_ROOT_DIR = Path(os.environ.get("ORENA_DATA_ROOT", "/projects/datasets_ML/orena/"))
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "sft_export"


def verify_fps(root_dir: Path, dataset: str, video_id: str, expected: float) -> float | None:
    """Read the real fps off the video file.

    The exported timestamps are computed as `frame_index / DATASET_BASE_FPS`. If a
    video's true rate were 29.97 while the constant says 30, every timestamp would
    drift ~3.6 s per hour -- past the `time` tolerance, and silently.
    """
    import decord

    path = root_dir / dataset / "videos" / video_id
    if not path.exists():
        return None
    vr = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    return float(vr.get_avg_fps())


def load_segment_records(dataset: str, split: DatasetSplit, root_dir: Path,
                         n_frames: int, frames_folder: str) -> list[dict]:
    base_fps = float(DATASET_BASE_FPS[dataset])
    ds = FocusDataset(dataset, split, Track.SEGMENT)

    records, dropped = [], 0
    for req, ref in ds:
        directory = frame_dir(root_dir, dataset, req.videoID, frames_folder)
        n_avail = frame_count(directory)
        if n_avail == 0:
            dropped += 1
            continue

        indices = sample_frame_indices(req.start_time, req.end_time, base_fps,
                                       n_frames=n_frames, n_available=n_avail)
        records.append({
            # qID is unique only WITHIN a dataset -- heico and lapchole each number
            # from 1, so a handful of ids collide across the merged export. Keep the
            # raw qID (the Evaluator matches references on it, per dataset) and add
            # a qualified uid for anything that pools the two.
            "uid": f"{dataset}/{req.qID}",
            "qID": req.qID,
            "source_dataset": dataset,
            # videoID is only unique within one dataset; qualify it so the
            # video-level split never merges same-named videos across datasets.
            "videoID": f"{dataset}/{req.videoID}",
            "procedure_type": req.procedure_type,
            "primary_capability": ref.primary.name,
            "secondary_capabilities": [c.name for c in ref.secondaries],
            "format": ref._format,
            "question": req.question,
            "answer": ref.answer,
            "start_time": req.start_time,
            "end_time": req.end_time,
            "duration": req.end_time - req.start_time,
            "base_fps": base_fps,
            "frame_dir": str(directory),
            "frames_indices": indices,
            "n_distinct_frames": len(set(indices)),
        })
    if dropped:
        print(f"  [{dataset}/{split.value}] dropped {dropped} rows with no frames on disk")
    return records


def to_chat_record(r: dict) -> dict:
    """One record for the trainer.

    `messages` holds only the text side; the clip enters as a bare
    `{"type": "video"}` placeholder that the chat template expands into
    `<|vision_start|><|video_pad|><|vision_end|>`. The collator fills that in from
    `frame_dir` + `frames_indices`, so pixels never touch the JSONL.
    """
    out = {k: r[k] for k in (
        "uid", "qID", "source_dataset", "videoID", "procedure_type", "primary_capability",
        "secondary_capabilities", "format", "start_time", "end_time", "duration",
        "base_fps", "frame_dir", "frames_indices", "n_distinct_frames",
    )}
    out["messages"] = [
        {"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": r["question"]},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": str(r["answer"])}]},
    ]
    return out


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(to_chat_record(r)) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["heico", "lapchole"],
                    choices=["heico", "lapchole"])
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--num-frames", type=int, default=DEFAULT_N_FRAMES,
                    help="frames per clip; must be even (temporal_patch_size=2)")
    ap.add_argument("--frames-folder", default="frames",
                    help="'frames' (default) or 'frames_overlay' for burned-in timestamps")
    ap.add_argument("--eval-frac", type=float, default=0.10,
                    help="fraction of TRAIN videos (per procedure_type) held out for eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-verify-fps", action="store_true",
                    help="skip checking each video's real fps against DATASET_BASE_FPS")
    args = ap.parse_args()

    if args.num_frames % 2:
        ap.error(f"--num-frames must be even, got {args.num_frames}")

    cfg = FocusConfig(root_dir=args.root_dir)
    set_config(cfg)

    train_records, test_records = [], []
    for dataset in args.datasets:
        print(f"Loading {dataset!r} segment track...")
        tr = load_segment_records(dataset, DatasetSplit.TRAIN, args.root_dir,
                                  args.num_frames, args.frames_folder)
        te = load_segment_records(dataset, DatasetSplit.TEST, args.root_dir,
                                  args.num_frames, args.frames_folder)
        print(f"  train: {len(tr)} rows   test: {len(te)} rows")
        train_records += tr
        test_records += te

    if not args.no_verify_fps:
        print("\nVerifying real fps against DATASET_BASE_FPS...")
        seen, bad = set(), []
        for r in train_records + test_records:
            key = r["videoID"]
            if key in seen:
                continue
            seen.add(key)
            dataset, vid = key.split("/", 1)
            actual = verify_fps(args.root_dir, dataset, vid, r["base_fps"])
            if actual is not None and abs(actual - r["base_fps"]) > 1e-6:
                bad.append((key, actual, r["base_fps"]))
        if bad:
            for key, actual, expected in bad:
                print(f"  MISMATCH {key}: real {actual} vs constant {expected}")
            raise SystemExit("fps mismatch would corrupt every exported timestamp; aborting.")
        print(f"  OK — {len(seen)} videos match their DATASET_BASE_FPS constant.")

    eval_videos = make_eval_video_split(train_records, args.eval_frac, args.seed)
    eval_records = [r for r in train_records if r["videoID"] in eval_videos]
    final_train = [r for r in train_records if r["videoID"] not in eval_videos]

    print(f"\nVideo-level eval split (eval_frac={args.eval_frac}): "
          f"{len(eval_videos)} eval videos, "
          f"{len({r['videoID'] for r in final_train})} train videos")
    print(f"  train: {len(final_train)} rows")
    print(f"  eval:  {len(eval_records)} rows")
    print(f"  test:  {len(test_records)} rows (untouched HF test split)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in (("train", final_train), ("eval", eval_records), ("test", test_records)):
        write_jsonl(recs, args.out_dir / f"{name}.jsonl")
    print(f"\nWrote train/eval/test.jsonl to {args.out_dir}/")

    sample = final_train[0]
    times = marker_times(sample["frames_indices"], sample["base_fps"])
    print(f"\nSanity sample — qID {sample['qID']} ({sample['format']}), "
          f"{sample['duration']:.0f}s clip, {len(sample['frames_indices'])} frames, "
          f"{len(times)} markers spanning {times[0]:.1f}s..{times[-1]:.1f}s")


if __name__ == "__main__":
    main()
