"""Build the training set from the RELEASED annotations, without regenerating them.

This is the path for anyone who did not produce the annotations themselves. Given

  * `all_gold.jsonl` -- the official export, built by build_segment_sft_dataset.py from
    FOCUS data you already hold, at the frame count you intend to train on; and
  * the released VQA files under annotations/,

it emits the merged training file. Nothing is re-derived, so no raw annotation sheets, no
LapChole videos and no question generators are needed.

Two ways a released question acquires its clip:

  HeiCo / LapChole -- the question annotates a window that already exists in the official
    export, so the matching row's clip fields are copied verbatim. This is why those frames
    are never redistributed: you rebuild them from your own copy of FOCUS.

  hernia_yt -- there is no official row to inherit from, so indices are computed from the
    window. base_fps is 1.0 for these videos, so an index is a second, and the frames ship
    with the release.

    python segment_track/integrate_released_annotations.py \
        --export sft_export_128 --annotations annotations --n-frames 128

Do NOT re-run the question generators against the released hernia frames: they clamp the
window using the frame count on disk, and the release is the referenced subset rather than
the full extraction, which silently collapses every index onto the last frame.
"""
from __future__ import annotations
import argparse, json, random
from collections import Counter
from pathlib import Path


def hernia_indices(start: int, end: int, n_frames: int) -> list[int]:
    return [int(round(start + (end - start) * i / (n_frames - 1))) for i in range(n_frames)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True,
                    help="directory holding all_gold.jsonl")
    ap.add_argument("--annotations", type=Path, required=True,
                    help="the annotations/ directory from the release")
    ap.add_argument("--n-frames", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra-frames", type=Path, default=None,
                    help="directory of frames extracted for windows absent from the "
                         "official export (rows flagged requires_extraction)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    gold = [json.loads(l) for l in (a.export / "all_gold.jsonl").open()]
    by_win = {}
    for r in gold:
        by_win.setdefault((r["source_dataset"], r["videoID"].split("/", 1)[-1],
                           int(r["start_time"]), int(r["end_time"])), r)
    print(f"official export: {len(gold)} rows, {len(by_win)} distinct windows")

    vqa = sorted(a.annotations.rglob("vqa/*.jsonl"))
    if not vqa:
        raise SystemExit(f"no vqa/*.jsonl under {a.annotations}")
    rows, stats, unmatched = list(gold), Counter(), Counter()
    frames_root = a.annotations / "public/frames/clips"
    extra_frames = a.extra_frames

    for f in vqa:
        for line in f.open():
            q = json.loads(line)
            ds, video = q["source_dataset"], q["videoID"].split("/", 1)[-1]
            if ds == "hernia_yt":
                idx = hernia_indices(q["start_time"], q["end_time"], a.n_frames)
                d = frames_root / video
                if not d.is_dir():
                    unmatched[ds] += 1
                    continue
                clip = {"frame_dir": str(d), "frames_indices": idx, "base_fps": 1.0,
                        "n_distinct_frames": len(set(idx))}
            elif q.get("requires_extraction"):
                # window absent from the official export: the release carries the exact
                # indices, and the frames must be extracted from the source video
                # videoID carries the container extension; frame dirs do not
                stem = Path(video).stem if Path(video).suffix else video
                d = Path(extra_frames) / stem if extra_frames else None
                if d is None or not d.is_dir():
                    unmatched[f"{ds} (needs --extra-frames)"] += 1
                    continue
                idx = q["frames_indices"]
                clip = {"frame_dir": str(d), "frames_indices": idx,
                        "base_fps": q["base_fps"], "n_distinct_frames": len(set(idx))}
            else:
                src = by_win.get((ds, video, int(q["start_time"]), int(q["end_time"])))
                if src is None:
                    unmatched[ds] += 1
                    continue
                clip = {k: src[k] for k in ("frame_dir", "frames_indices", "base_fps",
                                            "n_distinct_frames")}
            rows.append({
                "qID": q["qID"], "uid": f"{ds}/{q['qID']}", "videoID": q["videoID"],
                "source_dataset": ds, "procedure_type": q.get("procedure_type"),
                "start_time": q["start_time"], "end_time": q["end_time"],
                "duration": q["end_time"] - q["start_time"],
                "format": q["format"], "primary_capability": q["primary_capability"],
                "secondary_capabilities": q.get("secondary_capabilities", []),
                "messages": [
                    {"role": "user", "content": [{"type": "video"},
                                                 {"type": "text", "text": q["question"]}]},
                    {"role": "assistant", "content": [{"type": "text", "text": q["answer"]}]},
                ], **clip,
            })
            stats[ds] += 1
        print(f"  {f.relative_to(a.annotations)}: +{sum(stats.values())} cumulative")

    for ds, n in stats.most_common():
        print(f"  merged {n:5d} from {ds}")
    for ds, n in unmatched.most_common():
        print(f"  SKIPPED {n} {ds} (no matching window in your export)")

    seen, uniq = set(), []
    for r in rows:
        k = (r["source_dataset"], r["videoID"], int(r["start_time"]), int(r["end_time"]),
             r["messages"][0]["content"][-1]["text"].strip())
        if k in seen:
            continue
        seen.add(k); uniq.append(r)
    random.Random(a.seed).shuffle(uniq)

    out = a.out or a.export / f"train_{a.n_frames}.jsonl"
    with out.open("w") as f:
        for r in uniq:
            f.write(json.dumps(r) + "\n")
    n = {len(r["frames_indices"]) for r in uniq}
    print(f"\n{len(uniq)} rows -> {out}")
    if n != {a.n_frames}:
        raise SystemExit(f"FATAL: clip lengths {sorted(n)}, expected exactly [{a.n_frames}]")
    print(f"verified: every row carries {a.n_frames} frames")


if __name__ == "__main__":
    main()
