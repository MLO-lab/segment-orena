"""Rebuild the exact hernia training rows from the released files alone.

Verifies that public/vqa/hernia_mesh.jsonl + public/frames/clips/ are sufficient: the
frame indices are not stored in the release because they are a function of the annotated
window, the frame count, and base_fps. This replays that function and emits rows in the
training format.

    python annotations/reconstruct_hernia_rows.py --n-frames 128 --out hernia_rows.jsonl

NOTE the released frames are the subset the questions reference, not the full 1 fps
extraction. Do NOT re-run the question generator against them: it derives its clamp from the
number of frames on disk, which differs for a subset. Use this script, which reads the
window from the released annotation instead.
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def indices(start: int, end: int, n_frames: int) -> list[int]:
    """base_fps is 1.0 for these videos, so an index is a second."""
    return [int(round(start + (end - start) * i / (n_frames - 1))) for i in range(n_frames)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=128)
    ap.add_argument("--frames", type=Path, default=HERE / "public/frames/clips")
    ap.add_argument("--vqa", type=Path, default=HERE / "public/vqa/hernia_mesh.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "hernia_rows.jsonl")
    a = ap.parse_args()

    n_ok = 0
    with a.out.open("w") as out:
        for line in a.vqa.open():
            q = json.loads(line)
            video = q["videoID"].split("/", 1)[1]
            idx = indices(q["start_time"], q["end_time"], a.n_frames)
            for i in idx:
                p = a.frames / video / f"frame{i:07d}.jpg"
                if not p.exists():
                    raise SystemExit(f"FATAL: {p} missing -- release is incomplete")
            out.write(json.dumps({
                "qID": q["qID"], "videoID": q["videoID"],
                "source_dataset": q["source_dataset"],
                "start_time": q["start_time"], "end_time": q["end_time"],
                "duration": q["end_time"] - q["start_time"], "base_fps": 1.0,
                "frame_dir": str(a.frames / video),
                "frames_indices": idx,
                "n_distinct_frames": len(set(idx)),
                "format": q["format"],
                "primary_capability": q["primary_capability"],
                "secondary_capabilities": q["secondary_capabilities"],
                "messages": [
                    {"role": "user", "content": [{"type": "video"},
                                                 {"type": "text", "text": q["question"]}]},
                    {"role": "assistant", "content": [{"type": "text", "text": q["answer"]}]},
                ],
            }) + "\n")
            n_ok += 1
    print(f"rebuilt {n_ok} rows -> {a.out}  (every referenced frame present)")


if __name__ == "__main__":
    main()
