"""SEGMENT-track questions from the Job-1 lapchole pass, using only what is directly observed.

Most of SEGMENT is out of reach for a 10 s annotation grid: `time` (38% of the track) is
scored against a 1.3-4.3 s threshold, and `fo_class` (25%) needs a complete class set the
optional checklist never gave us. Interpolating between agreeing samples does not rescue it
either -- measured, two samples that agree still disagree with the sample between them 24% of
the time at a 20 s span and 45% at 40 s.

What survives needs no inference at all, and it comes from choosing where the clip *ends*:

* **last-visible quadrant.** Make the clip end on an annotated frame where the object is
  present. Then "the last time it is visible" is the final frame of the clip, and the answer
  is the quadrant we observed directly. Not a trick: of the gold clips with a KB annotation
  near their end, 163/248 (66%) also have the object still visible at the boundary. This is
  the largest SEGMENT multiple-choice template (458 gold examples).
* **co-occurrence.** A checklist tick is a positive observation, so a window containing a
  frame where two classes were both seen answers "yes" outright.

The same construction would also unlock the `time` format -- "at what time was a <C> last
visible" is exactly the clip end, no inference at all, and `time` is 38% of the track. It is
deliberately NOT emitted. Gold answers that template at the clip boundary only **10.5%** of
the time (923 questions: 97 at the end, 786 mid-clip), so every row we could make would say
`end_time` against a gold set that almost never does, and the model would learn to read the
boundary instead of finding the disappearance. The same measurement rules out the "first
visible" mirror even more strongly (0.3% at the start).

For the same reason the quadrant version is capped rather than shipped in full -- see
`LASTVIS_CAP`.

Windows are shared between questions wherever possible -- each one costs 80 extracted frames.

    python -m frame_data.lapchole_segment_questions --csv-dir <dir> --extract
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
import os
from pathlib import Path

from frame_data.lapchole_questions import (ANNOT, BASE_FPS, GALL, AHA, OUT_FRAMES, PLURAL,
                                           TARGETS, VIDEOS, secs, uncrop)

N_FRAMES = 80                       # default; --n-frames overrides (128 for the
                                    # retrain that fixes the time-threshold sampling limit)

# Frames are extracted on a FIXED grid over each window, not at the exact indices one
# sampling happens to want. The first version wrote only the 80 indices per window, which
# made the dataset single-use: changing N meant decoding everything again. At 15 fps any
# N up to 15*window_seconds is free, and the snap error is <= 1/(2*15) = 0.033 s -- inside
# the slack left by the 1.322 s time threshold (1.177 s sampling + 0.080 s reconstruction
# leaves 0.065 s). 5 fps would snap by 0.100 s and blow that budget.
EXTRACT_FPS = 15.0
STEP = int(round(BASE_FPS / EXTRACT_FPS))        # keep every STEP-th source frame
DURATIONS = [29, 119]               # gold uses 29 / 119 / 299; 299 is skipped as it rarely fits
GRID = 60                           # window-start quantisation for questions that only need
                                    # to CONTAIN a frame, so windows are reused
# Gold's last-visible quadrants are near-uniform by design (25/25/25/24.5). Gallstones are
# not uniformly placed, so ours run 33% top/left and 13% top/right; capping each quadrant
# keeps the pooled distribution from acquiring a positional prior gold does not have.
QUAD_CAP = 13

# How many boundary-anchored rows the pool can absorb. Our clips end ON an annotated frame,
# so the object is visible in the final frame -- correct, but gold answers this template at
# the clip boundary only 10.5% of the time (923 gold questions, 97 at the end, 786 mid-clip;
# measured from the answers, not from a KB proxy). Shipping all ~190 would take the pooled
# boundary rate to 24% and teach "read the last frame". 49 holds the pool at 15%.
LASTVIS_CAP = 49
# The in-domain negatives copy their clip verbatim from a real record, so this MUST point
# at the export being built -- otherwise they arrive carrying that export's frame count and
# the training file ends up with mixed clip lengths.
SEG_ALL = Path(os.environ.get("SEGMENT_EXPORT_DIR", ".")) / "sft_export/all.jsonl"

LASTVIS_Q = ("The last time a {c} is visible, where was the center of the {c} located relative "
             "to the image center? Please select one answer: top/left; top/right; bottom/left; "
             "bottom/right")
COOC_Q = "Do {a} and {b} co-occur in any frame of this video? Please answer with yes or no."


def snap(i: int) -> int:
    """Nearest index on the extracted grid."""
    return int(round(i / STEP)) * STEP


def clip_indices(start_s: float, end_s: float) -> list[int]:
    """N absolute frame indices on the extraction grid, the LAST one landing exactly on
    `end_s` -- that frame is the annotated one, and the last-visible construction rests on
    it being the clip's final frame. Annotation times are whole 10 s marks, so end_s*30 is a
    multiple of 300 and survives snapping exactly."""
    a, b = snap(int(round(start_s * BASE_FPS))), snap(int(round(end_s * BASE_FPS)))
    return [snap(int(round(a + (b - a) * i / (N_FRAMES - 1)))) for i in range(N_FRAMES)]


def record(video: str, start_s: float, end_s: float, uid: str, question: str, answer: str,
           fmt: str, cap: str, idx: list[int]) -> dict:
    return {"uid": uid, "qID": uid, "source_dataset": "lapchole",
            "videoID": f"lapchole/{video}.mp4", "procedure_type": "Cholecystectomy",
            "primary_capability": cap, "secondary_capabilities": [], "format": fmt,
            "start_time": int(start_s), "end_time": int(end_s),
            "duration": int(end_s - start_s), "base_fps": BASE_FPS,
            "frame_dir": str(OUT_FRAMES / video), "frames_indices": idx,
            "n_distinct_frames": len(set(idx)),
            "messages": [{"role": "user", "content": [{"type": "video"},
                                                      {"type": "text", "text": question}]},
                         {"role": "assistant",
                          "content": [{"type": "text", "text": answer}]}]}


def load(csv_dir: Path):
    """-> {video: {t: {"marks": {cls: [pts]}, "co": [cls]}}} and the video's sampled times."""
    obs: dict[str, dict] = defaultdict(dict)
    for fp in sorted(csv_dir.glob("*.csv")):
        for r in csv.DictReader(fp.open()):
            t = secs(r["timestamp"])
            rec = obs[r["video"]].setdefault(t, {"marks": {}, "co": []})
            for o in filter(None, (r.get("other_objects") or "").split(";")):
                if o not in rec["co"]:
                    rec["co"].append(o)
            cls, n = (r.get("target_class") or "").strip(), int(r.get("count") or 0)
            if cls and n > 0:
                rec["marks"][cls] = [tuple(float(v) for v in p.split(","))
                                     for p in filter(None, (r.get("points_xy") or "").split(";"))]
    return obs


def main() -> None:
    global N_FRAMES                       # must precede the first use of N_FRAMES below
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path,
                    default=Path(os.environ.get("LAPCHOLE_CSV_DIR",
                                 "annotations/private/raw/lapchole_annotations")),
                    help="frame-level annotation sheets (see annotations/README.md)")
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("SEGMENT_EXPORT_DIR", "sft_export"))
                            / "lapchole_segment.jsonl")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--max-lastvis", type=int, default=180)
    ap.add_argument("--max-cooc", type=int, default=200)
    ap.add_argument("--in-domain-negatives", type=int, default=200)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--seg-all", type=Path, default=SEG_ALL,
                    help="export whose clips the in-domain negatives inherit; must match "
                         "--n-frames or the training file gets mixed clip lengths")
    ap.add_argument("--n-frames", type=int, default=N_FRAMES,
                    help="frames per clip; must be even (temporal_patch_size=2)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.n_frames % 2:
        ap.error("--n-frames must be even")
    N_FRAMES = a.n_frames

    boxes = json.loads((ANNOT / "crop_boxes.json").read_text())
    obs = load(a.csv_dir)
    rng = random.Random(a.seed)

    # video length in seconds, from the sweep itself (the sweep covered the whole video)
    span = {v: max(t for t in ts) for v, ts in obs.items() if ts}

    lastvis, cooc = [], []
    for video, ts in sorted(obs.items()):
        box = boxes.get(video)
        for t, rec in sorted(ts.items()):
            for c, pts in rec["marks"].items():
                if len(pts) != 1:
                    continue
                p = uncrop(pts[0], box)
                if not p or abs(p[0] - .5) <= a.margin or abs(p[1] - .5) <= a.margin:
                    continue
                q = ("top" if p[1] < .5 else "bottom") + "/" + ("left" if p[0] < .5 else "right")
                lastvis.append((video, t, c, q))
            seen = sorted(set(rec["marks"]) | {o for o in rec["co"] if o in PLURAL})
            for i in range(len(seen)):
                for j in range(i + 1, len(seen)):
                    cooc.append((video, t, seen[i], seen[j]))

    rng.shuffle(lastvis)
    rng.shuffle(cooc)
    lastvis, cooc = lastvis[:min(a.max_lastvis, LASTVIS_CAP)], cooc[:a.max_cooc]

    rows, windows = [], {}
    def window(video, end_t, prefer_end):
        """Reuse an identical window when one already exists -- 80 frames each."""
        for d in DURATIONS:
            if end_t - d >= 0:
                dur = d
                break
        else:
            dur = int(end_t)
        if dur < 20:
            return None
        if prefer_end:
            start = end_t - dur                   # annotated frame must be the final one
        else:
            # Snap to a coarse grid so frames in the same episode share one window (each
            # window costs 80 extractions). A 119 s window on a 60 s grid always contains
            # the frame, since t is inside its own cell.
            dur = 119
            start = max(0.0, (int(end_t) // GRID) * GRID)
        end = start + dur
        if end > span.get(video, end_t):          # never run past what the sweep covered
            start, end = end_t - dur, end_t
        key = (video, round(start), round(end))
        if key not in windows:
            windows[key] = clip_indices(start, end)
        return key, windows[key]

    quad_used: Counter = Counter()
    for video, t, c, q in lastvis:
        if quad_used[q] >= QUAD_CAP:
            continue
        w = window(video, t, prefer_end=True)     # the annotated frame MUST be the last one
        if not w:
            continue
        quad_used[q] += 1
        key, idx = w
        rows.append(record(video, key[1], key[2], f"lapseg-lv-{video[:4]}-{int(t)}-{c[:4]}",
                           LASTVIS_Q.format(c=c), q, "multiple_choice",
                           "SPATIAL_LOCALIZATION_CAMERA", idx))
    for video, t, x, y in cooc:
        w = window(video, t, prefer_end=False)    # only needs to CONTAIN the frame
        if not w:
            continue
        key, idx = w
        if not (key[1] <= t <= key[2]):
            continue
        rows.append(record(video, key[1], key[2],
                           f"lapseg-co-{video[:4]}-{int(t)}-{x[:4]}-{y[:4]}",
                           COOC_Q.format(a=PLURAL[x], b=PLURAL[y]), "yes", "binary",
                           "OBJECT_IDENTIFICATION", idx))

    # In-domain negatives on real gold clips. AHA is safe everywhere (zero examples in any
    # split); Gallstone only on videos the KB never puts a gallstone in.
    neg = 0
    if a.in_domain_negatives and a.seg_all.exists():
        from frame_data.kb import build_kb
        kb, _ = build_kb()
        gall_videos = {v for (ds, v, _), f in kb.items() if "Gallstone" in (f.classes or [])}
        pool = [json.loads(l) for l in a.seg_all.open()]
        rng.shuffle(pool)
        per_video = Counter()
        for r in pool:
            if neg >= a.in_domain_negatives:
                break
            vid = r["videoID"].split("/")[-1]
            if per_video[vid] >= 4:
                continue
            tgt = AHA if (neg % 2 == 0 or vid in gall_videos) else GALL
            other = rng.choice([o for o in PLURAL if o not in (GALL, AHA, "Mesh")])
            rows.append({**{k: r[k] for k in (
                "source_dataset", "videoID", "procedure_type", "start_time", "end_time",
                "duration", "base_fps", "frame_dir", "frames_indices", "n_distinct_frames")},
                "uid": f"{r['uid']}-lapneg", "qID": f"{r['qID']}-lapneg",
                "primary_capability": "OBJECT_IDENTIFICATION", "secondary_capabilities": [],
                "format": "binary",
                "messages": [{"role": "user", "content": [
                    {"type": "video"},
                    {"type": "text", "text": COOC_Q.format(a=PLURAL[tgt], b=PLURAL[other])}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "no"}]}]})
            per_video[vid] += 1
            neg += 1

    if a.extract:
        # every STEP-th frame across each window, not just the indices this N wants
        need = defaultdict(set)
        for (video, s0, e0) in windows:
            need[video].update(range(snap(int(s0 * BASE_FPS)),
                                     snap(int(e0 * BASE_FPS)) + 1, STEP))
        print(f"extracting {sum(len(v) for v in need.values()):,} frames over {len(need)} videos")
        extract_indices(need)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    miss = sum(1 for r in rows if not (Path(r["frame_dir"]) /
               f"frame{r['frames_indices'][-1]:07d}.jpg").exists())
    print(f"\n{len(rows)} segment questions over {len(windows)} distinct windows "
          f"({len(windows) * N_FRAMES:,} frame slots)")
    print("  kinds:", dict(Counter(r["uid"].split("-")[1] if r["uid"].startswith("lapseg")
                                   else "in_domain_neg" for r in rows)))
    print("  formats:", dict(Counter(r["format"] for r in rows)))
    print("  capabilities:", dict(Counter(r["primary_capability"] for r in rows)))
    if miss:
        print(f"  ! {miss} records whose final frame is missing -- rerun with --extract")
    print(f"wrote {a.out}")


def extract_indices(need: dict[str, set[int]], quality: int = 90) -> None:
    import decord
    from PIL import Image

    for video, idxs in sorted(need.items()):
        src = VIDEOS / f"{video}.mp4"
        if not src.exists():
            print(f"  ! no source video for {video}")
            continue
        d = OUT_FRAMES / video
        d.mkdir(parents=True, exist_ok=True)
        vr = decord.VideoReader(str(src))
        want = {i: d / f"frame{i:07d}.jpg" for i in sorted(idxs)
                if not (d / f"frame{i:07d}.jpg").exists()}
        keys = [i for i in want if i < len(vr)]
        for s0 in range(0, len(keys), 64):
            chunk = keys[s0:s0 + 64]
            for fr, i in zip(vr.get_batch(chunk).asnumpy(), chunk):
                Image.fromarray(fr).save(want[i], quality=quality)
        print(f"  {video[:40]:42s} +{len(keys):5d} frames")


if __name__ == "__main__":
    main()
