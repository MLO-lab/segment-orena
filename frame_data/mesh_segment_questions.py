"""SEGMENT-track questions for Mesh, built from the hernia-video annotations.

Same premise as the FRAME version: Mesh has zero examples anywhere, and these 12 hernia
repairs are the only footage of it. A SEGMENT record is a *clip* -- 80 frame indices over a
window -- so this needs the 1 fps extraction under `hernia_yt/frames/<stem>/`, declared with
`base_fps=1.0` so an index is simply a second.

Emitted only where the annotation entails the answer:

* the class set of a mesh-containing window is exactly {Mesh} (reviewer confirmed no other
  registered object appears), so `fo_class`, class-count and co-occurrence all follow;
* in-domain negatives put "Mesh" in questions about OUR heico/lapchole clips, answer `no`,
  so mesh questions stop being answerable from source cues alone.

**No `time` questions.** They are 38% of SEGMENT and the obvious target, but the annotation
is sampled every 10 s while `time` is scored against a 1.3-4.3 s threshold. First-visible
would be wrong more often than right, and it would land that error in temporal_grounding.

    python -m frame_data.mesh_segment_questions --csv-dir <dir> --out seg_mesh.jsonl
"""
from __future__ import annotations
import argparse, csv, json, random
from collections import Counter, defaultdict
import os
from pathlib import Path

FRAMES = Path(os.environ.get("ORENA_DATA_ROOT", "/mnt/vast/workspaces/VL_LeJepa/data/orena")) / "hernia_yt/frames"
N_FRAMES, WINDOW = 80, 119          # the shipped recipe's clip length and modal window
OTHERS = ["Clip", "Sponge", "Specimen", "Specimen bag", "External drain",
          "Silicone loop", "Needle", "Gallstone"]

QS = [("Which foreign object classes appear in this video? Please provide the class names "
       "separated by commas.", "Mesh", "fo_class", "OBJECT_IDENTIFICATION"),
      ("What surgical foreign object is visible in this video? Please provide a class name.",
       "Mesh", "fo_class", "OBJECT_IDENTIFICATION"),
      ("How many different foreign object classes do you see in this video? Please provide "
       "a number.", "1", "number", "OBJECT_AGGREGATION"),
      ("What is the maximum number of Meshes appearing at once in a single frame? Please "
       "provide a number.", "1", "number", "OBJECT_AGGREGATION")]
COOC = ("Do Meshes and {o}s co-occur in any frame of this video? Please answer with yes or "
        "no.", "no", "binary", "INSTANCE_MATCHING")


def clip(frame_dir: Path, s: int, e: int, n_avail: int) -> list[int]:
    first, last = min(s, n_avail - 1), min(e, n_avail - 1)
    return [int(round(first + (last - first) * i / (N_FRAMES - 1))) for i in range(N_FRAMES)]


def main() -> None:
    global N_FRAMES                       # must precede the first use of N_FRAMES below
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path,
                    default=Path(os.environ.get("MESH_CSV_DIR",
                                 "annotations/public/raw/hernia_mesh_annotations")),
                    help="frame-level annotation sheets (see annotations/README.md)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--in-domain-negatives", type=int, default=300)
    ap.add_argument("--seg-all", type=Path,
                    default=Path(os.environ.get("SEGMENT_EXPORT_DIR", "sft_export"))
                            / "all.jsonl",
                    help="export whose clips the in-domain negatives inherit")
    ap.add_argument("--n-frames", type=int, default=N_FRAMES,
                    help="frames per clip; must be even (temporal_patch_size=2)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    N_FRAMES = a.n_frames

    # mesh-positive seconds per video, from the 10 s annotation grid
    mesh_at: dict[str, list[int]] = defaultdict(list)
    for fp in sorted(a.csv_dir.glob("*.csv")):
        for r in csv.DictReader(fp.open()):
            if int(r["count"] or 0) > 0:
                h, m, s = (int(x) for x in r["timestamp"].split(":"))
                mesh_at[r["video"]].append(h * 3600 + m * 60 + s)

    rng = random.Random(a.seed)
    rows, per_video = [], Counter()
    for video, secs in sorted(mesh_at.items()):
        d = FRAMES / video
        if not d.exists():
            continue
        n_avail = len(list(d.glob("*.jpg")))
        lo, hi = min(secs), max(secs)
        # windows fully inside the mesh period, so every sampled frame contains mesh
        starts = list(range(lo, max(lo + 1, hi - WINDOW + 1), WINDOW // 2)) or [lo]
        for st in starts:
            en = min(st + WINDOW, hi)
            if en - st < 20:
                continue
            idx = clip(d, st, en, n_avail)
            base = {"uid": f"hernia/{video}/{st}", "qID": f"mesh-{video[:2]}-{st}",
                    "source_dataset": "hernia_yt", "videoID": f"hernia_yt/{video}",
                    "procedure_type": "Hernia Repair", "secondary_capabilities": [],
                    "start_time": st, "end_time": en, "duration": en - st,
                    "base_fps": 1.0, "frame_dir": str(d), "frames_indices": idx,
                    "n_distinct_frames": len(set(idx))}
            for q, ans, fmt, leaf in QS:
                rows.append({**base, "format": fmt, "primary_capability": leaf,
                             "messages": [{"role": "user", "content": [
                                 {"type": "video"}, {"type": "text", "text": q}]},
                                 {"role": "assistant", "content": [{"type": "text", "text": ans}]}]})
            q, ans, fmt, leaf = COOC
            o = OTHERS[len(rows) % len(OTHERS)]
            rows.append({**base, "format": fmt, "primary_capability": leaf,
                         "messages": [{"role": "user", "content": [
                             {"type": "video"}, {"type": "text", "text": q.format(o=o)}]},
                             {"role": "assistant", "content": [{"type": "text", "text": ans}]}]})
            per_video[video] += 1

    # in-domain negatives: reuse real clips from all.jsonl, ask a Mesh co-occurrence
    neg = 0
    allj = a.seg_all          # was a RELATIVE path that did not resolve from the repo
                              # root, so the negatives were silently skipped
    if a.in_domain_negatives and allj.exists():
        pool = [json.loads(l) for l in allj.open()]
        rng.shuffle(pool)
        seen = Counter()
        for r in pool:
            if neg >= a.in_domain_negatives:
                break
            if seen[r["videoID"]] >= 4:
                continue
            o = OTHERS[neg % len(OTHERS)]
            rows.append({**{k: r[k] for k in (
                "uid", "qID", "source_dataset", "videoID", "procedure_type", "start_time",
                "end_time", "duration", "base_fps", "frame_dir", "frames_indices",
                "n_distinct_frames")},
                "uid": r["uid"] + "-meshneg", "qID": str(r["qID"]) + "-meshneg",
                "secondary_capabilities": [], "format": "binary",
                "primary_capability": "INSTANCE_MATCHING",
                "messages": [{"role": "user", "content": [
                    {"type": "video"},
                    {"type": "text", "text": COOC[0].format(o=o)}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "no"}]}]})
            seen[r["videoID"]] += 1
            neg += 1

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} segment questions: {len(rows)-neg} from {len(per_video)} hernia videos "
          f"({sum(per_video.values())} clips), {neg} in-domain negatives")
    print("  formats:", dict(Counter(r["format"] for r in rows)))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
