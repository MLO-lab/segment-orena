"""Append our derived SEGMENT questions to all.jsonl.

Every generated question is attached to a window that ALREADY EXISTS in the export, so we
copy that row's clip fields (frame_dir, frames_indices, base_fps, ...) verbatim and swap
only the question and answer. That is safer than re-deriving the clip: the N=80 sampling
policy stays bit-identical to the rows the model already trains on, with no chance of our
copy of clip_sampling.py drifting from theirs.

Source: segment_data/derive_segment.py -- strict entailments only (class counts from
complete class sets; co-occurrence negatives from the same sets; co-occurrence positives
witnessed by a single frame in the frame-track KB). No inferred labels, so no label noise.

    python merge_generated_segment.py --export sft_export
"""
from __future__ import annotations
import argparse, json, random
from collections import Counter
import os
from pathlib import Path

GEN = Path(os.environ.get("SEGMENT_EXPORT_DIR", ".")) / "segment_derived_all.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--gen", type=Path, default=GEN)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    allp = a.export / "all.jsonl"
    base = [json.loads(l) for l in allp.open()]
    # index the existing windows so a generated question can inherit its clip
    by_win = {}
    for r in base:
        by_win.setdefault((r["source_dataset"], r["videoID"].split("/", 1)[-1],
                           int(r["start_time"]), int(r["end_time"])), r)
    seen = {(r["source_dataset"], r["videoID"], r["messages"][0]["content"][-1]["text"].strip())
            for r in base}

    added, stats = [], Counter()
    for i, line in enumerate(a.gen.open()):
        g = json.loads(line)
        key = (g["dataset"], g["video"], int(g["start_time"]), int(g["end_time"]))
        tpl = by_win.get(key)
        if tpl is None:
            stats["dropped_window_not_in_export"] += 1
            continue
        if (tpl["source_dataset"], tpl["videoID"], g["question"].strip()) in seen:
            stats["dropped_duplicate"] += 1
            continue
        r = dict(tpl)                                   # inherit the clip verbatim
        r["uid"] = f"gen-{i:06d}"
        r["qID"] = f"gen-{i:06d}"
        r["format"] = g["format"]
        # a generator that knows the capability says so; the format-based guess files
        # everything non-numeric under object_identification, which is wrong for the
        # event-understanding templates
        r["primary_capability"] = g.get("capability") or (
            "OBJECT_AGGREGATION" if g["format"] == "number" else "OBJECT_IDENTIFICATION")
        r["secondary_capabilities"] = []
        r["generated_by"] = g["rule"]
        r["label_confidence"] = g.get("label_confidence", 1.0)
        user = [c for c in tpl["messages"][0]["content"] if c.get("type") != "text"]
        r["messages"] = [
            {"role": "user", "content": user + [{"type": "text", "text": g["question"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": str(g["answer"])}]},
        ]
        added.append(r)
        stats[f"kept_{g['rule']}"] += 1

    merged = base + added
    random.Random(a.seed).shuffle(merged)
    with allp.open("w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    print(f"base {len(base)} + generated {len(added)} = {len(merged)} rows -> {allp}")
    for k, v in sorted(stats.items()):
        print(f"  {k:34s} {v}")
    print("  format mix:", dict(Counter(r["format"] for r in merged).most_common()))


if __name__ == "__main__":
    main()
