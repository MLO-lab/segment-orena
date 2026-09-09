"""Derive SEGMENT-track questions that are ENTAILED by labels we already have.

The FRAME analogue (frame_data/derive.py) worked off a per-frame class set. SEGMENT is a
different problem and most of it is not derivable:

* `time` is 38.2% of the track, and nothing we hold entails "when was the 1st Clip
  inserted" for an arbitrary window -- shrink the window past that event and the answer
  changes. Not derivable, left alone.
* Only ONE question text gives a complete class set for a window -- "Which foreign object
  classes appear in this video?" (1,007 rows). The other fo_class texts are event-specific
  ("what was inserted", "1st unique class", "longest visible") and are NOT complete sets;
  deriving from them would invent labels.

So two sources, both strict entailments, no inference:

  A. complete class set S for window W
       -> "How many different foreign object classes do you see in this video?" = |S|
       -> "Do Xs and Ys co-occur in any frame of this video?" = **no** whenever X or Y is
          absent from S. (The positive direction does NOT follow: both classes appearing
          somewhere in the window does not put them in the same frame.)

  B. the FRAME per-frame knowledge base, restricted to frames inside W
       -> "Do Xs and Ys co-occur in any frame of this video?" = **yes** when a single frame
          inside W carries both. This is the one direction frame data settles, and it is
          the cross-track transfer the 2026-08-05 analysis pointed at.

    python -m segment_data.derive_segment --out segment_derived_all.jsonl
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from focus import FocusDataset, DatasetSplit, Track          # noqa: E402
from frame_data.kb import build_kb                            # noqa: E402

CLASSCOUNT_Q = ("How many different foreign object classes do you see in this video? "
                "Please provide a number.")
COOCCUR_Q = "Do {a}s and {b}s co-occur in any frame of this video? Please answer with yes or no."
VISIBLE_Q = ("What surgical foreign object is visible in this video? Please provide a class "
             "name or answer with none.")
RETENTION_Q = ("Does the surgical foreign object visible here usually remain intra-abdominally "
               "at the end of the procedure?")

# Whether an object is left in the patient is a property of the CLASS, not of the video --
# verified against gold on windows whose class set is a singleton: Specimen 35/35 no,
# Specimen bag 26/26 no, External drain 12/12 yes, Sponge 11/11 no, Silicone loop 4/4 no,
# Needle 4/4 no, Gallstone 2/2 no. Mesh and AHA have no gold at all; the FO definitions say
# both are left in place ("not removed at the end of the surgery" / "intended to be left in
# the body and absorbed").
#
# **Clip is deliberately absent.** Gold splits 35 yes / 8 no on it, which the definition
# explains -- "May potentially remain in the body" makes "usually" a judgement call. Emitting
# Clip either way would ship ~19% label noise on the largest single-class group (160 windows).
RETENTION = {"Specimen": "no", "Specimen bag": "no", "External drain": "yes", "Sponge": "no",
             "Silicone loop": "no", "Needle": "no", "Gallstone": "no", "Mesh": "yes",
             "Absorbable hemostatic agent": "yes"}
SEED_Q = "which foreign object classes appear in this video"


def norm(c: str) -> str:
    return c.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="segment_derived_all.jsonl")
    ap.add_argument("--datasets", nargs="+", default=["heico", "lapchole"])
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--max-binary", type=int, default=2400,
                    help="total co-occurrence binaries, split evenly yes/no")
    ap.add_argument("--max-cooccur-per-window", type=int, default=3,
                    help="cap so a handful of dense windows cannot flood the output")
    a = ap.parse_args()

    # ---- collect windows, and the complete class sets where we have them ----
    windows = []            # (dataset, videoID, start, end, qid)
    class_set = {}          # index into `windows` -> frozenset
    seen = set()            # (dataset, videoID, question) already asked verbatim
    split_of = {}
    for ds in a.datasets:
        for sp in a.splits:
            for req, ref in FocusDataset(ds, DatasetSplit(sp), Track.SEGMENT):
                key = (ds, req.videoID, round(req.start_time), round(req.end_time))
                seen.add((ds, req.videoID, key[2], key[3], req.question.strip()))
                fmt = str(getattr(ref._format, "value", ref._format))
                if fmt == "fo_class" and SEED_Q in req.question.lower():
                    cs = frozenset(norm(c) for c in str(ref.answer).replace(";", ",").split(",")
                                   if c.strip())
                    if cs:
                        class_set[key] = cs
                        split_of[key] = sp
                windows.append(key)
    windows = sorted(set(windows))
    print(f"{len(windows)} distinct windows, {len(class_set)} with a complete class set")

    # ---- frame KB, bucketed per video so a window lookup is cheap ----
    kb, _ = build_kb()
    frames_by_video = defaultdict(list)
    for (dset, vid, t), f in kb.items():
        if f.classes:
            frames_by_video[(dset, vid)].append((t, frozenset(f.classes)))
    for v in frames_by_video:
        frames_by_video[v].sort()
    print(f"frame KB: {sum(len(v) for v in frames_by_video.values())} labelled frames "
          f"over {len(frames_by_video)} videos")

    out, stats = [], Counter()

    def emit(ds, vid, s, e, question, answer, fmt, rule, src, capability=None):
        if (ds, vid, s, e, question) in seen:
            stats["skipped_already_asked"] += 1
            return
        seen.add((ds, vid, s, e, question))
        row = {"dataset": ds, "video": vid, "start_time": s, "end_time": e,
               "question": question, "answer": answer, "format": fmt,
               "rule": rule, "source": src, "label_confidence": 1.0}
        if capability:                      # else the merger infers it from the format, which
            row["capability"] = capability  # would file these under object_identification
        out.append(row)
        stats[f"kept_{rule}"] += 1

    # ---- A: entailed from the complete class set ----
    for key, cs in class_set.items():
        ds, vid, s, e = key
        emit(ds, vid, s, e, CLASSCOUNT_Q, str(len(cs)), "number", "class_count", "derived")

        # A singleton class set settles two more templates that nothing was generating.
        # The retention one is the single biggest template in EVENT_UNDERSTANDING -- 3.5% of
        # the data carrying 20% of the metric, and our weakest bucket.
        if len(cs) == 1:
            only = next(iter(cs))
            emit(ds, vid, s, e, VISIBLE_Q, only, "fo_class", "single_class_visible",
                 "derived", "OBJECT_IDENTIFICATION")
            keep = RETENTION.get(only) or RETENTION.get(only.capitalize())
            if keep:
                emit(ds, vid, s, e, RETENTION_Q, keep, "binary", "retention", "derived",
                     "FO_USAGE_PURPOSE")
        absent = [c for c in ("Clip", "Sponge", "Specimen", "Specimen bag", "External drain",
                              "Needle", "Silicone loop", "Gallstone") if c not in cs]
        present = sorted(cs)
        n = 0
        for x in present:                       # one present + one absent -> cannot co-occur
            for y in absent:
                if n >= a.max_cooccur_per_window:
                    break
                lo, hi = sorted([x, y])
                emit(ds, vid, s, e, COOCCUR_Q.format(a=lo, b=hi), "no", "binary",
                     "cooccur_absent", "derived")
                n += 1

    # ---- B: cross-track, frame data settles the positive co-occurrence direction ----
    for (ds, vid, s, e) in windows:
        fr = frames_by_video.get((ds, vid))
        if not fr:
            continue
        pairs = set()
        for t, cls in fr:
            if s <= t <= e and len(cls) >= 2:
                pairs |= {tuple(sorted(p)) for p in combinations(sorted(cls), 2)}
        for i, (x, y) in enumerate(sorted(pairs)):
            if i >= a.max_cooccur_per_window:
                break
            emit(ds, vid, s, e, COOCCUR_Q.format(a=x, b=y), "yes", "binary",
                 "cooccur_frame_witness", "frame_kb")

    # Balance and cap the co-occurrence binaries. Two reasons, both measured:
    #   * an imbalanced yes/no set teaches a prior rather than vision;
    #   * binary is only 7.1% of the real SEGMENT track (1,425 of 20,000 rows), so dumping
    #     12k binaries in would take the mixture to ~41% binary -- a bigger distribution
    #     shift than the data is worth. The cap keeps the addition proportionate.
    import random
    rng = random.Random(42)
    yes = [r for r in out if r["rule"] == "cooccur_frame_witness"]
    no = [r for r in out if r["rule"] == "cooccur_absent"]
    rest = [r for r in out if r["rule"] not in ("cooccur_frame_witness", "cooccur_absent")]
    half = min(len(yes), len(no), a.max_binary // 2)
    yes, no = rng.sample(yes, half), rng.sample(no, half)
    stats["binary_each_side"] = half
    out = rest + yes + no
    rng.shuffle(out)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"\nwrote {len(out)} questions -> {a.out}")
    for k, v in sorted(stats.items()):
        print(f"  {k:26s} {v}")
    print("  formats:", dict(Counter(r["format"] for r in out)))
    print("  answers:", dict(Counter(str(r["answer"]) for r in out).most_common(6)))


if __name__ == "__main__":
    main()
