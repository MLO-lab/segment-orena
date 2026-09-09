"""Fail if any LapChole-FOCUS-derived content appears under annotations/public/.

The data usage agreement prohibits publishing LapChole-FOCUS or its derivatives until the
organisers release it. The generators mix datasets -- in-domain negatives borrow clips from
the pooled export -- so the public/private split is per ROW, not per file. This asserts it.

    python annotations/check_no_lapchole_in_public.py
"""
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parent
bad = []

for p in sorted((root / "public").rglob("*.jsonl")):
    for i, line in enumerate(p.open(), 1):
        r = json.loads(line)
        if r.get("source_dataset") == "lapchole":
            bad.append(f"{p.relative_to(root)}:{i} source_dataset=lapchole")
        if "lapchole" in str(r.get("videoID", "")).lower():
            bad.append(f"{p.relative_to(root)}:{i} videoID={r['videoID']}")

for p in sorted((root / "public").rglob("*.csv")):
    head = p.open().read(4096).lower()
    if "cholecystectomy" in head or "lapchole" in head:
        bad.append(f"{p.relative_to(root)} mentions cholecystectomy")

if bad:
    print("FAIL: LapChole-derived content found under public/")
    for b in bad[:20]:
        print("  ", b)
    sys.exit(1)

# Hernia frames: every shipped frame must fall inside a window that a RELEASED question
# covers. The agreement with the clinical collaborator permits the annotated frames only,
# so a frame outside every annotated segment would be redistributing unannotated video.
clips = root / "public/frames/clips"
if clips.exists():
    from collections import defaultdict
    wins = defaultdict(list)
    for line in (root / "public/vqa/hernia_mesh.jsonl").open():
        r = json.loads(line)
        wins[r["videoID"].split("/", 1)[1]].append((r["start_time"], r["end_time"]))
    n = 0
    for vd in sorted(clips.iterdir()):
        ws = wins.get(vd.name, [])
        for f in vd.glob("*.jpg"):
            i = int(f.stem.replace("frame", ""))   # base_fps 1.0 -> index == second
            n += 1
            if not any(s <= i <= e for s, e in ws):
                bad.append(f"public/frames/clips/{vd.name}/{f.name} outside every annotated segment")
    if bad:
        print("FAIL: frames outside annotated segments")
        for b in bad[:20]:
            print("  ", b)
        sys.exit(1)
    print(f"OK: {n} hernia frames, all inside a released question window")

pub = sum(1 for p in (root / "public").rglob("*.jsonl") for _ in p.open())
prv = sum(1 for p in (root / "private").rglob("*.jsonl") for _ in p.open())
print(f"OK: public/ holds {pub} rows, none LapChole-derived; private/ holds {prv} rows")
