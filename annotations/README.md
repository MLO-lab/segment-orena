# Supplementary annotations — MLO-Lab, ORena FOCUS 2026 SEGMENT track

The annotations come in two halves.

| half | contents | where |
|---|---|---|
| public | HeiCo + hernia questions, hernia sheets, hernia frames | [`Machine-Learning-Oncology/orena-segment-annotations`](https://huggingface.co/datasets/Machine-Learning-Oncology/orena-segment-annotations) |
| private | LapChole questions and sheets | available **on request** |

```bash
# public half
python annotations/fetch.py

# private half, once you have the archive
mkdir -p annotations/private
tar xzf orena-segment-lapchole-annotations_*.tar.gz \
    -C annotations/private --strip-components=1
```

Neither half is committed to git. Rows you can rebuild: **21,440** with the public half
alone, **23,294** adding the private half, and all **23,454** once the 160 LapChole
questions whose windows are absent from the official export also have their frames
extracted (`--extra-frames`).

> **Licence.** The hernia frames under `public/frames/` are shared for **non-commercial
> research only**, by agreement with the clinical collaborator who supplied the videos.
> The raw videos are **not** redistributed. The LapChole material under `private/` is
> covered by the LapChole-FOCUS data usage agreement and must not be made public until the
> organisers release that dataset.

Annotations and VQA pairs we produced on top of the official FOCUS data.
**3,454 question–answer pairs** plus the **81 raw annotation sheets** they derive from.

```
public/                       release publicly
├── vqa/
│   ├── heico_derived.jsonl        1,300 pairs   HeiCo-FOCUS
│   └── hernia_mesh.jsonl            140 pairs   hernia videos (YouTube-sourced)
├── raw/
│   └── hernia_mesh_annotations/      12 CSVs    our frame-level annotations
└── frames/                          290 MB      REQUIRED -- see 2.4
    ├── clips/                     2,307 jpg     frames the questions refer to
    └── annotated/                   232 jpg     frames the annotators marked up

private/                      organisers only, until LapChole-FOCUS is released
├── vqa/
│   └── lapchole.jsonl             2,014 pairs   LapChole-FOCUS
└── raw/
    ├── lapchole_annotations/         69 CSVs    our frame-level annotations
    └── crop_boxes.json               70 entries geometry of the frames shown to annotators

check_no_lapchole_in_public.py    asserts the split is correct
```

**The split is per row, not per file.** Our generators mix datasets — questions about one
dataset can use clips from another as in-domain negatives — so every row is routed by its
own `source_dataset`. 238 of the public HeiCo rows were produced by the LapChole and hernia
generators but describe HeiCo clips. Run `check_no_lapchole_in_public.py` to verify that
nothing under `public/` is LapChole-derived.

---

## 1. Generation approach

### 1a. Derived from existing labels (2,554 pairs)

Re-asks questions that the official temporal object annotations already determine, over
windows where the label set is unambiguous. No model or human is in the loop; a question is
emitted only where the annotation logically entails the answer.

| rule | question form | answer source |
|---|---|---|
| visibility | is class *C* visible in this window | object track overlaps window |
| retention | is *C* left inside the patient | class property, verified against gold on single-class windows |
| counting | how many distinct FOs / instances | distinct tracks intersecting the window |
| co-occurrence | do *C1* and *C2* both appear | intersection of both tracks |

Classes whose retention is genuinely ambiguous are excluded rather than guessed — `Clip`
splits 35 yes / 8 no in the gold labels, so no retention question is emitted for it.

Produced by `segment_data/derive_segment.py`.

### 1b. New frame-level annotations (1,067 pairs, 900 pairs after dedup)

Human annotators marked foreign objects on frames sampled every 10 s:

* **LapChole-FOCUS** — 69 videos, 520 annotated frames, targeting `Gallstone` and
  `Absorbable Hemostatic Agent`.
* **Hernia (YouTube)** — 12 videos, targeting `Mesh`.

Both use the same sheet format (§2.1). Questions are generated from these sheets by
`frame_data/lapchole_segment_questions.py` and `frame_data/mesh_segment_questions.py`,
under the same entailment rule: only what the sheet settles becomes a question. Counts and
quadrants are used; the free-text `other_objects` column is not treated as an exhaustive
class list, so it does not produce `fo_class` questions on LapChole.

---

## 2. Structural format

### 2.1 Raw annotation sheets — CSV

One row per annotated frame per target class. Header:

```csv
"image","video","timestamp","reviewer","target_class","count","quadrants","points_xy","other_objects"
```

| column | type | description |
|---|---|---|
| `image` | string | frame filename, `<video>__HH-MM-SS.jpg` |
| `video` | string | source video stem |
| `timestamp` | `HH:MM:SS` | position in the source video |
| `reviewer` | string | annotator id (may be empty) |
| `target_class` | string | FO class being annotated, e.g. `Gallstone`, `Mesh` |
| `count` | integer | number of visible instances of `target_class` |
| `quadrants` | string | `top`/`bottom` + `/left`/`right`, `;`-separated; `center` = within a radius-0.18 disc of the frame centre |
| `points_xy` | string | `x,y` per instance, `;`-separated, normalised to `[0,1]` on the **full frame** |
| `other_objects` | string | other FO classes noticed, comma-separated; **not exhaustive** |

A header-only file means the video was reviewed and the target class never appeared — those
are the negatives, not failures.

**`crop_boxes.json`** records the geometry of the frames the annotators actually saw. Some
frames were auto-cropped to remove letterboxing, so `points_xy` is normalised to the *cropped*
image for those videos. This file maps each video back to full-frame coordinates, and the
generators apply it before computing quadrants:

```json
{"0021 - Laparoscopic Cholecystectomy": {
   "source": "0021 - Laparoscopic Cholecystectomy.mp4",
   "source_wh": [1280, 720], "crop_box": [75, 0, 1256, 720],
   "frame_wh": [1181, 720], "fps": 30.0}}
```

Without it, clicks on cropped videos land in the wrong place.

### 2.2 VQA pairs — JSON Lines

One JSON object per line:

```json
{
  "qID": "mesh-01-240",
  "videoID": "hernia_yt/01 - Ventral Hernia - Mesh Reinforcement",
  "source_dataset": "hernia_yt",
  "procedure_type": "Hernia Repair",
  "start_time": 240,
  "end_time": 350,
  "question": "What surgical foreign object is visible in this video? Please provide a class name.",
  "answer": "Mesh",
  "format": "fo_class",
  "primary_capability": "OBJECT_IDENTIFICATION",
  "secondary_capabilities": [],
  "provenance": "hand_annotation"
}
```

| field | type | description |
|---|---|---|
| `qID` | string | unique within this release |
| `videoID` | string | `<dataset>/<video file>`, matching FOCUS |
| `source_dataset` | string | `heico`, `lapchole`, `hernia_yt` |
| `procedure_type` | string \| null | procedure label |
| `start_time`, `end_time` | integer | window bounds, seconds from video start |
| `question` | string | question text as presented |
| `answer` | string | reference answer |
| `format` | string | `binary`, `number`, `fo_class`, `multiple_choice` — FOCUS answer formats |
| `primary_capability` | string | FOCUS capability leaf |
| `secondary_capabilities` | list | additional capability leaves |
| `provenance` | string | `derived_from_labels` or `hand_annotation` |

Times are in **source-video coordinates**, not clip-relative. Frame indices are not included:
they are a function of the window and the chosen frame count, reproduced by the training
code.

### 2.4 Frames

**HeiCo and LapChole frames are not redistributed.** They are regenerated from datasets the
recipient already holds: every frame is a JPEG named by its absolute index in the source
video (`frame0037625.jpg`), so a question's window plus `base_fps` determines the filenames.
`base_fps` is 25.0 for HeiCo and 30.0 for LapChole.

**Hernia frames ARE included, and must be**, because the videos are not part of FOCUS and
cannot be obtained by the recipient. Our agreement with the clinical collaborator permits
redistribution of the **annotated frames only** — never the raw video — for **non-commercial
research use**.

Every frame shipped here lies inside a segment that a released annotation covers. Because a
SEGMENT question annotates a *window* rather than a single instant, the frames it annotates
are the 128 sampled across that window; at `base_fps` 1.0 that is one frame per second of
the annotated segment. No frame outside an annotated segment is included, and
`check_no_lapchole_in_public.py` asserts this — currently 2,307 / 2,307 frames verified
inside a released question window. The raw videos are not redistributed in any form, and at
1 fps these stills are ~3% of the source frames.

`public/frames/` contains:

* `clips/<video>/frame<NNNNNNN>.jpg` — the 2,307 frames referenced by the 140 hernia
  questions. `base_fps` is 1.0 for these videos, so index *n* is second *n*.
* `annotated/<video>/<video>__HH-MM-SS.jpg` — the 232 frames carrying the CSV annotations,
  matching the `image` column.

Regenerate with `export_hernia_frames.py`.

### 2.3 Composition

| file | rows | binary | number | fo_class | multiple_choice |
|---|---:|---:|---:|---:|---:|
| `public/vqa/heico_derived.jsonl` | 1,300 | 1,300 | – | – | – |
| `public/vqa/hernia_mesh.jsonl` | 140 | 28 | 56 | 56 | – |
| `private/vqa/lapchole.jsonl` | 2,014 | 1,448 | 524 | – | 42 |
| **total** | **3,454** | 2,776 | 580 | 56 | 42 |

By provenance: 2,554 `derived_from_labels`, 900 `hand_annotation`.

---

## 3. Use in training

These pairs are appended to the 20,000 official SEGMENT examples, which are kept in full and
not rebalanced, giving **23,454** training rows. Duplicates on
`(source_dataset, videoID, start_time, end_time, question)` are removed — 167 in our run.

See the repository README; `slurm/1_prepare_data.sbatch` performs the merge.
