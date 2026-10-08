# Shelf product detection

A personal computer vision learning project inspired by working at GS25 in Vietnam. The long-term goal is to examine shelf photographs and help identify possible restocking needs. So far, this repository implements **generic product detection and a basic testing interface**.

YOLO11n detects visible product boxes. It does not recognize product names or SKUs, estimate shelf occupancy, measure inventory, or recommend restocking. A detection count is not an inventory count.

## What works now

- SKU-110K images and labels prepared for a single `object` class.
- Local YOLO11n training and further training from saved weights.
- Full validation through Ultralytics, plus separate educational IoU/NMS/matching code.
- Image prediction with label-free borders and a Streamlit interface.

## Setup

Run commands from the repository root. Python 3.10 or 3.11 is a practical choice; the inspected local environment uses Python 3.10.5, Ultralytics 8.4.173, PyTorch 2.12.1+cu126, and Streamlit 1.64.0.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Ultralytics installs PyTorch as a dependency. CPU inference is supported; CUDA is optional. For GPU use, install an appropriate PyTorch build using the [official installation instructions](https://pytorch.org/get-started/locally/). The local training hardware was an NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM. Dense-dataset training is much more demanding than testing one image. The requirements pin the inspected Ultralytics version but are not a complete environment lock.

### Checkpoints

The local default detector is `runs/train-continued-3/weights/best.pt`. Checkpoints and pretrained binaries are excluded from Git, so a fresh clone does not include this trained detector. Use your own trained checkpoint, train with the scripts below, or obtain this checkpoint separately from its owner. There is currently no published checkpoint download.

The dashboard has a checkpoint-path field; prediction and validation accept `--model`. `model.py` uses `src/yolo11n.pt` for initial training; Ultralytics can download official pretrained YOLO11n weights if this file is absent. Those pretrained weights are a starting point, not the SKU-trained model associated with the metrics below.

A local `src/weights/yolo26n.pt` also exists, but none of the inspected project scripts uses it. There is no evidence here of a YOLO26n training or evaluation experiment.

## Dataset and preparation

[SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) contains densely packed retail-product images and bounding-box annotations. This project uses one generic product class, not SKU identities. The dataset provider limits use to academic and non-commercial purposes; obtain the data directly and review its terms before sharing it.

The current local layout is:

```text
data/SKU-110K/
    annotations/                # original CSVs needed only to regenerate labels
        annotations_train.csv
        annotations_val.csv
        annotations_test.csv
    images/{train,val,test}/
    labels/{train,val,test}/
    train.txt, val.txt, test.txt # optional image lists created by conversion
```

The inspected checkout has 8,185 training images, 584 validation images, and 2,920 test images, with matching label filenames in each split. The original annotation CSVs and archive are absent locally; existing labels remain usable for training and validation.

For a fresh preparation:

```powershell
python src/data_download.py
python src/dataset.py
```

The downloader uses the archive linked by the dataset authors, downloads via HTTPS to a partial file, checks archive members, extracts in a temporary directory, and groups the archive's flat `train_*`, `val_*`, and `test_*` image filenames into split folders. It leaves an existing dataset untouched. The archive URL remains linked by the authors and returned HTTP 200 to a HEAD check during cleanup; the roughly 12.2 GB archive was not downloaded.

The converter reads each CSV row's corner coordinates and image dimensions. It computes center x/y and width/height, divides by image dimensions, and writes `0 center_x center_y width height` labels. Invalid dimensions or out-of-image/reversed boxes raise an error for inspection rather than being silently clipped. Labels go into separate split folders, so even colliding filenames cannot overwrite another split. Existing differing labels require an explicit `--overwrite`.

If preparing data manually, keep the original CSVs under `annotations/` and arrange images in their split folders before conversion. `--dataset PATH` supports another preparation directory; also update the YAML's image paths for training there.

`configs/sku110k.yaml` resolves image paths relative to the configuration directory, avoiding a machine-specific absolute path. Ultralytics finds labels in the matching `labels/` folders. The generated image lists are optional; the YAML uses image directories.

## Model and training history

YOLO11n was selected because it is lightweight enough for local experiments and practical inference: approximately 2.58 million parameters and 6.4 GFLOPs at image size 640. See the [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/).

| Local run | Recorded epochs | Evidence |
|---|---:|---|
| `train-2` | 10 | Saved CSV; YOLO11n initialization |
| `train-3` | 1 | Saved CSV; separate experiment |
| `train-continued-3` | 5 | Saved CSV; arguments select `train-2/weights/best.pt`, batch 2, image size 640, `resume: false` |
| `train`, `train-continued`, `train-continued-2` | Unknown | Partial artifacts, no results CSV or weights found |

Small copies of the three recorded CSVs are in [docs/experiments](docs/experiments). They preserve per-epoch results; their last rows are not the separate best-checkpoint validation results below. Full local run directories and checkpoints are preserved but ignored.

```powershell
python src/train.py
python src/resume_training.py
# Optionally continue from a different checkpoint:
python src/resume_training.py --model runs/train-continued-3/weights/best.pt
```

Initial training uses 10 epochs, image size 640, and batch 2. `resume_training.py` starts a **new five-epoch training schedule from saved weights**; it does not request restoration of optimizer/training state. Its historical name is retained for continuity. Both scripts now use `workers=0` for simple Windows execution and let Ultralytics create a new run directory rather than overwrite history. Archived continuation runs used eight workers; that operational change does not alter the archived results. No training was started during cleanup.

## Recorded validation

The supplied validation record for `runs/train-continued-3/weights/best.pt` reports:

| Metric | Value |
|---|---:|
| Precision | 0.878614 |
| Recall | 0.791622 |
| mAP@0.50 | 0.867211 |
| mAP@0.50:0.95 | 0.512002 |

These are **SKU-110K validation results, not GS25 accuracy figures**. The record covers 584 images and 90,456 objects, about 6.5 ms model inference per image, and best F1 about 0.83 at confidence 0.352. Inference timing is specific to the recorded environment and excludes the full application workflow.

The lower mAP50–95 means performance is weaker when tighter box overlap is required. This gap alone does not establish a particular failure mechanism; reviewing missed detections and box placement is still needed.

Local label inspection independently confirms the image/object counts and a maximum of 718 annotated objects in one validation image. The supplied record says Ultralytics raised the normal 300 detection limit to 718. Local `src/runs/detect/val-3/` has validation plots, but no complete machine-readable metrics/configuration log. The exact six-decimal metrics, timing, and F1 above come from the supplied validation record; full validation was not rerun during cleanup.

### Reproduce validation

```powershell
python src/evaluate_ultralytics.py
# Same default checkpoint/config, explicitly:
python src/evaluate_ultralytics.py --model runs/train-continued-3/weights/best.pt --data configs/sku110k.yaml
# If available GPU memory requires a smaller batch:
python src/evaluate_ultralytics.py --batch 2
```

Defaults are validation split, image size 640, batch 16, workers 0, confidence 0.001, NMS IoU 0.7, and `max_det=718`. The old script left batch/workers/detection cap implicit. Batch 16 follows the inspected standalone-validation default; the recorded batch and worker count cannot be recovered from the available plots. These settings are explicit reproduction defaults, not a claim of an exact historical configuration. Lowering batch can affect timing, and a rerun should be logged separately. Results and plots go into a new `runs/detect/val*` directory. Prediction uses different confidence/NMS defaults for qualitative viewing.

## Prediction and Streamlit

```powershell
python src/predict.py path/to/your/image.jpg
python src/predict.py path/to/your/image.jpg --confidence 0.35 --iou 0.5 --thickness 4
python -m streamlit run src/app.py
```

Prediction saves a PNG in `predictions/`, choosing a numbered filename if one already exists. `src/clean_label.py` is a compatibility entry point to the same CLI, replacing its old hardcoded paths and duplicate plotting.

The dashboard supports multiple uploads, confidence and NMS IoU sliders, adjustable border thickness, original/prediction previews side by side, and detection counts. It caches the selected model. PIL RGB images go directly to inference, and PIL draws label-free green borders on a copy of the original. This avoids ambiguous RGB/BGR plotting conversions. EXIF orientation is applied before prediction. Uploaded photographs are processed in memory.

### Example predictions

No store photographs or dataset images are newly published as examples because redistribution permission has not been established. Try a photograph you have permission to use with the commands above. Clean borders are the default. The formerly tracked `test_images/shelf.jpeg` is kept locally and removed from the current tracked tree; earlier Git commits still contain it. This cleanup does not rewrite history.

## From-scratch evaluation and tests

`src/evaluate_from_scratch.py` retains the learning implementations of rectangle intersection, IoU, confidence-ordered NMS, and greedy one-to-one ground-truth matching. Duplicate predictions become false positives once a ground-truth box is matched; unmatched ground truths become false negatives. Zero-area boxes have IoU zero. NMS suppresses overlaps strictly above its threshold; matching accepts IoU at or above its positive threshold.

These functions demonstrate simplified single-image, single-class behavior. They do not compute dataset AP/mAP or replace the full Ultralytics evaluator. Importing the file does not execute the demo.

```powershell
python src/evaluate_from_scratch.py
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

The small tests cover box edge cases, suppression, matching, split conversion, existing-label preservation, unsafe archive entries, and RGB preservation with different border widths. The Ultralytics input-order check runs when that dependency is installed.

## Repository structure

```text
configs/sku110k.yaml
src/
    app.py                     # Streamlit interface
    predict.py                 # image prediction CLI
    visualization.py           # shared PIL box rendering
    paths.py                   # shared local defaults
    clean_label.py             # compatibility CLI entry point
    model.py
    train.py
    resume_training.py         # continued training, new schedule
    data_download.py
    dataset.py
    evaluate_ultralytics.py
    evaluate_from_scratch.py
tests/test_project.py
docs/development_notes.md
docs/experiments/*.csv
requirements.txt
```

Local data, runs, model binaries, photographs, generated previews, archives, secrets, and caches are ignored. No Git LFS or binary distribution is configured. Ultralytics has its own [licensing terms](https://www.ultralytics.com/license); this repository does not grant additional rights to external datasets or weights.

## Current limitations

- No validated GS25 performance metrics; domain shift from SKU-110K remains unmeasured.
- Dense, overlapping, and partially obscured products can be difficult to separate.
- Generic product detection only; no SKU-level recognition or stock-depth information.
- No automatic shelf occupancy estimation or confirmed restocking recommendations.
- Detection count is not inventory count, including products hidden behind the front row.

## Roadmap

### Stage 1 — Detection foundation

- [x] SKU-110K preparation for the current local dataset.
- [x] YOLO11n training and continued training.
- [x] SKU-110K model validation.
- [x] Qualitative image inference.
- [x] Basic Streamlit frontend.
- [ ] Further error review and consistent experiment logging.

### Stage 2 — Occupancy baseline (next active task)

- [ ] Implement `covered_width(intervals)` with interval merging.
- [ ] Extract x-coordinate intervals from detected boxes.
- [ ] Define shelf regions manually at first.
- [ ] Associate detections with shelf regions.
- [ ] Compute horizontal box coverage and visualize candidate gaps.
- [ ] Compare estimates across images.

Coverage measures visible box projections, and candidate gaps still need interpretation. Interval merging and occupancy estimation are not implemented yet.

### Stage 3 — Real-store evaluation

- [ ] Obtain permission for relevant photographs.
- [ ] Build a small labeled GS25 subset and evaluate domain shift.
- [ ] Inspect false positives/misses and investigate whether fine-tuning helps.

### Stage 4 — Shelf-monitoring system

- [ ] Detect or define shelf regions automatically.
- [ ] Analyze capacity and expected arrangement.
- [ ] Distinguish intentional gaps from possible out-of-stock regions.
- [ ] Compare images over time and develop candidate recommendations.

### Stage 5 — Possible extensions

SKU classification, better occupancy models, multi-store generalization, performance optimization, and improved visualization/reporting are possible directions, not promised features.
