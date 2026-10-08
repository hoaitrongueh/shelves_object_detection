# Development notes

This is a learning project, motivated by shelf work at GS25. The supplied project history describes difficulties collecting and manually annotating shelf photographs, followed by a switch to SKU-110K for the initial detector. This cleanup did not reconstruct missing dates or experiments.

## Dataset and local training

CSV corner coordinates become normalized center/size labels. The inspected local dataset is already arranged into train/validation/test folders. The old converter wrote labels to a shared directory and generated paths for flat images, which would not reproduce the current YAML layout. The converter now writes labels and image lists per split. No image basenames overlap across the inspected splits, but separate label directories also protect against future collisions. Original CSVs are currently absent; conversion was verified with temporary fixtures, not by rewriting the real labels.

A 4 GB laptop GPU constrained training to small batches. Saved arguments/CSVs establish ten epochs in `train-2`, one in `train-3`, and five new epochs in `train-continued-3` starting from `train-2/weights/best.pt`. Other runs have partial artifacts. Starting from weights with `resume=False` is continued training with a new schedule, not exact optimizer-state resumption.

The supplied history reports Windows multiprocessing trouble. Main guards are retained and the scripts now explicitly use zero workers for a simple baseline. The archived continuation used eight workers. No new training was performed to compare these settings.

## Detection internals

The hand-written intersection, IoU, NMS, and matching functions are preserved as learning exercises. Matching sorts predictions by confidence and assigns each ground truth at most once. The audit guards invalid thresholds and prevents nonexistent matches; NMS now keeps overlaps equal to its threshold. These boundary behaviors have small tests. This is not an AP/mAP implementation or evidence that the learner can independently explain or rebuild every function.

## Validation and domain shift

The supplied best-checkpoint validation reports precision 0.878614, recall 0.791622, mAP50 0.867211, and mAP50–95 0.512002 on SKU-110K. The final training CSV row differs slightly and is kept separately. Local labels confirm 584 validation images, 90,456 boxes, and a maximum of 718 per image. Validation explicitly retains that dense-image detection cap.

Stricter IoU scoring lowers localization performance; the metric gap alone does not establish why. No quantitative GS25 evaluation exists. Domain shift is a concern to investigate, not a measured degradation in this audit.

## Color rendering and interface

The old dashboard passed a PIL-derived RGB NumPy array to inference. Installed Ultralytics 8.4.173 assumes NumPy images are BGR, while PIL images are RGB. Converting a separate array later for plotting does not correct the earlier inference input. The earlier `plot(pil=True)` route also made output-order assumptions harder to follow.

The dashboard and CLI now pass a PIL RGB image directly to inference and share a small PIL rectangle-drawing helper. A synthetic image with red, green, and blue regions verifies unchanged interiors and adjustable border width. EXIF orientation is applied consistently, class/confidence labels are hidden, and the dashboard displays each original/prediction pair once. `clean_label.py` delegates to prediction instead of maintaining a second renderer.

## Next implementation

Implement and test horizontal interval merging first, then clip/associate detected x intervals with manually defined shelf regions. Measure covered width and draw candidate gaps. Neither box coverage nor gap size alone establishes a restocking need.

## Cleanup verification — 2026-10-08

Python compilation and 19 unit tests passed. Additional smoke checks passed for script imports/CLI help, real-checkpoint inference on synthetic RGB input, prediction color preservation and non-overwriting output, the dashboard upload/render flow with a mocked detector, Streamlit AppTest startup, and the documented Streamlit command's localhost health endpoint. The Streamlit checks required execution outside sandbox socket restrictions; the temporary server was stopped.

The YAML resolves to the existing split directories. The dataset archive returned HTTP 200 to a HEAD request; no archive was downloaded. Full dataset conversion could not be repeated because the original CSVs are absent locally. No training, full-dataset validation, or new GS25 evaluation was performed.
