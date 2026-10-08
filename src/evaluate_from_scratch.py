"""Simplified single-image, single-class detection evaluation exercises."""


def intersection_calc(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    left = max(ax1, bx1)
    right = min(ax2, bx2)
    top = max(ay1, by1)
    bottom = min(ay2, by2)
    width = max(0, right - left)
    height = max(0, bottom - top)
    return width * height


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    intersection = intersection_calc(box_a, box_b)
    union = area_a + area_b - intersection
    if union == 0:
        return 0
    return intersection / union


def nms(predictions, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must be between 0 and 1.")
    predictions = sorted(
        predictions, key=lambda prediction: prediction["confidence"], reverse=True,
    )
    kept = []
    while predictions:
        current_prediction = predictions.pop(0)
        kept.append(current_prediction)
        remaining = []
        for prediction in predictions:
            overlap = iou(current_prediction["box"], prediction["box"])
            if overlap <= threshold:
                remaining.append(prediction)
        predictions = remaining
    return kept


def evaluate_detections(predictions, ground_truths, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValueError("Matching IoU threshold must be greater than 0 and at most 1.")
    predictions = sorted(
        predictions, key=lambda prediction: prediction["confidence"], reverse=True,
    )
    matched_gt = set()
    tp = 0
    fp = 0
    for prediction in predictions:
        best_iou = 0
        best_gt_index = -1
        for index, gt_box in enumerate(ground_truths):
            if index in matched_gt:
                continue
            overlap = iou(prediction["box"], gt_box)
            if overlap > best_iou:
                best_iou = overlap
                best_gt_index = index
        if best_gt_index != -1 and best_iou >= threshold:
            tp += 1
            matched_gt.add(best_gt_index)
        else:
            fp += 1
    fn = len(ground_truths) - len(matched_gt)
    return tp, fp, fn


def main():
    ground_truths = [[10, 10, 50, 50], [100, 100, 140, 140]]
    predictions = [
        {"box": [12, 12, 48, 48], "confidence": 0.95},
        {"box": [15, 15, 45, 45], "confidence": 0.80},
        {"box": [200, 200, 240, 240], "confidence": 0.70},
    ]
    tp, fp, fn = evaluate_detections(predictions, ground_truths)
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)


if __name__ == "__main__":
    main()
