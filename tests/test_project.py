import contextlib
import importlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import process_split, xyxy_to_yolo
from data_download import safe_extract
from evaluate_from_scratch import evaluate_detections, intersection_calc, iou, nms
from visualization import draw_boxes
from PIL import Image


def prediction(box, confidence=0.9):
    return {"box": box, "confidence": confidence}


class DetectionTests(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)

    def test_partial_overlap(self):
        self.assertEqual(intersection_calc([0, 0, 10, 10], [5, 5, 15, 15]), 25)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 5, 15, 15]), 1 / 7)

    def test_no_overlap(self):
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)

    def test_zero_area_and_reversed_corners(self):
        self.assertEqual(iou([0, 0, 0, 10], [0, 0, 0, 10]), 0)
        self.assertEqual(iou([10, 0, 0, 10], [0, 0, 10, 10]), 0)

    def test_nms_suppresses_duplicate_in_confidence_order(self):
        low = prediction([0, 0, 10, 10], 0.5)
        high = prediction([0, 0, 10, 10], 0.9)
        far = prediction([20, 20, 30, 30], 0.7)
        source = [low, far, high]
        self.assertEqual(nms(source), [high, far])
        self.assertEqual(source, [low, far, high])

    def test_nms_threshold_boundary(self):
        boxes = [prediction([0, 0, 10, 10]), prediction([0, 0, 10, 10])]
        self.assertEqual(len(nms(boxes, 1)), 2)
        self.assertEqual(len(nms([boxes[0], prediction([20, 20, 30, 30])], 0)), 2)

    def test_duplicate_matching_and_missing_ground_truth(self):
        predictions = [prediction([0, 0, 10, 10]), prediction([0, 0, 10, 10], 0.5)]
        gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
        self.assertEqual(evaluate_detections(predictions, gt), (1, 1, 1))

    def test_matching_empty_inputs(self):
        self.assertEqual(evaluate_detections([], []), (0, 0, 0))
        self.assertEqual(evaluate_detections([], [[0, 0, 10, 10]]), (0, 0, 1))
        self.assertEqual(evaluate_detections([prediction([0, 0, 10, 10])], []), (0, 1, 0))

    def test_matching_threshold_boundary(self):
        self.assertEqual(evaluate_detections(
            [prediction([0, 0, 5, 10])], [[0, 0, 10, 10]], 0.5,
        ), (1, 0, 0))

    def test_invalid_thresholds(self):
        for threshold in [-0.1, 1.1]:
            with self.assertRaises(ValueError):
                nms([], threshold)
        with self.assertRaises(ValueError):
            evaluate_detections([], [], 0)

    def test_import_has_no_demo_output(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            importlib.reload(importlib.import_module("evaluate_from_scratch"))
        self.assertEqual(output.getvalue(), "")


class DatasetTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(xyxy_to_yolo(10, 20, 30, 60, 100, 100), (0.2, 0.4, 0.2, 0.4))

    def test_invalid_coordinates(self):
        for args in [(0, 0, 1, 1, 0, 10), (2, 0, 1, 1, 10, 10), (-1, 0, 1, 1, 10, 10)]:
            with self.assertRaises(ValueError):
                xyxy_to_yolo(*args)

    def test_split_labels_and_preservation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "annotations").mkdir()
            for split in ["train", "val"]:
                (root / "images" / split).mkdir(parents=True)
                (root / "images" / split / "same.jpg").touch()
                (root / "annotations" / f"annotations_{split}.csv").write_text(
                    "same.jpg,10,20,30,60,object,100,100\n",
                )
                process_split(split, root)
            expected = "0 0.200000 0.400000 0.200000 0.400000\n"
            self.assertEqual((root / "labels/train/same.txt").read_text(), expected)
            self.assertEqual((root / "labels/val/same.txt").read_text(), expected)
            self.assertEqual((root / "val.txt").read_text(), "./images/val/same.jpg\n")
            label = root / "labels/train/same.txt"
            label.write_text("user edited label\n")
            with self.assertRaises(FileExistsError):
                process_split("train", root)
            self.assertEqual(label.read_text(), "user edited label\n")

    def test_missing_image(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "annotations").mkdir()
            (root / "annotations/annotations_val.csv").write_text(
                "missing.jpg,0,0,1,1,object,10,10\n",
            )
            with self.assertRaises(FileNotFoundError):
                process_split("val", root)

    def test_archive_rejects_traversal_and_links(self):
        for name, kind in [("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                           ("C:/escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]:
            with tempfile.TemporaryDirectory() as temporary_dir:
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w") as archive:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = "../outside"
                    archive.addfile(member)
                buffer.seek(0)
                with tarfile.open(fileobj=buffer) as archive:
                    with self.assertRaises(ValueError):
                        safe_extract(archive, temporary_dir)

    def test_archive_extracts_regular_file(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                member = tarfile.TarInfo("nested/file.txt")
                member.size = 2
                archive.addfile(member, io.BytesIO(b"ok"))
            buffer.seek(0)
            with tarfile.open(fileobj=buffer) as archive:
                safe_extract(archive, temporary_dir)
            self.assertEqual((Path(temporary_dir) / "nested/file.txt").read_text(), "ok")


class ColorTests(unittest.TestCase):
    def test_rgb_colors_and_border_thickness(self):
        image = Image.new("RGB", (90, 30))
        for x, color in [(0, "red"), (30, "green"), (60, "blue")]:
            image.paste(color, (x, 0, x + 30, 30))
        before = image.tobytes()
        thin = draw_boxes(image, [[2, 2, 27, 27]], 1)
        thick = draw_boxes(image, [[2, 2, 27, 27]], 4)
        for x in [15, 45, 75]:
            self.assertEqual(thick.getpixel((x, 15)), image.getpixel((x, 15)))
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(thin.getpixel((3, 15)), (255, 0, 0))
        self.assertEqual(thick.getpixel((3, 15)), (0, 255, 0))

    def test_ultralytics_pil_input_order(self):
        try:
            from ultralytics.data.loaders import LoadPilAndNumpy
        except ImportError:
            self.skipTest("Ultralytics unavailable")
        image = Image.new("RGB", (3, 1))
        image.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        bgr = LoadPilAndNumpy._single_check(image)
        self.assertEqual(bgr.tolist(), [[[0, 0, 255], [0, 255, 0], [255, 0, 0]]])


if __name__ == "__main__":
    unittest.main()
