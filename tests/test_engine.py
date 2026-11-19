from __future__ import annotations

import random
from pathlib import Path
import sys
import tempfile
import threading
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_augmenter.engine import (
    AugmentConfig,
    CandidateRejected,
    ImageGenerator,
    InputError,
    TransformPlan,
    YoloBox,
    affine_for_plan,
    apply_plan,
    load_source,
    parse_yolo_labels,
    transform_boxes,
)


class GeometryTests(unittest.TestCase):
    def test_horizontal_and_vertical_flip_preserve_class_ids(self) -> None:
        boxes = (
            YoloBox(3, 0.2, 0.3, 0.1, 0.2),
            YoloBox(9, 0.7, 0.6, 0.2, 0.1),
        )
        matrix = affine_for_plan(TransformPlan(("hflip", "vflip"), hflip=True, vflip=True))
        result = transform_boxes(boxes, matrix)
        self.assertEqual([box.class_id for box in result], [3, 9])
        self.assertAlmostEqual(result[0].x_center, 0.8)
        self.assertAlmostEqual(result[0].y_center, 0.7)
        self.assertAlmostEqual(result[0].width, 0.1)
        self.assertAlmostEqual(result[0].height, 0.2)
        self.assertAlmostEqual(result[1].x_center, 0.3)
        self.assertAlmostEqual(result[1].y_center, 0.4)

    def test_90_degree_rotation_swaps_axis_aligned_box_dimensions(self) -> None:
        box = YoloBox(4, 0.5, 0.5, 0.2, 0.4)
        matrix = affine_for_plan(TransformPlan(("rotate",), angle=90.0))
        result = transform_boxes((box,), matrix)[0]
        self.assertEqual(result.class_id, 4)
        self.assertAlmostEqual(result.x_center, 0.5)
        self.assertAlmostEqual(result.y_center, 0.5)
        self.assertAlmostEqual(result.width, 0.4)
        self.assertAlmostEqual(result.height, 0.2)

    def test_rotation_then_horizontal_flip_uses_same_order_as_pixels(self) -> None:
        box = YoloBox(2, 0.25, 0.25, 0.1, 0.2)
        plan = TransformPlan(("rotate", "hflip"), angle=90.0, hflip=True)
        result = transform_boxes((box,), affine_for_plan(plan))[0]
        self.assertAlmostEqual(result.x_center, 0.75)
        self.assertAlmostEqual(result.y_center, 0.75)
        self.assertAlmostEqual(result.width, 0.2)
        self.assertAlmostEqual(result.height, 0.1)

    def test_out_of_frame_box_rejects_entire_candidate_instead_of_clamping(self) -> None:
        boxes = (YoloBox(0, 0.04, 0.5, 0.06, 0.2), YoloBox(7, 0.5, 0.5, 0.1, 0.1))
        plan = TransformPlan(("rotate",), angle=30.0)
        with self.assertRaises(CandidateRejected):
            transform_boxes(boxes, affine_for_plan(plan))

    def test_photometric_changes_leave_all_boxes_identical(self) -> None:
        image = np.full((640, 640, 3), 100, dtype=np.uint8)
        boxes = (YoloBox(5, 0.4, 0.5, 0.2, 0.3), YoloBox(8, 0.7, 0.25, 0.1, 0.1))
        plan = TransformPlan(("brightness", "noise"), brightness=1.2, noise_sigma=4.0)
        result = apply_plan(image, boxes, plan, random.Random(123))
        self.assertEqual(result.boxes, boxes)
        self.assertFalse(np.array_equal(result.image, image))

    def test_rotated_pixels_and_transformed_box_share_the_same_geometry(self) -> None:
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        image[200:320, 180:280] = (255, 255, 255)
        box = YoloBox(6, 230 / 640, 260 / 640, 100 / 640, 120 / 640)
        plan = TransformPlan(("rotate", "hflip"), angle=23.0, hflip=True)
        result = apply_plan(image, (box,), plan, random.Random(5))
        pixels_y, pixels_x = np.where(result.image[:, :, 0] > 128)
        transformed = result.boxes[0]
        left = (transformed.x_center - transformed.width / 2) * 640
        right = (transformed.x_center + transformed.width / 2) * 640
        top = (transformed.y_center - transformed.height / 2) * 640
        bottom = (transformed.y_center + transformed.height / 2) * 640
        self.assertGreater(len(pixels_x), 0)
        self.assertAlmostEqual(left, int(pixels_x.min()), delta=2.0)
        self.assertAlmostEqual(right, int(pixels_x.max()) + 1, delta=2.0)
        self.assertAlmostEqual(top, int(pixels_y.min()), delta=2.0)
        self.assertAlmostEqual(bottom, int(pixels_y.max()) + 1, delta=2.0)

    def test_invalid_yolo_rows_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            label = Path(temp) / "bad.txt"
            label.write_text("0 0.5 0.5 1.2 0.2\n", encoding="utf-8")
            with self.assertRaises(InputError):
                parse_yolo_labels(label)


class GeneratorTests(unittest.TestCase):
    def _make_input(self, root: Path) -> tuple[Path, Path]:
        images, labels = root / "input" / "images", root / "input" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        image[200:400, 180:460] = (40, 120, 200)
        self.assertTrue(cv2.imwrite(str(images / "sample.png"), image))
        (labels / "sample.txt").write_text("3 0.50 0.45 0.30 0.25\n8 0.68 0.72 0.12 0.16\n", encoding="utf-8")
        return images, labels

    def test_pairs_size_classes_and_seed_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images, labels = self._make_input(root)
            config_a = AugmentConfig(
                images, labels, root / "out-a", copies_per_image=3,
                rotate_enabled=False, hflip_enabled=True, vflip_enabled=True,
                brightness_enabled=True, brightness_min=70, brightness_max=130,
                noise_enabled=True, noise_strength=4, seed=711,
            )
            config_b = AugmentConfig(
                images, labels, root / "out-b", copies_per_image=3,
                rotate_enabled=False, hflip_enabled=True, vflip_enabled=True,
                brightness_enabled=True, brightness_min=70, brightness_max=130,
                noise_enabled=True, noise_strength=4, seed=711,
            )
            report_a = ImageGenerator(config_a).run()
            report_b = ImageGenerator(config_b).run()
            self.assertEqual(report_a["generated"], 3)
            self.assertEqual(report_b["generated"], 3)
            output_a, output_b = root / "out-a", root / "out-b"
            files_a = sorted((output_a / "images").glob("*.jpg"))
            files_b = sorted((output_b / "images").glob("*.jpg"))
            self.assertEqual([p.name for p in files_a], [p.name for p in files_b])
            for image_a, image_b in zip(files_a, files_b):
                self.assertEqual(image_a.read_bytes(), image_b.read_bytes())
                image = cv2.imread(str(image_a))
                self.assertEqual(image.shape[:2], (640, 640))
                label_a = output_a / "labels" / f"{image_a.stem}.txt"
                label_b = output_b / "labels" / f"{image_a.stem}.txt"
                self.assertTrue(label_a.is_file())
                self.assertEqual(label_a.name, image_a.stem + ".txt")
                self.assertEqual(label_a.read_bytes(), label_b.read_bytes())
                boxes = parse_yolo_labels(label_a)
                self.assertEqual([box.class_id for box in boxes], [3, 8])
                for box in boxes:
                    self.assertGreaterEqual(box.x_center - box.width / 2, 0)
                    self.assertGreaterEqual(box.y_center - box.height / 2, 0)
                    self.assertLessEqual(box.x_center + box.width / 2, 1)
                    self.assertLessEqual(box.y_center + box.height / 2, 1)

    def test_stop_finishes_current_pair_and_leaves_no_partial_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images, labels = self._make_input(root)
            stop_event = threading.Event()
            config = AugmentConfig(
                images, labels, root / "out", copies_per_image=10,
                rotate_enabled=False, hflip_enabled=True, brightness_enabled=False, seed=9,
            )

            def stop_after_first_pair(_image: Path, _label: Path) -> None:
                stop_event.set()

            report = ImageGenerator(config, stop_event=stop_event, on_preview=stop_after_first_pair).run()
            self.assertTrue(report["halted"])
            self.assertEqual(report["generated"], 1)
            image_files = list((root / "out" / "images").glob("*.jpg"))
            label_files = list((root / "out" / "labels").glob("*.txt"))
            self.assertEqual(len(image_files), 1)
            self.assertEqual(len(label_files), 1)
            self.assertEqual(image_files[0].stem, label_files[0].stem)

    def test_load_source_requires_matching_txt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            images, labels = self._make_input(Path(temp))
            (labels / "sample.txt").unlink()
            with self.assertRaises(InputError):
                load_source(images / "sample.png", labels)

    def test_retry_limit_skips_a_cropped_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images, labels = self._make_input(root)
            (labels / "sample.txt").write_text("0 0.04 0.5 0.06 0.2\n", encoding="utf-8")
            log: list[str] = []
            config = AugmentConfig(
                images, labels, root / "out", copies_per_image=1,
                rotate_enabled=True, rotate_limit=90.0, hflip_enabled=False,
                vflip_enabled=False, brightness_enabled=False, noise_enabled=False,
                seed=72, max_retries=2,
            )
            report = ImageGenerator(config, on_log=log.append).run()
            self.assertEqual(report["generated"], 0)
            self.assertEqual(report["skipped_outputs"], 1)
            self.assertEqual(len(list((root / "out" / "images").glob("*.jpg"))), 0)
            self.assertTrue(any("2回試行後" in entry for entry in log))


if __name__ == "__main__":
    unittest.main()

