"""Image augmentation and YOLO annotation transformation engine."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import re
import tempfile
import threading
from typing import Callable

import cv2
import numpy as np


CANVAS_SIZE = 640
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
JPEG_QUALITY = 95


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Decode an image through Python's Unicode-aware file API (Windows-safe)."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if encoded.size == 0:
        return None
    try:
        return cv2.imdecode(encoded, flags)
    except cv2.error:
        return None


class InputError(ValueError):
    """An input folder, image, or annotation does not meet the requirements."""


class CandidateRejected(ValueError):
    """A sampled transform would crop at least one annotated object."""


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class AugmentConfig:
    images_dir: Path
    labels_dir: Path
    output_dir: Path
    copies_per_image: int = 10
    rotate_enabled: bool = True
    rotate_limit: float = 30.0
    hflip_enabled: bool = True
    vflip_enabled: bool = False
    brightness_enabled: bool = True
    brightness_min: float = 50.0
    brightness_max: float = 150.0
    noise_enabled: bool = False
    noise_strength: int = 8
    seed: int = 42
    max_retries: int = 50


@dataclass(frozen=True)
class TransformPlan:
    operations: tuple[str, ...]
    angle: float = 0.0
    hflip: bool = False
    vflip: bool = False
    brightness: float = 1.0
    noise_sigma: float = 0.0


@dataclass(frozen=True)
class Candidate:
    image: np.ndarray
    boxes: tuple[YoloBox, ...]
    plan: TransformPlan


@dataclass(frozen=True)
class SourceImage:
    image_path: Path
    label_path: Path
    image: np.ndarray
    boxes: tuple[YoloBox, ...]


ProgressCallback = Callable[[int, int, int, int, str], None]
LogCallback = Callable[[str], None]
PreviewCallback = Callable[[Path, Path], None]


def validate_config(config: AugmentConfig) -> None:
    if config.copies_per_image < 1:
        raise InputError("生成枚数は1枚以上にしてください。")
    if config.max_retries < 1:
        raise InputError("最大再試行回数は1回以上にしてください。")
    if config.rotate_enabled and not (0.0 < config.rotate_limit <= 180.0):
        raise InputError("回転角度の最大値は0より大きく180度以下にしてください。")
    if config.brightness_enabled and not (
        0.0 < config.brightness_min <= config.brightness_max <= 300.0
    ):
        raise InputError("明度は0%より大きく、最大300%の範囲で指定してください。")
    if config.noise_enabled and not (1 <= config.noise_strength <= 30):
        raise InputError("ノイズの強さは1～30の範囲で指定してください。")
    meaningful = (
        config.rotate_enabled
        or config.hflip_enabled
        or config.vflip_enabled
        or (config.brightness_enabled and (config.brightness_min != 100.0 or config.brightness_max != 100.0))
        or (config.noise_enabled and config.noise_strength > 0)
    )
    if not meaningful:
        raise InputError("変換機能を1つ以上ONにしてください。未変換画像の大量生成を防ぐためです。")


def parse_yolo_labels(path: Path) -> tuple[YoloBox, ...]:
    """Read strict, normalized YOLO rows; class IDs are retained verbatim."""
    boxes: list[YoloBox] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise InputError(f"ラベルを読み込めません: {exc}") from exc
    except UnicodeError as exc:
        raise InputError(f"ラベルがUTF-8形式ではありません: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise InputError(f"{path.name}:{line_number}: 5列（class x y width height）が必要です。")
        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise InputError(f"{path.name}:{line_number}: クラスIDは0以上の整数で指定してください。") from exc
        if class_id < 0:
            raise InputError(f"{path.name}:{line_number}: クラスIDは0以上にしてください。")
        try:
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise InputError(f"{path.name}:{line_number}: 座標が数値ではありません。") from exc
        values = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in values):
            raise InputError(f"{path.name}:{line_number}: 座標にNaNまたは無限大は指定できません。")
        if width <= 0.0 or height <= 0.0:
            raise InputError(f"{path.name}:{line_number}: 幅と高さは0より大きくしてください。")
        left, top = x_center - width / 2.0, y_center - height / 2.0
        right, bottom = x_center + width / 2.0, y_center + height / 2.0
        if left < 0.0 or top < 0.0 or right > 1.0 or bottom > 1.0:
            raise InputError(f"{path.name}:{line_number}: バウンディングボックス全体を0～1内に収めてください。")
        boxes.append(YoloBox(class_id, x_center, y_center, width, height))
    return tuple(boxes)


def load_source(image_path: Path, labels_dir: Path) -> SourceImage:
    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.is_file():
        raise InputError(f"対応するラベルファイルがありません: {label_path}")
    image = read_image(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise InputError(f"画像を読み込めません: {image_path}")
    height, width = image.shape[:2]
    if (width, height) != (CANVAS_SIZE, CANVAS_SIZE):
        raise InputError(f"画像サイズが{CANVAS_SIZE}×{CANVAS_SIZE}ではありません: {width}×{height}")
    boxes = parse_yolo_labels(label_path)
    return SourceImage(image_path, label_path, image, boxes)


def list_image_files(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        raise InputError(f"入力画像フォルダが存在しません: {images_dir}")
    return sorted(
        (path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: (path.stem.casefold(), path.name.casefold()),
    )


def _enabled_operations(config: AugmentConfig) -> list[str]:
    operations: list[str] = []
    if config.rotate_enabled:
        operations.append("rotate")
    if config.hflip_enabled:
        operations.append("hflip")
    if config.vflip_enabled:
        operations.append("vflip")
    if config.brightness_enabled and (config.brightness_min != 100.0 or config.brightness_max != 100.0):
        operations.append("brightness")
    if config.noise_enabled:
        operations.append("noise")
    return operations


def sample_plan(
    config: AugmentConfig,
    rng: random.Random,
    variant_index: int = 0,
) -> TransformPlan:
    """Sample a varied subset, ensuring single and combined patterns appear."""
    operations = _enabled_operations(config)
    if not operations:
        raise InputError("変換機能を1つ以上ONにしてください。")
    if len(operations) == 1:
        count = 1
    elif variant_index == 0:
        count = 1
    elif variant_index == 1:
        count = rng.randint(2, len(operations))
    else:
        count = rng.randint(1, len(operations))
    selected = tuple(rng.sample(operations, count))
    angle = rng.uniform(-config.rotate_limit, config.rotate_limit) if "rotate" in selected else 0.0
    if "rotate" in selected and abs(angle) < 1e-7:
        angle = config.rotate_limit if rng.random() < 0.5 else -config.rotate_limit
    brightness = 1.0
    if "brightness" in selected:
        low, high = config.brightness_min / 100.0, config.brightness_max / 100.0
        brightness = rng.uniform(low, high) if high > low else low
        # Avoid sampling the one neutral value when alternatives exist.
        if abs(brightness - 1.0) < 1e-9 and (high - low) > 1e-9:
            brightness = low if abs(low - 1.0) > 1e-9 else high
    return TransformPlan(
        operations=selected,
        angle=angle,
        hflip="hflip" in selected,
        vflip="vflip" in selected,
        brightness=brightness,
        noise_sigma=float(config.noise_strength) if "noise" in selected else 0.0,
    )


def affine_for_plan(plan: TransformPlan, width: int = CANVAS_SIZE, height: int = CANVAS_SIZE) -> np.ndarray:
    """Build the exact source-to-destination matrix used for pixels and boxes."""
    matrix = np.eye(3, dtype=np.float64)
    if "rotate" in plan.operations:
        radians = math.radians(plan.angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        # Expand by |cos|+|sin| so the rotated square covers all output corners.
        cover_scale = abs(cosine) + abs(sine)
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        rotation = np.array(
            [
                [cover_scale * cosine, cover_scale * sine, (1.0 - cover_scale * cosine) * cx - cover_scale * sine * cy],
                [-cover_scale * sine, cover_scale * cosine, cover_scale * sine * cx + (1.0 - cover_scale * cosine) * cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        matrix = rotation @ matrix
    if plan.hflip:
        matrix = np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ matrix
    if plan.vflip:
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]]) @ matrix
    return matrix[:2, :]


def transform_boxes(
    boxes: tuple[YoloBox, ...] | list[YoloBox],
    matrix: np.ndarray,
    width: int = CANVAS_SIZE,
    height: int = CANVAS_SIZE,
) -> tuple[YoloBox, ...]:
    """Transform all four box corners; reject the whole set if any box is cropped."""
    transformed: list[YoloBox] = []
    homogeneous = np.vstack((matrix, np.array([0.0, 0.0, 1.0])))
    for box in boxes:
        x1 = (box.x_center - box.width / 2.0) * width
        x2 = (box.x_center + box.width / 2.0) * width
        y1 = (box.y_center - box.height / 2.0) * height
        y2 = (box.y_center + box.height / 2.0) * height
        # YOLO coordinates describe image edges; OpenCV's affine uses pixel centers.
        corners = np.array(
            [[x1 - 0.5, y1 - 0.5, 1.0], [x2 - 0.5, y1 - 0.5, 1.0],
             [x2 - 0.5, y2 - 0.5, 1.0], [x1 - 0.5, y2 - 0.5, 1.0]],
            dtype=np.float64,
        )
        points = corners @ homogeneous.T
        xs, ys = points[:, 0] + 0.5, points[:, 1] + 0.5
        left, right = float(xs.min()), float(xs.max())
        top, bottom = float(ys.min()), float(ys.max())
        epsilon = 1e-7
        # Snap only floating-point noise at a mathematically exact canvas edge.
        if abs(left) <= epsilon:
            left = 0.0
        if abs(top) <= epsilon:
            top = 0.0
        if abs(right - width) <= epsilon:
            right = float(width)
        if abs(bottom - height) <= epsilon:
            bottom = float(height)
        if left < 0.0 or top < 0.0 or right > width or bottom > height:
            raise CandidateRejected(
                f"class {box.class_id} のボックスが画像外にはみ出します。別の変換を試します。"
            )
        if right <= left or bottom <= top:
            raise CandidateRejected(f"class {box.class_id} のボックスが空になりました。")
        transformed.append(
            YoloBox(
                box.class_id,
                (left + right) / (2.0 * width),
                (top + bottom) / (2.0 * height),
                (right - left) / width,
                (bottom - top) / height,
            )
        )
    return tuple(transformed)


def apply_plan(
    image: np.ndarray,
    boxes: tuple[YoloBox, ...],
    plan: TransformPlan,
    rng: random.Random,
) -> Candidate:
    height, width = image.shape[:2]
    geometry_changed = "rotate" in plan.operations or plan.hflip or plan.vflip
    if geometry_changed:
        matrix = affine_for_plan(plan, width, height)
        transformed_boxes = transform_boxes(boxes, matrix, width, height)
        output = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        # Photometric operations must leave annotation values exactly unchanged.
        transformed_boxes = boxes
        output = image.copy()
    if "brightness" in plan.operations:
        output = np.clip(output.astype(np.float32) * plan.brightness, 0.0, 255.0).astype(np.uint8)
    if "noise" in plan.operations and plan.noise_sigma > 0.0:
        noise_seed = rng.getrandbits(64)
        generator = np.random.default_rng(noise_seed)
        noise = generator.normal(0.0, plan.noise_sigma, output.shape).astype(np.float32)
        output = np.clip(output.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return Candidate(output, transformed_boxes, plan)


def create_candidate(
    image: np.ndarray,
    boxes: tuple[YoloBox, ...],
    config: AugmentConfig,
    rng: random.Random,
    variant_index: int = 0,
) -> Candidate:
    plan = sample_plan(config, rng, variant_index)
    return apply_plan(image, boxes, plan, rng)


def boxes_to_text(boxes: tuple[YoloBox, ...]) -> str:
    return "".join(
        f"{box.class_id} {box.x_center:.8f} {box.y_center:.8f} {box.width:.8f} {box.height:.8f}\n"
        for box in boxes
    )


def _next_sequence(image_dir: Path, label_dir: Path, stem: str) -> int:
    pattern = re.compile(rf"^{re.escape(stem)}_aug_(\d+)\.(?:jpg|jpeg|png|txt)$", re.IGNORECASE)
    maximum = 0
    for directory in (image_dir, label_dir):
        if directory.exists():
            for path in directory.iterdir():
                match = pattern.match(path.name)
                if match:
                    maximum = max(maximum, int(match.group(1)))
    return maximum + 1


def _atomic_save_pair(image: np.ndarray, image_path: Path, label_text: str, label_path: Path) -> None:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not success:
        raise OSError(f"JPEGエンコードに失敗しました: {image_path.name}")
    temp_paths: list[Path] = []
    installed: list[Path] = []
    try:
        for folder, payload in ((image_path.parent, encoded.tobytes()), (label_path.parent, label_text.encode("utf-8"))):
            with tempfile.NamedTemporaryFile(prefix=".yolo_aug_", suffix=".tmp", dir=folder, delete=False) as stream:
                temp_path = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temp_paths.append(temp_path)
        # Stop requests are checked before this pair commit, so a committed sample is complete.
        if image_path.exists() or label_path.exists():
            raise FileExistsError(f"出力ファイル名が既に存在します: {image_path.name}")
        os.replace(temp_paths[0], image_path)
        installed.append(image_path)
        os.replace(temp_paths[1], label_path)
        installed.append(label_path)
    except Exception:
        for path in installed:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


class ImageGenerator:
    def __init__(
        self,
        config: AugmentConfig,
        stop_event: threading.Event | None = None,
        on_progress: ProgressCallback | None = None,
        on_log: LogCallback | None = None,
        on_preview: PreviewCallback | None = None,
    ) -> None:
        self.config = config
        self.stop_event = stop_event or threading.Event()
        self.on_progress = on_progress or (lambda *_: None)
        self.on_log = on_log or (lambda *_: None)
        self.on_preview = on_preview or (lambda *_: None)

    def _check_folders(self) -> None:
        validate_config(self.config)
        if not self.config.images_dir.is_dir():
            raise InputError(f"入力画像フォルダが存在しません: {self.config.images_dir}")
        if not self.config.labels_dir.is_dir():
            raise InputError(f"入力ラベルフォルダが存在しません: {self.config.labels_dir}")
        output_images = (self.config.output_dir / "images").resolve()
        output_labels = (self.config.output_dir / "labels").resolve()
        if output_images == self.config.images_dir.resolve() or output_labels == self.config.labels_dir.resolve():
            raise InputError("出力先は入力フォルダと分けてください。入力画像・ラベルの上書きを防ぐためです。")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        for folder in (self.config.output_dir, output_images, output_labels):
            try:
                with tempfile.TemporaryFile(dir=folder):
                    pass
            except OSError as exc:
                raise InputError(f"出力フォルダに書き込めません: {folder} ({exc})") from exc

    def run(self) -> dict[str, int | bool]:
        self._check_folders()
        image_paths = list_image_files(self.config.images_dir)
        if not image_paths:
            raise InputError(f"入力画像がありません: {self.config.images_dir}")
        stem_groups: dict[str, list[Path]] = {}
        for image_path in image_paths:
            stem_groups.setdefault(image_path.stem.casefold(), []).append(image_path)
        duplicates = {stem: paths for stem, paths in stem_groups.items() if len(paths) > 1}
        duplicate_paths = {path for paths in duplicates.values() for path in paths}
        target = len(image_paths) * self.config.copies_per_image
        generated = 0
        skipped_sources = 0
        skipped_outputs = 0
        rng = random.Random(self.config.seed)
        out_images = self.config.output_dir / "images"
        out_labels = self.config.output_dir / "labels"

        for source_index, image_path in enumerate(image_paths, start=1):
            if self.stop_event.is_set():
                break
            current_name = image_path.name
            if image_path in duplicate_paths:
                error = f"同じファイル名（拡張子を除く）の画像が複数あります: {', '.join(p.name for p in duplicates[image_path.stem.casefold()])}"
                self.on_log(f"スキップ: {error}")
                skipped_sources += 1
                self.on_progress(source_index, len(image_paths), generated, target, current_name)
                continue
            try:
                source = load_source(image_path, self.config.labels_dir)
            except InputError as exc:
                self.on_log(f"スキップ: {image_path.name}: {exc}")
                skipped_sources += 1
                self.on_progress(source_index, len(image_paths), generated, target, current_name)
                continue

            sequence = _next_sequence(out_images, out_labels, image_path.stem)
            for output_index in range(self.config.copies_per_image):
                if self.stop_event.is_set():
                    break
                candidate: Candidate | None = None
                last_reason = "有効な変換候補を作れませんでした。"
                for retry in range(self.config.max_retries):
                    if self.stop_event.is_set():
                        break
                    try:
                        candidate = create_candidate(
                            source.image,
                            source.boxes,
                            self.config,
                            rng,
                            variant_index=output_index + retry,
                        )
                        break
                    except CandidateRejected as exc:
                        last_reason = str(exc)
                if self.stop_event.is_set():
                    break
                if candidate is None:
                    skipped_outputs += 1
                    self.on_log(
                        f"スキップ: {image_path.name} の{output_index + 1}枚目は{self.config.max_retries}回試行後も生成できません。{last_reason}"
                    )
                    self.on_progress(source_index, len(image_paths), generated, target, current_name)
                    continue

                while True:
                    image_name = f"{image_path.stem}_aug_{sequence:03d}.jpg"
                    label_name = f"{image_path.stem}_aug_{sequence:03d}.txt"
                    final_image, final_label = out_images / image_name, out_labels / label_name
                    if not final_image.exists() and not final_label.exists():
                        break
                    sequence += 1
                try:
                    _atomic_save_pair(candidate.image, final_image, boxes_to_text(candidate.boxes), final_label)
                except OSError as exc:
                    raise InputError(f"出力を書き込めません ({final_image.name}): {exc}") from exc
                sequence += 1
                generated += 1
                self.on_preview(final_image, final_label)
                self.on_log(
                    f"生成: {image_name} | {', '.join(candidate.plan.operations)}"
                    + (f" | 回転 {candidate.plan.angle:.1f}°" if "rotate" in candidate.plan.operations else "")
                    + (f" | 明度 {candidate.plan.brightness * 100:.0f}%" if "brightness" in candidate.plan.operations else "")
                )
                self.on_progress(source_index, len(image_paths), generated, target, current_name)

        halted = self.stop_event.is_set()
        return {
            "total_inputs": len(image_paths),
            "generated": generated,
            "target": target,
            "skipped_sources": skipped_sources,
            "skipped_outputs": skipped_outputs,
            "halted": halted,
        }


def draw_boxes(image: np.ndarray, boxes: tuple[YoloBox, ...]) -> np.ndarray:
    """Draw each transformed box and its class ID on a BGR preview copy."""
    preview = image.copy()
    for box in boxes:
        x1 = int(round((box.x_center - box.width / 2.0) * CANVAS_SIZE))
        y1 = int(round((box.y_center - box.height / 2.0) * CANVAS_SIZE))
        x2 = int(round((box.x_center + box.width / 2.0) * CANVAS_SIZE))
        y2 = int(round((box.y_center + box.height / 2.0) * CANVAS_SIZE))
        hue = (box.class_id * 47 + 23) % 180
        color = tuple(int(channel) for channel in cv2.cvtColor(np.uint8([[[hue, 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(preview, f"class {box.class_id}", (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return preview

