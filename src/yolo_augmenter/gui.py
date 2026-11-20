"""PySide6 desktop interface for YOLO dataset augmentation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import random
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .engine import (
    AugmentConfig,
    CandidateRejected,
    ImageGenerator,
    InputError,
    create_candidate,
    draw_boxes,
    list_image_files,
    load_source,
    validate_config,
    parse_yolo_labels,
)


class GenerationWorker(QThread):
    progress_changed = Signal(int, int, int, int, str)
    log_message = Signal(str)
    preview_ready = Signal(str, str)

    def __init__(self, config: AugmentConfig) -> None:
        super().__init__()
        self.config = config
        self.stop_event = threading.Event()
        self.report: dict | None = None
        self.error: str | None = None
        self._last_progress_emit = 0.0
        self._last_preview_emit = 0.0
        self._last_success_log_emit = 0.0
        self._last_file_index = 0

    def _progress(self, current: int, total: int, generated: int, target: int, name: str) -> None:
        now = time.monotonic()
        if current != self._last_file_index or now - self._last_progress_emit >= 0.2 or generated >= target:
            self.progress_changed.emit(current, total, generated, target, name)
            self._last_progress_emit = now
            self._last_file_index = current

    def _log(self, message: str) -> None:
        if message.startswith("生成:"):
            now = time.monotonic()
            if now - self._last_success_log_emit < 0.5:
                return
            self._last_success_log_emit = now
        self.log_message.emit(message)

    def _preview(self, image_path: Path, label_path: Path) -> None:
        now = time.monotonic()
        if now - self._last_preview_emit >= 0.25:
            self.preview_ready.emit(str(image_path), str(label_path))
            self._last_preview_emit = now

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        try:
            self.report = ImageGenerator(
                self.config,
                stop_event=self.stop_event,
                on_progress=self._progress,
                on_log=self._log,
                on_preview=self._preview,
            ).run()
        except Exception as exc:  # surfaced in the GUI after the worker exits
            self.error = str(exc)


class FolderPicker(QWidget):
    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.caption = caption
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("フォルダを選択")
        self.browse_button = QPushButton("参照…")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.path_edit, 1)
        row.addWidget(self.browse_button)
        self.browse_button.clicked.connect(self._browse)

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, self.caption, self.path_edit.text())
        if selected:
            self.path_edit.setText(selected)

    def path(self) -> Path:
        return Path(self.path_edit.text().strip()).expanduser()


class PreviewTile(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None, large: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("previewTile")
        self.title_label = QLabel(title)
        self.title_label.setObjectName("previewTitle")
        self.image_label = QLabel("画像を選択してください")
        self.image_label.setObjectName("previewImage")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(160, 130) if not large else self.image_label.setMinimumSize(260, 260)
        self.image_label.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)

    def show_image(self, image: np.ndarray) -> None:
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.image_label.setPixmap(
            QPixmap.fromImage(qimage).scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def clear_image(self, message: str = "画像を選択してください") -> None:
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(message)

    def resizeEvent(self, event) -> None:  # keep preview pixmaps fitted as the window changes
        pixmap = self.image_label.pixmap()
        if pixmap and not pixmap.isNull():
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO 学習画像データ拡張")
        self.resize(1380, 960)
        self.setMinimumSize(1120, 760)
        self.worker: GenerationWorker | None = None
        self._selected_paths: list[Path] = []
        self._last_source_image: np.ndarray | None = None
        self._build_ui()
        self._apply_style()
        self._folder_timer = QTimer(self)
        self._folder_timer.setSingleShot(True)
        self._folder_timer.setInterval(300)
        self._folder_timer.timeout.connect(self._refresh_input_choices)
        self.images_picker.path_edit.textChanged.connect(lambda _text: self._folder_timer.start())
        self.labels_picker.path_edit.textChanged.connect(lambda _text: self._folder_timer.start())
        self._refresh_input_choices()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        title = QLabel("YOLO 学習画像データ拡張")
        title.setObjectName("appTitle")
        subtitle = QLabel("画像とYOLOラベルを同じ変換で拡張します。切れるボックスがある候補は自動で再生成します。")
        subtitle.setObjectName("appSubtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        folders = QGroupBox("フォルダ")
        folder_form = QFormLayout(folders)
        self.images_picker = FolderPicker("入力画像フォルダを選択")
        self.labels_picker = FolderPicker("入力ラベルフォルダを選択")
        self.output_picker = FolderPicker("出力フォルダを選択")
        folder_form.addRow("入力画像", self.images_picker)
        folder_form.addRow("入力ラベル", self.labels_picker)
        folder_form.addRow("出力先", self.output_picker)
        left_layout.addWidget(folders)

        transformations = QGroupBox("変換設定")
        transform_form = QFormLayout(transformations)
        self.rotate_check = QCheckBox("回転を適用")
        self.rotate_check.setChecked(True)
        self.rotate_angle = QDoubleSpinBox()
        self.rotate_angle.setRange(0.1, 180.0)
        self.rotate_angle.setDecimals(1)
        self.rotate_angle.setSingleStep(1.0)
        self.rotate_angle.setValue(30.0)
        angle_row = QWidget()
        angle_layout = QHBoxLayout(angle_row)
        angle_layout.setContentsMargins(0, 0, 0, 0)
        angle_layout.addWidget(self.rotate_check)
        angle_layout.addWidget(QLabel("最大角度"))
        angle_layout.addWidget(self.rotate_angle)
        angle_layout.addWidget(QLabel("°"))
        angle_layout.addStretch(1)
        transform_form.addRow("回転", angle_row)

        self.hflip_check = QCheckBox("左右反転を適用")
        self.hflip_check.setChecked(True)
        transform_form.addRow("左右反転", self.hflip_check)
        self.vflip_check = QCheckBox("上下反転を適用")
        transform_form.addRow("上下反転", self.vflip_check)

        self.brightness_check = QCheckBox("明度変更を適用")
        self.brightness_check.setChecked(True)
        self.brightness_min = QSpinBox()
        self.brightness_min.setRange(1, 300)
        self.brightness_min.setValue(50)
        self.brightness_max = QSpinBox()
        self.brightness_max.setRange(1, 300)
        self.brightness_max.setValue(150)
        brightness_row = QWidget()
        brightness_layout = QHBoxLayout(brightness_row)
        brightness_layout.setContentsMargins(0, 0, 0, 0)
        brightness_layout.addWidget(self.brightness_check)
        brightness_layout.addWidget(QLabel("最小"))
        brightness_layout.addWidget(self.brightness_min)
        brightness_layout.addWidget(QLabel("%　最大"))
        brightness_layout.addWidget(self.brightness_max)
        brightness_layout.addWidget(QLabel("%"))
        brightness_layout.addStretch(1)
        transform_form.addRow("明度", brightness_row)

        self.noise_check = QCheckBox("ノイズを追加")
        self.noise_strength = QSpinBox()
        self.noise_strength.setRange(1, 30)
        self.noise_strength.setValue(8)
        noise_row = QWidget()
        noise_layout = QHBoxLayout(noise_row)
        noise_layout.setContentsMargins(0, 0, 0, 0)
        noise_layout.addWidget(self.noise_check)
        noise_layout.addWidget(QLabel("強さ"))
        noise_layout.addWidget(self.noise_strength)
        noise_layout.addWidget(QLabel("(最大30)"))
        noise_layout.addStretch(1)
        transform_form.addRow("ノイズ", noise_row)
        left_layout.addWidget(transformations)

        generation = QGroupBox("生成設定")
        generation_form = QFormLayout(generation)
        self.copies_spin = QSpinBox()
        self.copies_spin.setRange(1, 10000)
        self.copies_spin.setValue(10)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-2147483648, 2147483647)
        self.seed_spin.setValue(42)
        self.retries_spin = QSpinBox()
        self.retries_spin.setRange(1, 500)
        self.retries_spin.setValue(50)
        generation_form.addRow("生成枚数／元画像", self.copies_spin)
        generation_form.addRow("乱数シード", self.seed_spin)
        generation_form.addRow("最大再試行回数", self.retries_spin)
        left_layout.addWidget(generation)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("生成開始")
        self.start_button.setObjectName("primaryButton")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        left_layout.addLayout(button_row)
        left_layout.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setWidget(left_content)
        splitter.addWidget(left_scroll)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(8, 0, 0, 0)
        preview_layout.setSpacing(10)
        preview_group = QGroupBox("プレビュー")
        preview_group_layout = QVBoxLayout(preview_group)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("入力画像"))
        self.image_combo = QComboBox()
        self.image_combo.setMinimumWidth(260)
        selector_row.addWidget(self.image_combo, 1)
        preview_group_layout.addLayout(selector_row)
        main_tiles = QHBoxLayout()
        self.input_preview = PreviewTile("入力画像", large=True)
        self.output_preview = PreviewTile("変換プレビュー（ボックス・クラスID付き）", large=True)
        main_tiles.addWidget(self.input_preview, 1)
        main_tiles.addWidget(self.output_preview, 1)
        preview_group_layout.addLayout(main_tiles, 1)
        self.preview_info = QLabel("入力画像とラベルを選ぶとプレビューを表示します。")
        self.preview_info.setWordWrap(True)
        preview_group_layout.addWidget(self.preview_info)
        preview_layout.addWidget(preview_group, 3)

        brightness_group = QGroupBox("明度プレビュー")
        brightness_layout = QHBoxLayout(brightness_group)
        self.brightness_min_preview = PreviewTile("最小明度")
        self.brightness_original_preview = PreviewTile("元画像 100%")
        self.brightness_max_preview = PreviewTile("最大明度")
        brightness_layout.addWidget(self.brightness_min_preview, 1)
        brightness_layout.addWidget(self.brightness_original_preview, 1)
        brightness_layout.addWidget(self.brightness_max_preview, 1)
        preview_layout.addWidget(brightness_group, 2)
        splitter.addWidget(preview_panel)
        splitter.setSizes([480, 850])
        root.addWidget(splitter, 1)

        self.stats_label = QLabel("入力画像 0枚　|　目標生成 0枚　|　生成済み 0枚")
        self.status_label = QLabel("待機中")
        root.addWidget(self.stats_label)
        root.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("処理状況、エラー、スキップ理由を表示します。")
        self.log.setMinimumHeight(120)
        root.addWidget(self.log)

        self.start_button.clicked.connect(self._start_generation)
        self.stop_button.clicked.connect(self._request_stop)
        self.image_combo.currentIndexChanged.connect(self._update_preview)
        self.copies_spin.valueChanged.connect(self._update_counts)
        for widget in (
            self.rotate_check, self.hflip_check, self.vflip_check, self.brightness_check,
            self.noise_check,
        ):
            widget.toggled.connect(self._update_preview)
        for widget in (
            self.rotate_angle, self.brightness_min, self.brightness_max, self.noise_strength,
            self.seed_spin,
        ):
            widget.valueChanged.connect(self._update_preview)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f6fa; color: #1d2939; font-size: 10pt; }
            QGroupBox { background: #ffffff; border: 1px solid #d8dee9; border-radius: 9px;
                        margin-top: 10px; padding: 12px 10px 10px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #344054; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                background: #ffffff; border: 1px solid #cbd5e1; border-radius: 5px; padding: 5px; }
            QPushButton { background: #e9edf5; border: 1px solid #d2dae6; border-radius: 6px;
                          padding: 8px 12px; font-weight: 600; }
            QPushButton:hover { background: #dfe7f3; }
            QPushButton:disabled { color: #98a2b3; }
            QPushButton#primaryButton { background: #2457c5; color: #ffffff; border: 0; }
            QPushButton#primaryButton:hover { background: #1947a8; }
            QLabel#appTitle { font-size: 19pt; font-weight: 700; color: #102a56; }
            QLabel#appSubtitle { color: #667085; padding-bottom: 4px; }
            QFrame#previewTile { background: #ffffff; border: 1px solid #e0e5ed; border-radius: 7px; }
            QLabel#previewTitle { font-weight: 600; color: #475467; padding: 3px 5px; }
            QLabel#previewImage { background: #f8fafc; border-radius: 5px; color: #98a2b3; }
            QProgressBar { background: #e9edf5; border: 0; border-radius: 5px; height: 13px; text-align: center; }
            QProgressBar::chunk { background: #3574df; border-radius: 5px; }
            """
        )

    def _config(self, require_output: bool = True) -> AugmentConfig:
        if not self.images_picker.path_edit.text().strip() or not self.labels_picker.path_edit.text().strip():
            raise InputError("入力画像フォルダと入力ラベルフォルダを指定してください。")
        if require_output and not self.output_picker.path_edit.text().strip():
            raise InputError("出力フォルダを指定してください。")
        return AugmentConfig(
            images_dir=self.images_picker.path(),
            labels_dir=self.labels_picker.path(),
            output_dir=self.output_picker.path() if self.output_picker.path_edit.text().strip() else Path.cwd() / "output",
            copies_per_image=self.copies_spin.value(),
            rotate_enabled=self.rotate_check.isChecked(),
            rotate_limit=self.rotate_angle.value(),
            hflip_enabled=self.hflip_check.isChecked(),
            vflip_enabled=self.vflip_check.isChecked(),
            brightness_enabled=self.brightness_check.isChecked(),
            brightness_min=float(self.brightness_min.value()),
            brightness_max=float(self.brightness_max.value()),
            noise_enabled=self.noise_check.isChecked(),
            noise_strength=self.noise_strength.value(),
            seed=self.seed_spin.value(),
            max_retries=self.retries_spin.value(),
        )

    def _refresh_input_choices(self) -> None:
        previous = self.image_combo.currentData()
        try:
            paths = list_image_files(self.images_picker.path())
        except (InputError, OSError):
            paths = []
        self._selected_paths = paths
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        for path in paths:
            self.image_combo.addItem(path.name, str(path))
        if previous:
            index = self.image_combo.findData(previous)
            if index >= 0:
                self.image_combo.setCurrentIndex(index)
        self.image_combo.blockSignals(False)
        self._update_counts()
        self._update_preview()

    def _update_counts(self) -> None:
        count = len(self._selected_paths)
        target = count * self.copies_spin.value()
        current_generated = self.worker.report["generated"] if self.worker and self.worker.report else 0
        self.stats_label.setText(
            f"入力画像 {count:,}枚　|　目標生成 {target:,}枚　|　生成済み {current_generated:,}枚"
        )

    def _update_preview(self, *_args) -> None:
        path_text = self.image_combo.currentData()
        if not path_text:
            self.input_preview.clear_image()
            self.output_preview.clear_image()
            for tile in (self.brightness_min_preview, self.brightness_original_preview, self.brightness_max_preview):
                tile.clear_image()
            return
        image_path = Path(path_text)
        try:
            source = load_source(image_path, self.labels_picker.path())
            image = source.image
            self._last_source_image = image
            self.input_preview.show_image(image)
            low = np.clip(image.astype(np.float32) * (self.brightness_min.value() / 100.0), 0, 255).astype(np.uint8)
            high = np.clip(image.astype(np.float32) * (self.brightness_max.value() / 100.0), 0, 255).astype(np.uint8)
            self.brightness_min_preview.title_label.setText(f"最小明度 {self.brightness_min.value()}%")
            self.brightness_original_preview.title_label.setText("元画像 100%")
            self.brightness_max_preview.title_label.setText(f"最大明度 {self.brightness_max.value()}%")
            self.brightness_min_preview.show_image(low)
            self.brightness_original_preview.show_image(image)
            self.brightness_max_preview.show_image(high)
            config = self._config(require_output=False)
            preview_rng = random.Random(
                int.from_bytes(
                    hashlib.sha256(f"{config.seed}:{image_path.name}:preview".encode("utf-8")).digest()[:8],
                    "big",
                )
            )
            candidate = None
            reason = ""
            for attempt in range(config.max_retries):
                try:
                    candidate = create_candidate(source.image, source.boxes, config, preview_rng, variant_index=attempt + 1)
                    break
                except CandidateRejected as exc:
                    reason = str(exc)
            if candidate is None:
                raise InputError(f"変換プレビューを作れませんでした。{reason}")
            self.output_preview.show_image(draw_boxes(candidate.image, candidate.boxes))
            class_ids = ", ".join(str(box.class_id) for box in candidate.boxes) or "ラベルなし"
            transform_text = ", ".join(candidate.plan.operations)
            if "rotate" in candidate.plan.operations:
                transform_text += f"（{candidate.plan.angle:.1f}°）"
            if "brightness" in candidate.plan.operations:
                transform_text += f" 明度{candidate.plan.brightness * 100:.0f}%"
            self.preview_info.setText(f"変換: {transform_text}　|　物体数: {len(candidate.boxes)}　|　クラスID: {class_ids}")
        except Exception as exc:
            self._last_source_image = None
            self.input_preview.clear_image("入力画像を読み込めません")
            self.output_preview.clear_image("プレビューを表示できません")
            for tile in (self.brightness_min_preview, self.brightness_original_preview, self.brightness_max_preview):
                tile.clear_image("プレビューを表示できません")
            self.preview_info.setText(str(exc))

    def _start_generation(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        try:
            config = self._config()
            validate_config(config)
            images = list_image_files(config.images_dir)
            if not images:
                raise InputError(f"入力画像がありません: {config.images_dir}")
            if not config.labels_dir.is_dir():
                raise InputError(f"入力ラベルフォルダが存在しません: {config.labels_dir}")
            if config.output_dir == config.images_dir or config.output_dir == config.labels_dir:
                raise InputError("出力先は入力画像・ラベルフォルダと分けてください。")
        except (InputError, OSError) as exc:
            QMessageBox.warning(self, "設定を確認してください", str(exc))
            return

        self.progress_bar.setValue(0)
        self.log.appendPlainText("--- 生成を開始します ---")
        self.status_label.setText("初期化中…")
        self.worker = GenerationWorker(config)
        self.worker.progress_changed.connect(self._on_progress)
        self.worker.log_message.connect(self.log.appendPlainText)
        self.worker.preview_ready.connect(self._show_saved_preview)
        self.worker.finished.connect(self._on_finished)
        self._set_busy(True)
        self.worker.start()

    def _set_busy(self, busy: bool) -> None:
        self.start_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        for widget in (
            self.images_picker, self.labels_picker, self.output_picker,
            self.rotate_check, self.rotate_angle, self.hflip_check, self.vflip_check,
            self.brightness_check, self.brightness_min, self.brightness_max,
            self.noise_check, self.noise_strength, self.copies_spin, self.seed_spin, self.retries_spin,
        ):
            widget.setEnabled(not busy)

    def _request_stop(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.stop_button.setEnabled(False)
            self.status_label.setText("停止要求を受け付けました。現在の画像とラベルを保存後に停止します…")

    def _on_progress(self, current_file: int, total_files: int, generated: int, target: int, source_name: str) -> None:
        percentage = int(current_file * 100 / total_files) if total_files else 0
        self.progress_bar.setValue(percentage)
        self.status_label.setText(
            f"処理中: {source_name}　|　入力 {current_file}/{total_files}　|　生成 {generated}/{target}"
        )
        self.stats_label.setText(
            f"入力画像 {total_files:,}枚　|　目標生成 {target:,}枚　|　生成済み {generated:,}枚"
        )

    def _show_saved_preview(self, image_path: str, label_path: str) -> None:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return
        try:
            boxes = parse_yolo_labels(Path(label_path))
        except InputError:
            return
        self.output_preview.show_image(draw_boxes(image, boxes))
        class_ids = ", ".join(str(box.class_id) for box in boxes) or "ラベルなし"
        self.preview_info.setText(f"出力プレビュー: {Path(image_path).name}　|　物体数: {len(boxes)}　|　クラスID: {class_ids}")

    def _on_finished(self) -> None:
        self._set_busy(False)
        if not self.worker:
            return
        if self.worker.error:
            self.status_label.setText("処理を終了しました（エラー）")
            self.log.appendPlainText(f"エラー: {self.worker.error}")
            QMessageBox.critical(self, "生成エラー", self.worker.error)
            return
        report = self.worker.report or {}
        generated, target = int(report.get("generated", 0)), int(report.get("target", 0))
        shortage = max(0, target - generated)
        halted = bool(report.get("halted", False))
        self.progress_bar.setValue(100 if not halted else self.progress_bar.value())
        result = f"生成済み {generated:,}/{target:,}枚"
        if report.get("skipped_sources") or report.get("skipped_outputs"):
            result += f"　|　入力スキップ {report.get('skipped_sources', 0)}件　|　生成スキップ {report.get('skipped_outputs', 0)}枚"
        if shortage:
            result += f"　|　目標未達 {shortage:,}枚"
        if halted:
            result = "停止しました。" + result
        else:
            result = "完了。" + result
        self.status_label.setText(result)
        self.log.appendPlainText(result)
        self._update_counts()

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "処理中", "停止ボタンで処理を停止してから閉じてください。")
            event.ignore()
            return
        event.accept()

