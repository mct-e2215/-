"""Launch the YOLO augmentation GUI."""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from yolo_augmenter.gui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO 学習画像データ拡張")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

