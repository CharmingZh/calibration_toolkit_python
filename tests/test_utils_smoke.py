from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calib.core.board_spec import DEFAULT_BOARD_SPEC
from calib.utils.board import create_circle_board_object_points
from calib.utils.images import read_image_robust


def test_create_circle_board_object_points_basic() -> None:
    count = 41
    points = create_circle_board_object_points(count)

    assert points.shape == (count, 3)
    assert np.allclose(points[:, 2], 0.0)

    spacing = DEFAULT_BOARD_SPEC.center_spacing_mm
    xs = points[:, 0]
    ys = points[:, 1]

    assert np.isclose(xs.max() - xs.min(), spacing * 5, atol=1e-4)
    assert np.isclose(ys.max() - ys.min(), spacing * 6, atol=1e-4)


def test_read_image_robust_roundtrip(tmp_path: Path) -> None:
    img = np.random.randint(0, 255, size=(32, 24), dtype=np.uint8)
    target = tmp_path / "sample.png"
    assert cv2.imwrite(str(target), img)

    loaded = read_image_robust(target)
    assert loaded is not None
    assert loaded.shape == img.shape
    assert loaded.dtype == np.uint8
    assert np.mean(np.abs(loaded.astype(np.int16) - img.astype(np.int16))) < 1
