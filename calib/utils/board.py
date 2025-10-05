# -*- coding: utf-8 -*-
"""Board geometry helpers shared by calibration scripts."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..core.board_spec import BoardSpec, DEFAULT_BOARD_SPEC
from ..core.types import BoardResult

__all__ = [
    "create_circle_board_object_points",
    "create_circle_board_object_points_adaptive",
    "extract_image_points_ordered",
]


def create_circle_board_object_points(
    count: int,
    spacing: Optional[float] = None,
    board_spec: Optional[BoardSpec] = None,
) -> np.ndarray:
    """Generate physical board coordinates for the standard 7×6 circle grid.

    Parameters
    ----------
    count:
        Number of points required.
    spacing:
        Optional override for circle centre spacing in millimetres. Ignored if
        ``board_spec`` is provided.
    board_spec:
        Board specification describing circle spacing. Falls back to
        ``DEFAULT_BOARD_SPEC`` when unspecified.
    """

    base_spec = board_spec or DEFAULT_BOARD_SPEC
    spacing_mm = float(base_spec.center_spacing_mm if board_spec else (spacing or base_spec.center_spacing_mm))

    coords = []
    for r in range(6, -1, -1):
        for c in range(5, -1, -1):
            if r == 3 and c == 3:
                continue
            coords.append([c * spacing_mm, r * spacing_mm, 0.0])
            if len(coords) >= count:
                return np.array(coords, dtype=np.float32)
    return np.array(coords, dtype=np.float32)


def create_circle_board_object_points_adaptive(
    count: int,
    spacing: Optional[float] = None,
    board_spec: Optional[BoardSpec] = None,
) -> np.ndarray:
    """Alias retained for backwards compatibility."""

    return create_circle_board_object_points(count, spacing=spacing, board_spec=board_spec)


def extract_image_points_ordered(result: Optional[BoardResult]) -> Optional[np.ndarray]:
    """Extract small-circle image coordinates in canonical order.

    Returns the detected points in the original image coordinate system when a
    valid homography is available; otherwise the rectified coordinates are
    returned.
    """

    if result is None or not result.small_numbered:
        return None

    numbered = sorted(result.small_numbered, key=lambda item: item.seq)
    points_rect = np.array([[pt.x, pt.y] for pt in numbered], dtype=np.float32)

    H = np.asarray(getattr(result, "homography", None), dtype=np.float64)
    if H.shape != (3, 3):
        return points_rect

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return points_rect

    pts_h = cv2.perspectiveTransform(points_rect.reshape(-1, 1, 2), H_inv.astype(np.float64))
    return pts_h.reshape(-1, 2).astype(np.float32)
