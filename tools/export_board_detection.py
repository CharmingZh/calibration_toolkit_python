#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-image board detection helper for the C++ GUI.

This script reuses the Python calibration pipeline to detect the calibration
board layout on a single image. The result is emitted as JSON to stdout so the
C++ application can stay perfectly aligned with the reference Python
implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - handled at runtime
    print(json.dumps({
        "success": False,
        "message": f"OpenCV import failed: {exc}"
    }, ensure_ascii=False), file=sys.stdout)
    sys.exit(0)

try:
    from calib.core.calib_core import Calibrator
    from calib.core.board_spec import BoardSpec, DEFAULT_BOARD_SPEC
    from calib.utils.board import (
        create_circle_board_object_points_adaptive,
        extract_image_points_ordered,
    )
    from calib.utils.images import read_image_robust
except Exception as exc:  # pragma: no cover - handled at runtime
    print(json.dumps({
        "success": False,
        "message": f"Failed to import calibration modules: {exc}"
    }, ensure_ascii=False), file=sys.stdout)
    sys.exit(0)


def detect_board(image_path: str, board_spec: BoardSpec) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "image": image_path,
        "name": os.path.splitext(os.path.basename(image_path))[0],
        "success": False,
        "message": "",  # will be filled later
        "elapsed_ms": 0.0,
        "image_size": None,
    }

    gray = read_image_robust(image_path)
    if gray is None:
        entry["message"] = "failed_to_read_image"
        return entry

    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)

    h, w = gray.shape[:2]
    entry["image_size"] = [int(w), int(h)]

    calibrator = Calibrator()
    start = time.perf_counter()
    try:
        board_result = calibrator.process_gray(gray)
    except Exception as exc:  # pragma: no cover - runtime issue handling
        entry["message"] = f"calibrator_exception: {exc}"
        entry["elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)
        return entry

    entry["elapsed_ms"] = float((time.perf_counter() - start) * 1000.0)

    if board_result is None:
        entry["message"] = "detector_returned_none"
        return entry

    image_points = extract_image_points_ordered(board_result)
    if image_points is None or len(image_points) == 0:
        entry["message"] = "no_ordered_points"
        return entry

    image_points = np.asarray(image_points, dtype=np.float32)
    object_points = create_circle_board_object_points_adaptive(
        len(image_points),
        spacing=board_spec.center_spacing_mm,
        board_spec=board_spec,
    )

    entry.update({
        "success": True,
        "message": "ok",
        "count": int(len(image_points)),
        "image_points": image_points.tolist(),
        "object_points": object_points.astype(float).tolist(),
        "homography": getattr(board_result, "homography", None),
        "quad": getattr(board_result, "quad", None),
    })
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect calibration board using the reference Python pipeline.")
    parser.add_argument("--image", required=True, help="Path to the image to analyse")
    parser.add_argument("--small-diameter", dest="small_diameter", type=float,
                        default=DEFAULT_BOARD_SPEC.small_diameter_mm, help="Small circle diameter in millimetres")
    parser.add_argument("--circle-spacing", dest="circle_spacing", type=float,
                        default=DEFAULT_BOARD_SPEC.center_spacing_mm, help="Circle centre spacing in millimetres")
    args = parser.parse_args()

    board_spec = BoardSpec(
        small_diameter_mm=float(args.small_diameter),
        center_spacing_mm=float(args.circle_spacing),
    )

    result = detect_board(args.image, board_spec)
    result["board_spec"] = {
        "small_diameter_mm": board_spec.small_diameter_mm,
        "center_spacing_mm": board_spec.center_spacing_mm,
    }

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
