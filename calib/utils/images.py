# -*- coding: utf-8 -*-
"""Image I/O helpers used by multiple scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

try:  # Pillow is optional but commonly available via environment.yml
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - Pillow is optional
    Image = None  # type: ignore

PathLike = Union[str, Path]


def read_image_robust(path: PathLike) -> Optional[np.ndarray]:
    """Read a grayscale image from disk with graceful fallbacks.

    Parameters
    ----------
    path:
        Input filepath. DNG files are preferentially decoded with Pillow to
        avoid OpenCV failures. For other formats we try OpenCV first, then fall
        back to Pillow when available.

    Returns
    -------
    Optional[np.ndarray]
        Grayscale uint8 image on success, otherwise ``None``.
    """

    fp = Path(path)
    path_lower = fp.suffix.lower()

    # Prefer Pillow for DNGs because OpenCV frequently fails to decode them.
    if path_lower == ".dng" and Image is not None:
        try:
            return np.array(Image.open(fp).convert("L"))
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("[WARN] Pillow failed to read DNG %s: %s", fp, exc)

    # OpenCV fast-path for standard formats.
    try:
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("[WARN] OpenCV failed to read %s: %s", fp, exc)

    # Final fallback: attempt Pillow for any remaining case.
    if Image is not None:
        try:
            return np.array(Image.open(fp).convert("L"))
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("[WARN] Pillow fallback failed %s: %s", fp, exc)

    return None
