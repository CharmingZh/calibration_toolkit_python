# -*- coding: utf-8 -*-
"""Generate calibration quality visualizations.

Outputs:
1. Board coverage heatmap (where the calibration board appears within the image frame).
2. Interpolated reprojection error heatmap across the full image.
3. Scatter plot of per-point reprojection error vectors (dx, dy).

Usage example:
    python tools/evaluation/generate_visuals.py \
        --input data/raw/calibration/calib_25 \
        --calibration-json outputs/calibration/latest/camera_calibration_improved.json \
        --output outputs/calibration/latest/visuals
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Times New Roman"


def find_project_root(start_dir: str, marker_rel: str = os.path.join("calib", "__init__.py")) -> str:
    """Locate repository root so we can import project modules."""
    cur = os.path.abspath(start_dir)
    last = None
    while cur and cur != last:
        if os.path.isfile(os.path.join(cur, marker_rel)):
            return cur
        last = cur
        cur = os.path.dirname(cur)
    raise RuntimeError("Unable to locate project root; ensure calib/__init__.py exists")


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = find_project_root(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from calib.core.calib_core import Calibrator  # noqa: E402
from calib.core.board_spec import BoardSpec, DEFAULT_BOARD_SPEC  # noqa: E402
from calib.utils.board import (  # noqa: E402
    create_circle_board_object_points_adaptive,
    extract_image_points_ordered,
)
from calib.utils.images import read_image_robust  # noqa: E402


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_image_path(input_dir: str, stem: str) -> str | None:
    candidates = [
        os.path.join(input_dir, stem + ext)
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dng", ".DNG")
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def collect_detection_data(
    calibrator: Calibrator,
    image_stems: List[str],
    input_dir: str,
    board_spec: BoardSpec,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[np.ndarray],
    Tuple[int, int],
]:
    """Run detection and gather reprojection statistics for each image."""

    all_image_points: List[np.ndarray] = []
    all_errors_px: List[np.ndarray] = []
    all_errors_mm: List[np.ndarray] = []
    all_object_points: List[np.ndarray] = []
    all_quads: List[np.ndarray] = []

    width = height = None

    for stem in image_stems:
        image_path = resolve_image_path(input_dir, stem)
        if image_path is None:
            logging.warning("Image file not found: %s", stem)
            continue

        gray = read_image_robust(image_path)
        if gray is None:
            logging.warning("Failed to read image: %s", image_path)
            continue

        if width is None:
            height, width = gray.shape

        result = calibrator.process_gray(gray)
        if result is None or not result.small_numbered:
            logging.warning("Calibration board detection failed: %s", stem)
            continue

        image_points = extract_image_points_ordered(result)
        if image_points is None or len(image_points) == 0:
            logging.warning("Failed to extract calibration points: %s", stem)
            continue

        object_points = create_circle_board_object_points_adaptive(
            len(image_points), board_spec.center_spacing_mm, board_spec
        ).reshape(-1, 3)

        object_points_cv = object_points.reshape(-1, 1, 3)
        image_points_cv = image_points.reshape(-1, 1, 2)

        ok, rvec, tvec = cv2.solvePnP(
            object_points_cv,
            image_points_cv,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            logging.warning("solvePnP failed: %s", stem)
            continue

        projected_points, _ = cv2.projectPoints(
            object_points_cv,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        projected_points = projected_points.reshape(-1, 2)
        errors_px = image_points - projected_points

        errors_mm = np.zeros_like(errors_px)
        for idx, (pt_obj, err_px) in enumerate(zip(object_points, errors_px)):
            base = projected_points[idx]
            axis_x_obj = (pt_obj + np.array([1.0, 0.0, 0.0], dtype=np.float32)).reshape(1, 1, 3)
            axis_y_obj = (pt_obj + np.array([0.0, 1.0, 0.0], dtype=np.float32)).reshape(1, 1, 3)

            proj_x, _ = cv2.projectPoints(axis_x_obj, rvec, tvec, camera_matrix, dist_coeffs)
            proj_y, _ = cv2.projectPoints(axis_y_obj, rvec, tvec, camera_matrix, dist_coeffs)

            base_vec = base.reshape(2)
            J = np.column_stack([
                (proj_x.reshape(2) - base_vec),
                (proj_y.reshape(2) - base_vec),
            ])

            if np.linalg.cond(J) < 1e8:
                try:
                    errors_mm[idx] = np.linalg.solve(J, err_px)
                except np.linalg.LinAlgError:
                    errors_mm[idx] = np.array([np.nan, np.nan], dtype=np.float32)
            else:
                errors_mm[idx] = np.array([np.nan, np.nan], dtype=np.float32)

        all_image_points.append(image_points)
        all_errors_px.append(errors_px)
        all_errors_mm.append(errors_mm)
        all_object_points.append(object_points)
        all_quads.append(np.asarray(result.quad, dtype=np.float32))

    if not all_image_points:
        raise RuntimeError("No valid calibration images were processed")

    image_points_concat = np.concatenate(all_image_points, axis=0)
    errors_px_concat = np.concatenate(all_errors_px, axis=0)
    errors_mm_concat = np.concatenate(all_errors_mm, axis=0)
    object_points_concat = np.concatenate(all_object_points, axis=0)

    assert width is not None and height is not None
    image_size = (width, height)

    return (
        image_points_concat,
        errors_px_concat,
        errors_mm_concat,
        object_points_concat,
        all_quads,
        image_size,
    )


def compute_board_coverage_heatmap(
    image_size: Tuple[int, int],
    quads: List[np.ndarray],
    smoothing_kernel: int = 45,
) -> np.ndarray:
    width, height = image_size
    accumulator = np.zeros((height, width), dtype=np.float32)

    for quad in quads:
        if quad.shape[0] < 4:
            continue
        polygon = np.round(quad).astype(np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        mask = np.zeros_like(accumulator, dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 1)
        accumulator += mask.astype(np.float32)

    if accumulator.max() > 0:
        accumulator /= accumulator.max()

    if smoothing_kernel > 0:
        kernel = (smoothing_kernel | 1)
        accumulator = cv2.GaussianBlur(accumulator, (kernel, kernel), 0)

    return accumulator


def compute_error_interpolation(
    extent: Tuple[float, float, float, float],
    points: np.ndarray,
    values: np.ndarray,
    output_size: Tuple[int, int],
    grid_size: Tuple[int, int] = (160, 120),
    power: float = 2.0,
) -> np.ndarray:
    min_x, max_x, min_y, max_y = extent
    grid_w, grid_h = grid_size

    xs = np.linspace(min_x, max_x, grid_w)
    ys = np.linspace(min_y, max_y, grid_h)
    grid_x, grid_y = np.meshgrid(xs, ys)

    heatmap = np.zeros((grid_h, grid_w), dtype=np.float64)
    weight_sum = np.zeros_like(heatmap)

    eps = 1e-6
    for (px, py), value in zip(points, values):
        dx = grid_x - px
        dy = grid_y - py
        dist_sq = dx * dx + dy * dy + eps
        weights = 1.0 / np.power(dist_sq, power / 2.0)
        heatmap += weights * value
        weight_sum += weights

    mask = weight_sum > 0
    heatmap[mask] = heatmap[mask] / weight_sum[mask]
    heatmap[~mask] = np.nan

    width_out, height_out = output_size
    heatmap_full = cv2.resize(heatmap, (width_out, height_out), interpolation=cv2.INTER_CUBIC)

    return heatmap_full


def plot_and_save_board_heatmap(heatmap: np.ndarray, image_size: Tuple[int, int], output_path: str) -> None:
    width, height = image_size
    plt.figure(figsize=(6, 8))
    plt.imshow(heatmap, cmap="magma", origin="upper", extent=[0, width, height, 0])
    plt.colorbar(label="Coverage ratio (0-1)")
    plt.title("Calibration board coverage heatmap")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_and_save_error_heatmap_pixels(
    heatmap: np.ndarray, extent: Tuple[float, float, float, float], output_path: str
) -> None:
    min_x, max_x, min_y, max_y = extent
    plt.figure(figsize=(6, 8))
    im = plt.imshow(heatmap, cmap="inferno", origin="upper", extent=[min_x, max_x, max_y, min_y])
    cbar = plt.colorbar(im, label="Reprojection error (pixels)")
    cbar.ax.set_ylabel("Reprojection error (pixels)")
    plt.title("Per-pixel reprojection error heatmap")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_and_save_error_heatmap_board(
    heatmap: np.ndarray,
    extent: Tuple[float, float, float, float],
    grid_points: np.ndarray,
    output_path: str,
) -> None:
    min_x, max_x, min_y, max_y = extent
    plt.figure(figsize=(6, 6))
    im = plt.imshow(heatmap, cmap="viridis", origin="lower", extent=[min_x, max_x, min_y, max_y])
    cbar = plt.colorbar(im, label="Δr (mm)")
    cbar.ax.set_ylabel("Δr (mm)")

    if grid_points.size:
        plt.scatter(grid_points[:, 0], grid_points[:, 1], marker="*", c="#ffb000", s=70, edgecolors="k")

    plt.title("Δr heatmap (mm) with corrected grid points")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.grid(color="white", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_and_save_error_scatter(errors: np.ndarray, output_path: str) -> None:
    magnitudes = np.linalg.norm(errors, axis=1)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(errors[:, 0], errors[:, 1], c=magnitudes, cmap="viridis", alpha=0.8, s=20)
    plt.axhline(0, color="lightgray", linewidth=1)
    plt.axvline(0, color="lightgray", linewidth=1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Δx (pixels)")
    plt.ylabel("Δy (pixels)")
    plt.title("Reprojection error distribution (dx, dy)")
    plt.colorbar(scatter, label="Error magnitude (pixels)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def load_calibration_data(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], BoardSpec, Tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("success", False):
        raise RuntimeError("Calibration result indicates failure; please verify the JSON file")

    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    image_size = tuple(int(v) for v in data.get("image_size", []))
    stems = data.get("successful_images")
    if not stems:
        raise RuntimeError("JSON file is missing the 'successful_images' field")

    board_info = data.get("board_spec", {})
    board_spec = BoardSpec(
        small_diameter_mm=float(board_info.get("small_diameter_mm", DEFAULT_BOARD_SPEC.small_diameter_mm)),
        center_spacing_mm=float(board_info.get("center_spacing_mm", DEFAULT_BOARD_SPEC.center_spacing_mm)),
    )

    return camera_matrix, dist_coeffs, stems, board_spec, image_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate calibration quality visualizations")
    parser.add_argument("--input", default="data/raw/calibration/calib_25", help="Directory containing calibration images")
    parser.add_argument(
        "--calibration-json",
        default="outputs/calibration/latest/camera_calibration_improved.json",
        help="Path to calibration result JSON file",
    )
    parser.add_argument(
        "--output",
        default="outputs/calibration/latest/visuals",
        help="Output directory for generated figures",
    )
    parser.add_argument("--log", default="INFO", help="Logging level (DEBUG/INFO/WARNING)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    camera_matrix, dist_coeffs, stems, board_spec, image_size = load_calibration_data(args.calibration_json)

    ensure_dir(args.output)

    calibrator = Calibrator()

    logging.info("Processing %d images to generate statistics...", len(stems))
    (
        image_points,
        errors_px,
        errors_mm,
        object_points,
        quads,
        detected_size,
    ) = collect_detection_data(
        calibrator,
        stems,
        args.input,
        board_spec,
        camera_matrix,
        dist_coeffs,
    )

    if image_size and image_size != detected_size:
        logging.warning(
            "Image size in JSON %s differs from detected %s; using detected size",
            image_size,
            detected_size,
        )
        image_size = detected_size
    elif not image_size:
        image_size = detected_size

    board_heatmap = compute_board_coverage_heatmap(image_size, quads)
    board_heatmap_path = os.path.join(args.output, "board_coverage_heatmap.png")
    plot_and_save_board_heatmap(board_heatmap, image_size, board_heatmap_path)
    logging.info("Saved calibration board coverage heatmap: %s", board_heatmap_path)

    pixel_extent = (0.0, image_size[0] - 1.0, 0.0, image_size[1] - 1.0)
    pixel_magnitudes = np.linalg.norm(errors_px, axis=1)
    pixel_heatmap = compute_error_interpolation(pixel_extent, image_points, pixel_magnitudes, image_size)
    pixel_heatmap_path = os.path.join(args.output, "reprojection_error_heatmap_pixels.png")
    plot_and_save_error_heatmap_pixels(pixel_heatmap, pixel_extent, pixel_heatmap_path)
    logging.info("Saved per-pixel reprojection error heatmap: %s", pixel_heatmap_path)

    board_points = object_points[:, :2]
    board_magnitudes = np.linalg.norm(errors_mm, axis=1)
    finite_mask = np.isfinite(board_magnitudes)
    board_points = board_points[finite_mask]
    board_magnitudes = board_magnitudes[finite_mask]

    if board_points.size == 0:
        raise RuntimeError("Cannot generate board-plane heatmap: all error values are invalid")

    board_extent = (
        float(board_points[:, 0].min()),
        float(board_points[:, 0].max()),
        float(board_points[:, 1].min()),
        float(board_points[:, 1].max()),
    )
    board_heatmap_mm = compute_error_interpolation(board_extent, board_points, board_magnitudes, (600, 600))
    board_heatmap_mm_path = os.path.join(args.output, "reprojection_error_heatmap_board.png")
    unique_grid_points = np.unique(board_points, axis=0)
    plot_and_save_error_heatmap_board(board_heatmap_mm, board_extent, unique_grid_points, board_heatmap_mm_path)
    logging.info("Saved board-plane reprojection error heatmap: %s", board_heatmap_mm_path)

    scatter_path = os.path.join(args.output, "reprojection_error_scatter.png")
    plot_and_save_error_scatter(errors_px, scatter_path)
    logging.info("Saved reprojection error scatter plot: %s", scatter_path)


if __name__ == "__main__":
    main()
