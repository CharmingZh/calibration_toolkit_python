# -*- coding: utf-8 -*-
"""Board detection & calibration evaluation tool."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def configure_matplotlib_fonts() -> Tuple[bool, List[str]]:
    preferred = [
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Heiti SC",
        "SimHei",
        "Microsoft YaHei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return True, [name]
    return False, ["DejaVu Sans"]


HAS_CJK_FONT, FONT_SANS_SERIF = configure_matplotlib_fonts()
plt.rcParams["font.sans-serif"] = FONT_SANS_SERIF
plt.rcParams["axes.unicode_minus"] = False


logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)


def find_project_root(start_dir: Path, marker_rel: Path = Path("calib") / "__init__.py") -> Path:
    cur = start_dir
    last = None
    while cur != last:
        if (cur / marker_rel).is_file():
            return cur
        last = cur
        cur = cur.parent
    return start_dir


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = find_project_root(_THIS_DIR)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from calib.core.calib_core import (  # type: ignore  # noqa: E402
    Calibrator,
    DEFAULT_CONFIG,
    HIGH_RECALL_CONFIG,
    DetectionConfig,
    create_detection_config,
)
from calib.core.board_spec import BoardSpec, DEFAULT_BOARD_SPEC  # type: ignore  # noqa: E402
from calib.core.types import BoardResult  # type: ignore  # noqa: E402
from calib.utils.board import (  # type: ignore  # noqa: E402
    create_circle_board_object_points,
    extract_image_points_ordered,
)
from calib.utils.images import read_image_robust  # type: ignore  # noqa: E402
from calib.viz.viz_steps import Visualizer  # type: ignore  # noqa: E402


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img_min, img_max = float(np.min(img)), float(np.max(img))
    if img_max - img_min < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - img_min) / (img_max - img_min)
    return (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)


def _gray_to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(_ensure_uint8(img), cv2.COLOR_GRAY2BGR)
    return img


def save_debug_artifacts(target_root: Path, base: str, gray: np.ndarray,
                         debug: Dict[str, object], result: Optional[BoardResult],
                         expected_small: int, expected_big: int, success: bool) -> None:
    try:
        target_dir = target_root / base
        ensure_dir(target_dir)

        raw_path = target_dir / "00_raw.png"
        cv2.imwrite(str(raw_path), _gray_to_bgr(gray))

        edges = debug.get("hough_edges") if debug else None
        if isinstance(edges, np.ndarray):
            overlay = _gray_to_bgr(gray)
            mask = edges > 0
            overlay[mask] = (0, 0, 255)
            cv2.imwrite(str(target_dir / "01_hough_edges.png"), overlay)

        quad_overlay = None
        if debug:
            for key in ("quad_refined", "quad_detected", "hough_quad", "white_quad", "quad_expanded"):
                val = debug.get(key)
                if isinstance(val, np.ndarray):
                    quad_overlay = val
                    break
        if isinstance(quad_overlay, np.ndarray):
            quad_canvas = _gray_to_bgr(gray)
            pts = quad_overlay.reshape(-1, 2).astype(np.int32)
            cv2.polylines(quad_canvas, [pts], True, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(target_dir / "03_quad.png"), quad_canvas)

        mask = debug.get("white_mask") if debug else None
        if isinstance(mask, np.ndarray):
            mask_u8 = _ensure_uint8(mask)
            heat = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
            cv2.imwrite(str(target_dir / "02_white_region.png"), heat)

        rect = debug.get("rect") if debug else None
        if isinstance(rect, np.ndarray):
            cv2.imwrite(str(target_dir / "10_rect.png"), _gray_to_bgr(rect))

        rect_pre = debug.get("rect_pre") if debug else None
        if isinstance(rect_pre, np.ndarray):
            rect_pre_u8 = _ensure_uint8(rect_pre)
            cv2.imwrite(str(target_dir / "11_rect_pre.png"), _gray_to_bgr(rect_pre_u8))

            kp_canvas = cv2.cvtColor(rect_pre_u8, cv2.COLOR_GRAY2BGR)
            for kp in debug.get("keypoints_raw", []) if debug else []:
                try:
                    x = int(round(kp.get("x", 0.0)))
                    y = int(round(kp.get("y", 0.0)))
                    r = max(1, int(round(kp.get("size", 0.0) * 0.5)))
                    cv2.circle(kp_canvas, (x, y), r, (0, 255, 255), 1, cv2.LINE_AA)
                except Exception:  # pragma: no cover - debug best effort
                    continue
            cv2.imwrite(str(target_dir / "12_rect_keypoints.png"), kp_canvas)

            selected = debug.get("small_selected") if debug else None
            if isinstance(selected, list) and selected:
                small_canvas = kp_canvas.copy()
                for it in selected:
                    try:
                        cx = int(round(it.get("x", 0.0)))
                        cy = int(round(it.get("y", 0.0)))
                        r = max(1, int(round(it.get("r", 1.0))))
                        cv2.circle(small_canvas, (cx, cy), r, (0, 255, 0), 1, cv2.LINE_AA)
                    except Exception:
                        continue
                cv2.imwrite(str(target_dir / "13_rect_small_selected.png"), small_canvas)

            big_selected = debug.get("big_selected") if debug else None
            if isinstance(big_selected, list) and big_selected:
                big_canvas = kp_canvas.copy()
                for it in big_selected:
                    try:
                        cx = int(round(it.get("x", 0.0)))
                        cy = int(round(it.get("y", 0.0)))
                        r = max(2, int(round(it.get("r", 1.0))))
                        cv2.circle(big_canvas, (cx, cy), r, (0, 0, 255), 2, cv2.LINE_AA)
                    except Exception:
                        continue
                cv2.imwrite(str(target_dir / "14_rect_big_selected.png"), big_canvas)

        info = {
            "success": success,
            "stage": debug.get("stage") if debug else None,
            "fail_reason": debug.get("fail_reason") if debug else None,
            "detected_small": len(result.small_numbered) if (result and result.small_numbered) else len(debug.get("small_selected") or []) if debug else 0,
            "detected_big": len(result.big_circles) if (result and result.big_circles) else len(debug.get("big_selected") or []) if debug else 0,
            "expected_small": expected_small,
            "expected_big": expected_big,
        }
        (target_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - debug best effort
        logging.warning("[DEBUG] 保存调试可视化失败 %s: %s", base, exc)


@dataclass
class DetectionRecord:
    name: str
    success: bool
    reason: str
    elapsed_ms: float
    small_count: int
    big_count: int
    image_points: Optional[np.ndarray]
    object_points: Optional[np.ndarray]
    resolution: Optional[Tuple[int, int]]
    inlier_count: int = 0
    original_point_count: int = 0
    inlier_ratio: float = 1.0
    initial_rvec: Optional[np.ndarray] = field(default=None, repr=False)
    initial_tvec: Optional[np.ndarray] = field(default=None, repr=False)
    inlier_indices: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class CalibrationRecord:
    name: str
    mean_error: float
    median_error: float
    max_error: float
    std_error: float
    translation: Tuple[float, float, float]
    rotation_deg: Tuple[float, float, float]
    num_points: int
    rvec: Tuple[float, float, float]
    axis_error_mm: Optional[Tuple[float, float, float]] = None
    abs_error_mm: Optional[Tuple[float, float, float]] = None
    rel_error: Optional[Tuple[float, float, float]] = None
    residuals: Optional[List[float]] = None
    residual_vectors: Optional[List[Tuple[float, float]]] = None
    image_points: Optional[List[Tuple[float, float]]] = None
    inlier_count: int = 0


def serialize_calibration_record(record: CalibrationRecord) -> Dict[str, object]:
    data: Dict[str, object] = {
        "name": record.name,
        "mean_error_px": float(record.mean_error),
        "median_error_px": float(record.median_error),
        "max_error_px": float(record.max_error),
        "std_error_px": float(record.std_error),
        "translation_mm": [float(v) for v in record.translation],
        "rotation_deg": [float(v) for v in record.rotation_deg],
        "num_points": int(record.num_points),
        "inlier_count": int(getattr(record, "inlier_count", record.num_points)),
    }
    if record.axis_error_mm is not None:
        data["axis_error_mm"] = [float(v) for v in record.axis_error_mm]
    if record.abs_error_mm is not None:
        data["abs_error_mm"] = [float(v) for v in record.abs_error_mm]
    if record.rel_error is not None:
        data["rel_error_pct"] = [float(v) * 100.0 for v in record.rel_error]
    return data


def format_translation_stats_for_log(stats: dict) -> dict:
    return {
        "mean_translation_mm": [float(x) for x in stats["mean_translation"]],
        "axis_error_mean_mm": [float(x) for x in stats["axis_error_mean"]],
        "axis_error_std_mm": [float(x) for x in stats["axis_error_std"]],
        "abs_error_mean_mm": [float(x) for x in stats["abs_error_mean"]],
        "abs_error_std_mm": [float(x) for x in stats["abs_error_std"]],
        "rel_error_mean_pct": [float(x) * 100.0 for x in stats["rel_error_mean"]],
        "rel_error_std_pct": [float(x) * 100.0 for x in stats["rel_error_std"]],
        "per_sample": [
            {
                "name": entry["name"],
                "axis_error_mm": [float(v) for v in entry["axis_error_mm"]],
                "abs_error_mm": [float(v) for v in entry["abs_error_mm"]],
                "rel_error_pct": [float(v) * 100.0 for v in entry["rel_error"]],
            }
            for entry in stats.get("per_sample", [])
        ],
    }


def write_axis_error_log(output_dir: Path, entries: Dict[str, dict]) -> Optional[Path]:
    if not entries:
        return None
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": entries,
    }
    log_path = output_dir / "axis_error_log.json"
    ensure_dir(log_path.parent)
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return log_path


def split_per_image_stats(stats: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    kept: List[Dict[str, object]] = []
    removed: List[Dict[str, object]] = []
    for item in stats or []:
        entry = {
            "name": str(item.get("name", "")),
            "mean_error_px": float(item.get("mean", 0.0)),
            "median_error_px": float(item.get("median", 0.0)),
            "max_error_px": float(item.get("max", 0.0)),
            "std_error_px": float(item.get("std", 0.0)),
            "num_points": int(item.get("num_points", 0)),
        }
        status = str(item.get("status", "kept"))
        entry["status"] = status
        if status == "removed":
            removed.append(entry)
        else:
            kept.append(entry)
    return kept, removed


def build_calibration_export_payload(
    model: dict,
    board_spec: BoardSpec,
    image_size: Optional[Tuple[int, int]],
    kept_records: Sequence[DetectionRecord],
    per_image_stats: Sequence[Dict[str, object]],
    calib_records: Sequence[CalibrationRecord],
    summary: Optional[dict],
    translation_stats: Optional[dict],
    filter_report: Optional[dict],
) -> Dict[str, object]:
    if model is None:
        raise ValueError("model is required for calibration export")

    cam_raw = model.get("camera_matrix")
    dist_raw = model.get("dist_coeffs")
    if cam_raw is None or dist_raw is None:
        raise ValueError("model missing camera_matrix or dist_coeffs")

    camera_matrix = np.asarray(cam_raw, dtype=np.float64)
    dist_coeffs = np.asarray(dist_raw, dtype=np.float64).reshape(-1)

    export_size = image_size
    if export_size is None:
        for rec in kept_records:
            if rec.resolution is not None:
                export_size = rec.resolution
                break

    kept_names: List[str] = []
    for rec in kept_records:
        if rec.name not in kept_names:
            kept_names.append(rec.name)

    kept_stats, removed_stats = split_per_image_stats(per_image_stats or [])

    summary_payload: Dict[str, object] = {}
    if summary:
        for key, value in summary.items():
            if isinstance(value, np.ndarray):
                summary_payload[key] = value.astype(float).tolist()
            else:
                summary_payload[key] = value

    translation_payload: Optional[Dict[str, object]] = None
    if translation_stats:
        translation_payload = {
            "mean_translation_mm": np.asarray(translation_stats["mean_translation"]).round(3).tolist(),
            "axis_error_mean_mm": np.asarray(translation_stats["axis_error_mean"]).round(3).tolist(),
            "axis_error_std_mm": np.asarray(translation_stats["axis_error_std"]).round(3).tolist(),
            "abs_error_mean_mm": np.asarray(translation_stats["abs_error_mean"]).round(3).tolist(),
            "abs_error_std_mm": np.asarray(translation_stats["abs_error_std"]).round(3).tolist(),
            "rel_error_mean_pct": (np.asarray(translation_stats["rel_error_mean"]) * 100.0).round(3).tolist(),
            "rel_error_std_pct": (np.asarray(translation_stats["rel_error_std"]) * 100.0).round(3).tolist(),
            "per_sample": [
                {
                    "name": entry["name"],
                    "axis_error_mm": [float(v) for v in entry["axis_error_mm"]],
                    "abs_error_mm": [float(v) for v in entry["abs_error_mm"]],
                    "rel_error_pct": [float(v) * 100.0 for v in entry["rel_error"]],
                }
                for entry in translation_stats.get("per_sample", [])
            ],
        }

    filter_history: List[Dict[str, object]] = []
    if filter_report and filter_report.get("history"):
        for entry in filter_report["history"]:
            filter_history.append({
                "iteration": int(entry.get("iteration", len(filter_history) + 1)),
                "num_candidates": int(entry.get("num_candidates", 0)),
                "num_flagged": int(entry.get("num_flagged", 0)),
                "median_mean_px": float(entry.get("median_mean_px", 0.0)),
                "mad_mean_px": float(entry.get("mad_mean_px", 0.0)),
                "threshold_mean_px": float(entry.get("threshold_mean_px")) if entry.get("threshold_mean_px") is not None else None,
                "median_max_px": float(entry.get("median_max_px", 0.0)),
                "mad_max_px": float(entry.get("mad_max_px", 0.0)),
                "threshold_max_px": float(entry.get("threshold_max_px")) if entry.get("threshold_max_px") is not None else None,
            })

    removed_details: List[Dict[str, object]] = []
    if filter_report and filter_report.get("removed_samples"):
        for item in filter_report["removed_samples"]:
            removed_details.append({
                "name": str(item.get("name", "")),
                "mean_error_px": float(item.get("mean", 0.0)),
                "median_error_px": float(item.get("median", 0.0)),
                "max_error_px": float(item.get("max", 0.0)),
                "std_error_px": float(item.get("std", 0.0)),
                "num_points": int(item.get("num_points", 0)),
                "iteration_removed": int(item.get("iteration_removed", 0)),
            })

    payload: Dict[str, object] = {
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist(),
        "image_size": [int(export_size[0]), int(export_size[1])] if export_size else None,
        "num_images": len(kept_names),
        "reprojection_error": float(model.get("rms", 0.0)),
        "focal_length_x": float(camera_matrix[0, 0]),
        "focal_length_y": float(camera_matrix[1, 1]),
        "principal_point_x": float(camera_matrix[0, 2]),
        "principal_point_y": float(camera_matrix[1, 2]),
        "board_spec": board_spec.to_dict(),
        "successful_images": kept_names,
        "summary": summary_payload,
        "translation_stats": translation_payload,
        "per_sample_metrics": [serialize_calibration_record(rec) for rec in calib_records],
        "kept_samples": kept_stats,
        "removed_overview": removed_stats,
        "removed_samples": removed_details,
        "filter_history": filter_history,
    }

    return payload


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_override(expr: str) -> Tuple[str, object]:
    if "=" not in expr:
        raise ValueError(f"Override '{expr}' 缺少 '=' 分隔符")
    key, raw = expr.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override '{expr}' 的键为空")
    raw = raw.strip()
    if not raw:
        value: object = None
    else:
        try:
            value = ast.literal_eval(raw)
        except Exception:  # pylint: disable=broad-except
            lower = raw.lower()
            if lower in {"true", "false"}:
                value = lower == "true"
            else:
                value = raw
    return key, value


def build_detection_config(preset: str, overrides: Sequence[str]) -> Tuple[DetectionConfig, Dict[str, object]]:
    if preset == "high_recall":
        base = create_detection_config(**asdict(HIGH_RECALL_CONFIG))
    else:
        base = DetectionConfig()
    override_map: Dict[str, object] = {}
    for expr in overrides:
        key, value = _parse_override(expr)
        if not hasattr(base, key):
            raise AttributeError(f"检测配置不存在字段 '{key}'")
        setattr(base, key, value)
        override_map[key] = value
    return base, override_map


class ProgressPrinter:
    def __init__(self, total: int, width: int = 32) -> None:
        self.total = max(total, 1)
        self.width = width

    def update(self, current: int) -> None:
        filled = int(self.width * current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stdout.write(f"\r[Progress] [{bar}] {current}/{self.total}")
        sys.stdout.flush()
        if current >= self.total:
            sys.stdout.write("\n")


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        y = math.degrees(math.atan2(-R[2, 0], sy))
        z = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        x = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        y = math.degrees(math.atan2(-R[2, 0], sy))
        z = 0.0
    return (x, y, z)


def solve_pnp_metrics(object_points: np.ndarray, image_points: np.ndarray,
                      camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Optional[CalibrationRecord]:
    obj = np.asarray(object_points, dtype=np.float64)
    img = np.asarray(image_points, dtype=np.float64)
    if obj.shape[0] < 6:
        return None

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, camera_matrix, dist_coeffs,
            iterationsCount=200,
            reprojectionError=2.5,
            confidence=0.999,
            flags=cv2.SOLVEPNP_AP3P,
        )
    except cv2.error:
        success, rvec, tvec, inliers = False, None, None, None

    if not success or inliers is None or len(inliers) < 6:
        try:
            success, rvec, tvec = cv2.solvePnP(obj, img, camera_matrix, dist_coeffs,
                                               flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            success, rvec, tvec = False, None, None
        if not success:
            return None
        inliers = np.arange(obj.shape[0]).reshape(-1, 1)

    inlier_idx = inliers.reshape(-1)
    obj_in = obj[inlier_idx]
    img_in = img[inlier_idx]
    try:
        refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
            obj_in, img_in, camera_matrix, dist_coeffs,
            np.asarray(rvec, dtype=np.float64),
            np.asarray(tvec, dtype=np.float64))
    except cv2.error:
        refined_rvec = np.asarray(rvec, dtype=np.float64)
        refined_tvec = np.asarray(tvec, dtype=np.float64)

    projected, _ = cv2.projectPoints(obj, refined_rvec, refined_tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    diff = img - projected
    residuals = np.linalg.norm(diff, axis=1)
    mean_error = float(np.mean(residuals))
    median_error = float(np.median(residuals))
    max_error = float(np.max(residuals))
    std_error = float(np.std(residuals))

    R, _ = cv2.Rodrigues(refined_rvec)
    euler = rotation_matrix_to_euler(R)
    translation = tuple(float(x) for x in refined_tvec.reshape(-1))

    residuals_list = residuals.tolist()
    residual_vectors_list = [(float(vec[0]), float(vec[1])) for vec in diff.tolist()]
    image_points_list = img.tolist()
    rvec_tuple = tuple(float(x) for x in refined_rvec.reshape(-1))

    record = CalibrationRecord(
        name="",
        mean_error=mean_error,
        median_error=median_error,
        max_error=max_error,
        std_error=std_error,
        translation=translation,
        rotation_deg=euler,
        num_points=int(obj.shape[0]),
        rvec=rvec_tuple,
        residuals=residuals_list,
        residual_vectors=residual_vectors_list,
        image_points=[(float(pt[0]), float(pt[1])) for pt in image_points_list],
        inlier_count=int(len(inlier_idx)),
    )
    return record


def collect_image_paths(image_dir: Path, patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        paths.extend(sorted(image_dir.glob(pat)))
    return sorted(set(paths))


def summarize_detection(records: List[DetectionRecord]) -> dict:
    total = len(records)
    success = sum(1 for r in records if r.success)
    failure = total - success
    elapsed = [r.elapsed_ms for r in records]
    summary = {
        "total_images": total,
        "success": success,
        "failure": failure,
        "success_rate": (success / total * 100.0) if total else 0.0,
        "time_avg_ms": float(np.mean(elapsed)) if elapsed else 0.0,
        "time_min_ms": float(np.min(elapsed)) if elapsed else 0.0,
        "time_max_ms": float(np.max(elapsed)) if elapsed else 0.0,
    }
    return summary


def summarize_calibration(calib_records: List[CalibrationRecord]) -> dict:
    if not calib_records:
        return {}
    errors = [rec.mean_error for rec in calib_records]
    median_errors = [rec.median_error for rec in calib_records]
    max_errors = [rec.max_error for rec in calib_records]
    std_errors = [rec.std_error for rec in calib_records]
    translations = np.array([rec.translation for rec in calib_records])
    rotations = np.array([rec.rotation_deg for rec in calib_records])

    summary = {
        "num_samples": len(calib_records),
        "mean_reprojection_px": float(statistics.mean(errors)),
        "median_reprojection_px": float(statistics.mean(median_errors)),
        "max_reprojection_px": float(max(max_errors)),
        "std_reprojection_px": float(statistics.mean(std_errors)),
        "translation_mean_mm": translations.mean(axis=0).round(3).tolist(),
        "translation_std_mm": translations.std(axis=0).round(3).tolist(),
        "rotation_mean_deg": rotations.mean(axis=0).round(3).tolist(),
        "rotation_std_deg": rotations.std(axis=0).round(3).tolist(),
    }
    return summary


def compute_translation_error_stats(
    calib_records: List[CalibrationRecord],
    camera_matrix: np.ndarray,
    detection_lookup: Optional[Dict[str, DetectionRecord]] = None,
) -> Optional[dict]:
    if not calib_records or camera_matrix is None:
        return None

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])

    valid_records: List[CalibrationRecord] = []
    axis_errors_list: List[np.ndarray] = []
    per_sample: List[Dict[str, object]] = []

    for rec in calib_records:
        det_rec = detection_lookup.get(rec.name) if detection_lookup else None
        if det_rec is None or det_rec.object_points is None or rec.residual_vectors is None or rec.rvec is None:
            rec.axis_error_mm = None
            rec.abs_error_mm = None
            rec.rel_error = None
            continue

        object_points = np.asarray(det_rec.object_points, dtype=np.float64).reshape(-1, 3)
        residual_vectors = np.asarray(rec.residual_vectors, dtype=np.float64).reshape(-1, 2)
        if object_points.shape[0] == 0 or residual_vectors.shape[0] == 0:
            rec.axis_error_mm = None
            rec.abs_error_mm = None
            rec.rel_error = None
            continue

        if residual_vectors.shape[0] != object_points.shape[0]:
            count = min(residual_vectors.shape[0], object_points.shape[0])
            residual_vectors = residual_vectors[:count]
            object_points = object_points[:count]

        rvec = np.asarray(rec.rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(rec.translation, dtype=np.float64).reshape(3, 1)
        try:
            rotation, _ = cv2.Rodrigues(rvec)
        except cv2.error:  # pragma: no cover - defensive guard
            rec.axis_error_mm = None
            rec.abs_error_mm = None
            rec.rel_error = None
            continue

        camera_points = rotation @ object_points.T + tvec
        X = camera_points[0]
        Y = camera_points[1]
        Z = camera_points[2]
        valid_mask = Z > 1e-6
        if not np.any(valid_mask):
            rec.axis_error_mm = None
            rec.abs_error_mm = None
            rec.rel_error = None
            continue

        X = X[valid_mask]
        Y = Y[valid_mask]
        Z = Z[valid_mask]
        residual_use = residual_vectors[valid_mask]

        rows = residual_use.shape[0]
        A = np.zeros((rows * 2, 3), dtype=np.float64)
        b = residual_use.reshape(-1)
        inv_z = 1.0 / Z
        inv_z2 = inv_z * inv_z
        A[0::2, 0] = fx * inv_z
        A[1::2, 1] = fy * inv_z
        A[0::2, 2] = -fx * (X * inv_z2)
        A[1::2, 2] = -fy * (Y * inv_z2)

        try:
            delta, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - degeneracy guard
            rec.axis_error_mm = None
            rec.abs_error_mm = None
            rec.rel_error = None
            continue

        valid_records.append(rec)
        axis_errors_list.append(delta)
        per_sample.append({
            "name": rec.name,
            "axis_error_mm": [float(delta[0]), float(delta[1]), float(delta[2])],
        })

    if not valid_records or not axis_errors_list:
        return None

    axis_errors = np.vstack(axis_errors_list)
    abs_errors = np.abs(axis_errors)
    translations_arr = np.vstack([np.asarray(rec.translation, dtype=np.float64) for rec in valid_records])
    mean_translation = translations_arr.mean(axis=0)
    denom = np.maximum(np.abs(mean_translation), 1e-6)
    rel_errors = abs_errors / denom

    for rec, axis_err, abs_err, rel_err, entry in zip(valid_records, axis_errors, abs_errors, rel_errors, per_sample):
        rec.axis_error_mm = tuple(float(x) for x in axis_err)
        rec.abs_error_mm = tuple(float(x) for x in abs_err)
        rec.rel_error = tuple(float(x) for x in rel_err)
        entry["abs_error_mm"] = [float(x) for x in abs_err]
        entry["rel_error"] = [float(x) for x in rel_err]

    return {
        "mean_translation": mean_translation,
        "axis_errors": axis_errors,
        "abs_errors": abs_errors,
        "rel_errors": rel_errors,
        "axis_error_mean": axis_errors.mean(axis=0),
        "axis_error_std": axis_errors.std(axis=0),
        "abs_error_mean": abs_errors.mean(axis=0),
        "abs_error_std": abs_errors.std(axis=0),
        "rel_error_mean": rel_errors.mean(axis=0),
        "rel_error_std": rel_errors.std(axis=0),
        "per_sample": per_sample,
    }


def evaluate_reprojection(camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                           records: List[DetectionRecord]) -> dict:
    if not records:
        return {}
    per_image = []
    all_errors = []
    for rec in records:
        if rec.object_points is None or rec.image_points is None:
            continue
        projected, _ = cv2.projectPoints(rec.object_points, np.zeros((3, 1)), np.zeros((3, 1)),
                                         camera_matrix, dist_coeffs)
        residuals = np.linalg.norm(projected.reshape(-1, 2) - rec.image_points, axis=1)
        all_errors.append(residuals)
        per_image.append({
            "name": rec.name,
            "mean": float(np.mean(residuals)),
            "median": float(np.median(residuals)),
            "max": float(np.max(residuals)),
            "std": float(np.std(residuals)),
        })
    if not all_errors:
        return {}
    stacked = np.concatenate(all_errors)
    return {
        "mean": float(np.mean(stacked)),
        "median": float(np.median(stacked)),
        "max": float(np.max(stacked)),
        "std": float(np.std(stacked)),
        "per_image": per_image,
    }


def refine_record_correspondences(record: DetectionRecord,
                                  reprojection_threshold: float = 2.5,
                                  min_inliers: int = 28) -> None:
    if record.image_points is None or record.object_points is None or record.resolution is None:
        record.inlier_count = 0
        record.original_point_count = 0
        record.inlier_ratio = 0.0
        record.initial_rvec = None
        record.initial_tvec = None
        record.inlier_indices = None
        return

    obj = np.asarray(record.object_points, dtype=np.float64)
    img = np.asarray(record.image_points, dtype=np.float64)
    original_n = int(obj.shape[0])
    record.original_point_count = original_n
    if original_n < 6:
        record.inlier_count = original_n
        record.inlier_ratio = 1.0
        record.initial_rvec = None
        record.initial_tvec = None
        record.inlier_indices = None
        return

    image_size = record.resolution
    try:
        camera_guess = cv2.initCameraMatrix2D([obj], [img], image_size)
    except cv2.error:
        fx = fy = max(image_size)
        cx, cy = image_size[0] * 0.5, image_size[1] * 0.5
        camera_guess = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, camera_guess, None,
            iterationsCount=200,
            reprojectionError=float(reprojection_threshold),
            confidence=0.999,
            flags=cv2.SOLVEPNP_AP3P,
        )
    except cv2.error:
        success, rvec, tvec, inliers = False, None, None, None

    if not success or inliers is None or len(inliers) < min(min_inliers, original_n):
        try:
            success_iter, rvec_iter, tvec_iter = cv2.solvePnP(
                obj, img, camera_guess, None, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            success_iter, rvec_iter, tvec_iter = False, None, None
        if success_iter:
            record.initial_rvec = np.asarray(rvec_iter, dtype=np.float64).reshape(3, 1)
            record.initial_tvec = np.asarray(tvec_iter, dtype=np.float64).reshape(3, 1)
        else:
            record.initial_rvec = None
            record.initial_tvec = None
        record.inlier_count = original_n
        record.inlier_ratio = 1.0
        record.inlier_indices = None
        return

    inliers = inliers.reshape(-1)
    obj_in = obj[inliers]
    img_in = img[inliers]
    try:
        refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
            obj_in, img_in, camera_guess, None,
            np.asarray(rvec, dtype=np.float64),
            np.asarray(tvec, dtype=np.float64),
        )
    except cv2.error:
        refined_rvec = np.asarray(rvec, dtype=np.float64)
        refined_tvec = np.asarray(tvec, dtype=np.float64)

    record.image_points = img_in.astype(np.float32)
    record.object_points = obj_in.astype(np.float32)
    record.initial_rvec = np.asarray(refined_rvec, dtype=np.float64).reshape(3, 1)
    record.initial_tvec = np.asarray(refined_tvec, dtype=np.float64).reshape(3, 1)
    record.inlier_count = int(len(inliers))
    record.inlier_ratio = float(record.inlier_count) / max(1, original_n)
    record.inlier_indices = inliers.astype(np.int32)


def preprocess_detection_records(records: List[DetectionRecord]) -> Dict[str, float]:
    ratios: List[float] = []
    counts: List[int] = []
    for rec in records:
        refine_record_correspondences(rec)
        if rec.original_point_count > 0:
            ratios.append(rec.inlier_ratio)
            counts.append(rec.inlier_count)
    if not ratios:
        return {"avg_ratio": 1.0, "min_ratio": 1.0, "avg_inliers": 0.0}
    return {
        "avg_ratio": float(np.mean(ratios)),
        "min_ratio": float(np.min(ratios)),
        "avg_inliers": float(np.mean(counts)),
    }


def calibrate_camera_robust(records: List[DetectionRecord], image_size: Tuple[int, int],
                            trim_percent: float = 0.15, max_iters: int = 4) -> Optional[dict]:
    valid_records = [rec for rec in records if rec.object_points is not None and rec.image_points is not None]
    if len(valid_records) < 3:
        return None

    obj_points = [np.asarray(rec.object_points, dtype=np.float32) for rec in valid_records]
    img_points = [np.asarray(rec.image_points, dtype=np.float32) for rec in valid_records]

    width, height = image_size

    def clamp_principal_point(K: np.ndarray) -> np.ndarray:
        K_adj = K.copy()
        K_adj[0, 2] = float(np.clip(K_adj[0, 2], 0.05 * width, 0.95 * width))
        K_adj[1, 2] = float(np.clip(K_adj[1, 2], 0.05 * height, 0.95 * height))
        return K_adj

    camera_matrix_init = clamp_principal_point(cv2.initCameraMatrix2D(obj_points, img_points, image_size))
    criteria_stage1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 1e-8)
    criteria_stage2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 180, 1e-10)
    flags_stage2 = (cv2.CALIB_USE_INTRINSIC_GUESS |
                    cv2.CALIB_USE_EXTRINSIC_GUESS |
                    cv2.CALIB_RATIONAL_MODEL |
                    cv2.CALIB_THIN_PRISM_MODEL |
                    cv2.CALIB_TILTED_MODEL)

    def ensure_extrinsic_guesses(obj_list: List[np.ndarray], img_list: List[np.ndarray],
                                 camera_matrix: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        guesses_r: List[np.ndarray] = []
        guesses_t: List[np.ndarray] = []
        for rec, obj, img in zip(valid_records, obj_list, img_list):
            if rec.initial_rvec is not None and rec.initial_tvec is not None:
                guesses_r.append(np.asarray(rec.initial_rvec, dtype=np.float64).reshape(3, 1))
                guesses_t.append(np.asarray(rec.initial_tvec, dtype=np.float64).reshape(3, 1))
                continue
            try:
                success, rvec, tvec = cv2.solvePnP(obj, img, camera_matrix, None, flags=cv2.SOLVEPNP_AP3P)
            except cv2.error:
                success, rvec, tvec = False, None, None
            if not success:
                try:
                    success, rvec, tvec = cv2.solvePnP(obj, img, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
                except cv2.error:
                    success, rvec, tvec = False, None, None
            if success:
                guesses_r.append(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
                guesses_t.append(np.asarray(tvec, dtype=np.float64).reshape(3, 1))
            else:
                guesses_r.append(np.zeros((3, 1), dtype=np.float64))
                guesses_t.append(np.zeros((3, 1), dtype=np.float64))
        return guesses_r, guesses_t

    def run_multistage(obj_list: List[np.ndarray], img_list: List[np.ndarray],
                       camera_matrix_guess: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        rvecs_guess, tvecs_guess = ensure_extrinsic_guesses(obj_list, img_list, camera_matrix_guess)
        flags_stage1 = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_USE_EXTRINSIC_GUESS
        rms_stage1, cam_stage1, dist_stage1, r_stage1, t_stage1 = cv2.calibrateCamera(
            obj_list, img_list, image_size,
            camera_matrix_guess.copy(), None,
            rvecs_guess, tvecs_guess,
            flags=flags_stage1,
            criteria=criteria_stage1)
        cam_stage1 = clamp_principal_point(cam_stage1)
        dist_full_init = np.zeros((14, 1), dtype=np.float64)
        if dist_stage1 is not None:
            d = np.asarray(dist_stage1, dtype=np.float64).reshape(-1)
            dist_full_init[:d.shape[0], 0] = d
        rms_stage2, cam_stage2, dist_stage2, r_stage2, t_stage2 = cv2.calibrateCamera(
            obj_list, img_list, image_size,
            cam_stage1, dist_full_init,
            r_stage1, t_stage1,
            flags=flags_stage2,
            criteria=criteria_stage2)
        cam_stage2 = clamp_principal_point(cam_stage2)
        return float(rms_stage2), cam_stage2, dist_stage2, r_stage2, t_stage2

    current_obj = obj_points
    current_img = img_points
    camera_matrix = camera_matrix_init.copy()
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = run_multistage(current_obj, current_img, camera_matrix)

    trim_ratio = max(0.0, float(trim_percent))
    max_iters = max(0, int(max_iters))
    for _ in range(max_iters):
        if trim_ratio <= 1e-3:
            break
        filtered_obj: List[np.ndarray] = []
        filtered_img: List[np.ndarray] = []
        changed = False
        for obj, img, rvec, tvec in zip(current_obj, current_img, rvecs, tvecs):
            projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
            residuals = np.linalg.norm(projected.reshape(-1, 2) - img, axis=1)
            thresh = np.percentile(residuals, (1.0 - trim_ratio) * 100.0)
            mask = residuals <= thresh
            min_keep = max(20, int(0.6 * len(obj)))
            if np.count_nonzero(mask) >= min_keep and not np.all(mask):
                filtered_obj.append(obj[mask])
                filtered_img.append(img[mask])
                changed = True
            else:
                filtered_obj.append(obj)
                filtered_img.append(img)
        if not changed:
            break
        current_obj = filtered_obj
        current_img = filtered_img
        trim_ratio *= 0.5
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = run_multistage(current_obj, current_img, camera_matrix)

    return {
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "num_samples": len(current_obj),
        "rvecs": rvecs,
        "tvecs": tvecs,
    }


def evaluate_records_against_model(records: List[DetectionRecord], camera_matrix: np.ndarray,
                                   dist_coeffs: np.ndarray) -> Tuple[List[CalibrationRecord], List[Dict[str, float]]]:
    cal_records: List[CalibrationRecord] = []
    stats: List[Dict[str, float]] = []
    for rec in records:
        if rec.object_points is None or rec.image_points is None:
            continue
        cal_rec = solve_pnp_metrics(rec.object_points, rec.image_points, camera_matrix, dist_coeffs)
        if cal_rec is None:
            continue
        cal_rec.name = rec.name
        cal_records.append(cal_rec)
        stats.append({
            "name": rec.name,
            "mean": float(cal_rec.mean_error),
            "median": float(cal_rec.median_error),
            "max": float(cal_rec.max_error),
            "std": float(cal_rec.std_error),
            "num_points": float(cal_rec.num_points),
            "inliers": float(cal_rec.inlier_count),
        })
    return cal_records, stats


def iterative_calibration_with_outlier_filter(
    records: List[DetectionRecord],
    image_size: Tuple[int, int],
    max_mean_error: Optional[float] = None,
    max_point_error: Optional[float] = None,
    max_iterations: int = 3,
    min_samples: int = 10,
) -> Dict[str, object]:
    usable_records = [rec for rec in records if rec.object_points is not None and rec.image_points is not None]
    filtered_records = list(usable_records)
    removed_details: List[Dict[str, object]] = []
    history: List[Dict[str, object]] = []
    best_model: Optional[dict] = None
    final_cal_records: List[CalibrationRecord] = []
    final_stats: List[Dict[str, float]] = []

    if len(filtered_records) < 3:
        logging.warning("可用于鲁棒标定的样本数不足: %d", len(filtered_records))
        return {
            "model": None,
            "records": filtered_records,
            "calibration_records": final_cal_records,
            "summary": {},
            "translation_stats": None,
            "per_image_stats": [],
            "removed": removed_details,
            "history": history,
        }

    max_iterations = max(1, int(max_iterations))
    min_samples = max(3, int(min_samples))
    mean_limit = None if max_mean_error is None or max_mean_error <= 0 else float(max_mean_error)
    point_limit = None if max_point_error is None or max_point_error <= 0 else float(max_point_error)

    for iteration in range(max_iterations):
        calib = calibrate_camera_robust(filtered_records, image_size)
        if calib is None:
            logging.warning("标定失败，终止剔除流程")
            break

        cal_records, stats = evaluate_records_against_model(filtered_records, calib["camera_matrix"], calib["dist_coeffs"])
        if not stats:
            break

        mean_values = np.array([item["mean"] for item in stats], dtype=float)
        median_mean = float(np.median(mean_values))
        mad_mean = float(np.median(np.abs(mean_values - median_mean)))
        mad_mean = mad_mean if mad_mean > 1e-6 else 1e-6
        dynamic_threshold = median_mean + 3.5 * mad_mean
        mean_threshold = min(dynamic_threshold, mean_limit) if mean_limit is not None else dynamic_threshold

        max_values = np.array([item["max"] for item in stats], dtype=float)
        median_max = float(np.median(max_values))
        mad_max = float(np.median(np.abs(max_values - median_max)))
        mad_max = mad_max if mad_max > 1e-6 else 1e-6
        dynamic_point_threshold = median_max + 3.5 * mad_max
        point_threshold = point_limit if point_limit is not None else dynamic_point_threshold

        flagged_indices: List[int] = []
        for idx, item in enumerate(stats):
            if item["mean"] > mean_threshold or item["max"] > point_threshold:
                flagged_indices.append(idx)

        history.append({
            "iteration": iteration + 1,
            "num_candidates": len(filtered_records),
            "num_flagged": len(flagged_indices),
            "median_mean_px": median_mean,
            "mad_mean_px": mad_mean,
            "threshold_mean_px": mean_threshold,
            "median_max_px": median_max,
            "mad_max_px": mad_max,
            "threshold_max_px": point_threshold,
            "mean_of_means_px": float(mean_values.mean()),
            "max_of_means_px": float(mean_values.max()),
            "removed_samples": [filtered_records[i].name for i in flagged_indices],
        })

        best_model = calib
        final_cal_records = cal_records
        final_stats = stats

        if not flagged_indices:
            break

        if len(filtered_records) - len(flagged_indices) < min_samples:
            logging.warning("剔除 %d 个样本后将低于最小保留数量 %d，停止剔除", len(flagged_indices), min_samples)
            break

        remaining: List[DetectionRecord] = []
        flagged_set = set(flagged_indices)
        for idx, rec in enumerate(filtered_records):
            if idx in flagged_set:
                info = stats[idx]
                removed_details.append({
                    "name": rec.name,
                    "mean": float(info["mean"]),
                    "median": float(info["median"]),
                    "max": float(info["max"]),
                    "std": float(info["std"]),
                    "num_points": float(info.get("num_points", 0.0)),
                    "iteration_removed": iteration + 1,
                    "status": "removed",
                })
        for idx, rec in enumerate(filtered_records):
            if idx not in flagged_set:
                remaining.append(rec)

        logging.info("第 %d 轮剔除 %d 个样本，剩余 %d", iteration + 1, len(flagged_indices), len(remaining))
        filtered_records = remaining

    if best_model is None and len(filtered_records) >= 3:
        best_model = calibrate_camera_robust(filtered_records, image_size)
        if best_model is not None:
            final_cal_records, final_stats = evaluate_records_against_model(
                filtered_records, best_model["camera_matrix"], best_model["dist_coeffs"])

    summary: Dict[str, object] = {}
    translation_stats = None
    if final_cal_records:
        summary = summarize_calibration(final_cal_records)
        detection_map = {rec.name: rec for rec in filtered_records}
        camera_matrix = best_model.get("camera_matrix") if best_model else None
        translation_stats = compute_translation_error_stats(final_cal_records, camera_matrix, detection_map)
        if translation_stats:
            summary.update({
                "translation_axis_error_mean_mm": translation_stats["axis_error_mean"].round(3).tolist(),
                "translation_axis_error_std_mm": translation_stats["axis_error_std"].round(3).tolist(),
                "translation_abs_error_mean_mm": translation_stats["abs_error_mean"].round(3).tolist(),
                "translation_abs_error_std_mm": translation_stats["abs_error_std"].round(3).tolist(),
                "translation_rel_error_mean_pct": (translation_stats["rel_error_mean"] * 100).round(3).tolist(),
                "translation_rel_error_std_pct": (translation_stats["rel_error_std"] * 100).round(3).tolist(),
            })

    kept_stats: List[Dict[str, object]] = []
    for item in final_stats:
        kept_stats.append({
            "name": item["name"],
            "mean": float(item["mean"]),
            "median": float(item["median"]),
            "max": float(item["max"]),
            "std": float(item["std"]),
            "num_points": float(item.get("num_points", 0.0)),
            "status": "kept",
        })

    combined_stats = kept_stats + removed_details

    return {
        "model": best_model,
        "records": filtered_records,
        "calibration_records": final_cal_records,
        "summary": summary,
        "translation_stats": translation_stats,
        "per_image_stats": combined_stats,
        "removed": removed_details,
        "history": history,
    }


def perform_kfold_cross_validation(records: List[DetectionRecord], folds: int, seed: int,
                                   image_size: Tuple[int, int]) -> Optional[dict]:
    usable_records = [rec for rec in records if rec.object_points is not None and rec.image_points is not None]
    if len(usable_records) < 3:
        return None
    folds = max(2, min(folds, len(usable_records)))
    indices = list(range(len(usable_records)))
    random.Random(seed).shuffle(indices)
    fold_splits: List[List[int]] = [[] for _ in range(folds)]
    for idx, rec_idx in enumerate(indices):
        fold_splits[idx % folds].append(rec_idx)

    fold_reports = []
    aggregate_errors = []
    for fold_id, val_idx in enumerate(fold_splits):
        train_idx = [i for i in indices if i not in val_idx]
        if len(train_idx) < 3 or not val_idx:
            continue
        train_records = [usable_records[i] for i in train_idx]
        val_records = [usable_records[i] for i in val_idx]

        calib = calibrate_camera_robust(train_records, image_size)
        if calib is None:
            continue

        camera_matrix = calib["camera_matrix"]
        dist_coeffs = calib["dist_coeffs"]
        per_image_stats = []
        all_errors = []
        for rec in val_records:
            success, rvec, tvec = cv2.solvePnP(rec.object_points, rec.image_points,
                                               camera_matrix, dist_coeffs,
                                               flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                logging.warning("[CV] solvePnP 失败: %s", rec.name)
                continue

            projected, _ = cv2.projectPoints(rec.object_points, rvec, tvec, camera_matrix, dist_coeffs)
            residuals = np.linalg.norm(projected.reshape(-1, 2) - rec.image_points, axis=1)
            if residuals.size == 0:
                continue

            R, _ = cv2.Rodrigues(rvec)
            euler = rotation_matrix_to_euler(R)
            translation = tuple(float(x) for x in tvec.reshape(-1))

            all_errors.append(residuals)
            per_image_stats.append({
                "name": rec.name,
                "mean": float(np.mean(residuals)),
                "median": float(np.median(residuals)),
                "max": float(np.max(residuals)),
                "std": float(np.std(residuals)),
                "translation_mm": translation,
                "rotation_deg": euler,
            })

        if not all_errors:
            continue
        stacked = np.concatenate(all_errors)
        aggregate_errors.append(stacked)
        fold_reports.append({
            "fold": fold_id + 1,
            "train_size": len(train_records),
            "val_size": len(val_records),
            "rms": calib["rms"],
            "mean_error": float(np.mean(stacked)),
            "median_error": float(np.median(stacked)),
            "max_error": float(np.max(stacked)),
            "std_error": float(np.std(stacked)),
            "camera_matrix": calib["camera_matrix"].tolist(),
            "dist_coeffs": calib["dist_coeffs"].ravel().tolist(),
            "per_image": per_image_stats,
        })

    if not fold_reports:
        return None

    combined = np.concatenate(aggregate_errors)
    return {
        "folds": fold_reports,
        "overall": {
            "mean_error": float(np.mean(combined)),
            "median_error": float(np.median(combined)),
            "max_error": float(np.max(combined)),
            "std_error": float(np.std(combined)),
        },
    }


def select_samples(calib_records: List[CalibrationRecord], sample_size: int, seed: int) -> List[CalibrationRecord]:
    if not calib_records or sample_size <= 0:
        return []
    sample_size = min(sample_size, len(calib_records))
    random.seed(seed)
    return random.sample(calib_records, sample_size)


def generate_visualizations(output_dir: Path, detection_records: List[DetectionRecord],
                             calib_records: List[CalibrationRecord],
                             translation_stats: Optional[dict],
                             per_image_stats: Optional[List[Dict[str, object]]] = None,
                             filter_history: Optional[List[Dict[str, object]]] = None) -> List[str]:
    if not detection_records:
        return []

    ensure_dir(output_dir)
    plt.style.use("seaborn-v0_8")
    plt.rcParams["font.sans-serif"] = FONT_SANS_SERIF
    plt.rcParams["axes.unicode_minus"] = False

    saved_figures: List[str] = []

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    success = sum(1 for r in detection_records if r.success)
    failure = len(detection_records) - success
    labels = ["成功", "失败"] if HAS_CJK_FONT else ["Success", "Fail"]
    axes[0].bar(labels, [success, failure], color=["#4caf50", "#f44336"], alpha=0.85)
    axes[0].set_title("检测结果分布" if HAS_CJK_FONT else "Detection outcomes")
    axes[0].set_ylabel("数量" if HAS_CJK_FONT else "Count")
    axes[0].grid(axis="y", alpha=0.2)

    if calib_records:
        errors = [rec.mean_error for rec in calib_records]
        axes[1].boxplot(errors, vert=True, patch_artist=True, boxprops=dict(facecolor="#2196f3", alpha=0.7))
        axes[1].scatter(np.ones(len(errors)), errors, color="#0d47a1", alpha=0.5, s=20)
        axes[1].set_title("平均重投影误差分布 (px)" if HAS_CJK_FONT else "Mean reprojection error (px)")
        axes[1].set_xticks([])
        axes[1].set_ylabel("像素" if HAS_CJK_FONT else "Pixels")
        axes[1].grid(axis="y", alpha=0.2)
    else:
        axes[1].text(0.5, 0.5,
                     "暂无重投影数据" if HAS_CJK_FONT else "No reprojection data",
                     ha="center", va="center", fontsize=12)
        axes[1].set_axis_off()

    if calib_records:
        translations = np.array([rec.translation for rec in calib_records])
        sc = axes[2].scatter(translations[:, 0], translations[:, 1], c=translations[:, 2], cmap="viridis", s=80)
        axes[2].set_title("相机位姿散点 (平面 X-Y, 颜色=Z)" if HAS_CJK_FONT else "Camera pose scatter (X-Y plane, color=Z)")
        axes[2].set_xlabel("X (mm)")
        axes[2].set_ylabel("Y (mm)")
        axes[2].grid(alpha=0.2)
        fig.colorbar(sc, ax=axes[2], label="Z (mm)")
    else:
        axes[2].text(0.5, 0.5,
                     "暂无位姿数据" if HAS_CJK_FONT else "No pose data",
                     ha="center", va="center", fontsize=12)
        axes[2].set_axis_off()

    fig.tight_layout()
    fig_path = output_dir / "evaluation_overview.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    saved_figures.append(fig_path.name)

    if calib_records:
        residual_components: List[np.ndarray] = []
        for rec in calib_records:
            if not rec.residual_vectors:
                continue
            arr = np.asarray(rec.residual_vectors, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.size > 0:
                residual_components.append(arr)
        if residual_components:
            residual_stack = np.vstack(residual_components)
            if residual_stack.shape[0] > 12000:
                idx = np.linspace(0, residual_stack.shape[0] - 1, 12000, dtype=int)
                residual_stack = residual_stack[idx]
            magnitudes = np.linalg.norm(residual_stack, axis=1)
            fig_scatter, ax_scatter = plt.subplots(figsize=(6.6, 6.2))
            scatter = ax_scatter.scatter(
                residual_stack[:, 0],
                residual_stack[:, 1],
                c=magnitudes,
                cmap="viridis",
                s=22,
                alpha=0.8,
                linewidths=0,
            )
            ax_scatter.axhline(0.0, color="#b0bec5", linestyle="--", linewidth=1.0)
            ax_scatter.axvline(0.0, color="#b0bec5", linestyle="--", linewidth=1.0)
            ax_scatter.set_aspect("equal", adjustable="box")
            if HAS_CJK_FONT:
                ax_scatter.set_title("重投影误差散点 (Δx, Δy)")
                ax_scatter.set_xlabel("Δx (像素)")
                ax_scatter.set_ylabel("Δy (像素)")
                color_label = "误差幅值 (像素)"
            else:
                ax_scatter.set_title("Reprojection error scatter (Δx, Δy)")
                ax_scatter.set_xlabel("Δx (pixels)")
                ax_scatter.set_ylabel("Δy (pixels)")
                color_label = "Error magnitude (pixels)"
            cbar = fig_scatter.colorbar(scatter, ax=ax_scatter, fraction=0.046, pad=0.04)
            cbar.set_label(color_label)
            fig_scatter.tight_layout()
            scatter_path = output_dir / "reprojection_error_scatter.png"
            fig_scatter.savefig(scatter_path, dpi=220)
            plt.close(fig_scatter)
            saved_figures.append(scatter_path.name)

    if calib_records and translation_stats:
        axes_labels = ["X", "Y", "Z"]
        errors = [rec.mean_error for rec in calib_records]
        abs_mean = translation_stats["abs_error_mean"]
        abs_std = translation_stats["abs_error_std"]
        rel_mean = translation_stats["rel_error_mean"] * 100.0
        rel_std = translation_stats["rel_error_std"] * 100.0

        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        axes2[0].hist(errors, bins=min(15, len(errors)), color="#64b5f6", alpha=0.8, edgecolor="white")
        axes2[0].set_title("重投影误差直方图" if HAS_CJK_FONT else "Reprojection error histogram")
        axes2[0].set_xlabel("像素" if HAS_CJK_FONT else "Pixels")
        axes2[0].set_ylabel("数量" if HAS_CJK_FONT else "Count")
        axes2[0].grid(alpha=0.2)

        axes2[1].bar(axes_labels, abs_mean, yerr=abs_std, color="#81c784", capsize=6, alpha=0.85)
        axes2[1].set_title("轴向绝对误差 (mm)" if HAS_CJK_FONT else "Axis absolute error (mm)")
        axes2[1].set_ylabel("毫米" if HAS_CJK_FONT else "mm")
        axes2[1].grid(axis="y", alpha=0.2)

        axes2[2].bar(axes_labels, rel_mean, yerr=rel_std, color="#ffb74d", capsize=6, alpha=0.85)
        axes2[2].set_title("轴向相对误差 (%)" if HAS_CJK_FONT else "Axis relative error (%)")
        axes2[2].set_ylabel("百分比" if HAS_CJK_FONT else "%")
        axes2[2].grid(axis="y", alpha=0.2)

        fig2.tight_layout()
        fig2_path = output_dir / "pose_error_breakdown.png"
        fig2.savefig(fig2_path, dpi=180)
        plt.close(fig2)
        saved_figures.append(fig2_path.name)

        mean_translation = np.asarray(translation_stats["mean_translation"], dtype=float).reshape(1, 3)
        translations = np.array([rec.translation for rec in calib_records], dtype=float)
        offsets = translations - mean_translation
        fig_offsets, axes_offsets = plt.subplots(1, 3, figsize=(18, 5))
        axis_titles = [
            ("X 轴偏移分布" if HAS_CJK_FONT else "X-axis offset distribution"),
            ("Y 轴偏移分布" if HAS_CJK_FONT else "Y-axis offset distribution"),
            ("Z 轴偏移分布" if HAS_CJK_FONT else "Z-axis offset distribution"),
        ]
        axis_labels = ["Δx (mm)" if HAS_CJK_FONT else "Δx (mm)",
                       "Δy (mm)" if HAS_CJK_FONT else "Δy (mm)",
                       "Δz (mm)" if HAS_CJK_FONT else "Δz (mm)"]
        colors = ["#1976d2", "#388e3c", "#d32f2f"]
        for idx, ax_offset in enumerate(axes_offsets):
            data = offsets[:, idx]
            if data.size == 0:
                continue
            bins = min(30, max(10, data.size // 4))
            ax_offset.hist(data, bins=bins, color=colors[idx], alpha=0.8, edgecolor="white")
            mean_val = float(np.mean(data))
            std_val = float(np.std(data))
            ax_offset.axvline(mean_val, color="#ffa726", linestyle="--", linewidth=1.3,
                               label=("均值" if HAS_CJK_FONT else "Mean") + f" {mean_val:.2f} mm")
            ax_offset.axvline(mean_val + std_val, color="#fb8c00", linestyle=":", linewidth=1.0,
                               label=("±1σ" if HAS_CJK_FONT else "±1σ"))
            ax_offset.axvline(mean_val - std_val, color="#fb8c00", linestyle=":", linewidth=1.0,
                               label="_nolegend_")
            ax_offset.set_title(axis_titles[idx])
            ax_offset.set_xlabel(axis_labels[idx])
            ax_offset.set_ylabel("数量" if HAS_CJK_FONT else "Count")
            ax_offset.grid(alpha=0.25)
            handles, _ = ax_offset.get_legend_handles_labels()
            if handles:
                ax_offset.legend(loc="upper right", fontsize=9)

        fig_offsets.tight_layout()
        offsets_path = output_dir / "translation_axis_offsets.png"
        fig_offsets.savefig(offsets_path, dpi=180)
        plt.close(fig_offsets)
        saved_figures.append(offsets_path.name)

    if calib_records:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore

        object_lookup: Dict[str, np.ndarray] = {
            rec.name: rec.object_points for rec in detection_records
            if rec.object_points is not None
        }
        pose_entries: List[Tuple[CalibrationRecord, np.ndarray]] = []
        board_points: List[np.ndarray] = []
        for rec in calib_records:
            obj = object_lookup.get(rec.name)
            if obj is None or len(obj) == 0:
                continue
            obj = np.asarray(obj, dtype=float).reshape(-1, 3)
            min_x, max_x = float(np.min(obj[:, 0])), float(np.max(obj[:, 0]))
            min_y, max_y = float(np.min(obj[:, 1])), float(np.max(obj[:, 1]))
            board_corners = np.array([
                [min_x, min_y, 0.0],
                [max_x, min_y, 0.0],
                [max_x, max_y, 0.0],
                [min_x, max_y, 0.0],
            ], dtype=float)
            rvec = np.array(rec.rvec, dtype=float).reshape(3, 1)
            translation = np.array(rec.translation, dtype=float).reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            board_in_camera = (R @ board_corners.T + translation).T
            pose_entries.append((rec, board_in_camera))
            board_points.append(board_in_camera)

        if pose_entries:
            fig_pose = plt.figure(figsize=(9.5, 8))
            ax_pose = fig_pose.add_subplot(111, projection="3d")

            axis_len = 60.0
            ax_pose.plot([0, axis_len], [0, 0], [0, 0], color="#d32f2f", linewidth=2.0)
            ax_pose.plot([0, 0], [0, axis_len], [0, 0], color="#388e3c", linewidth=2.0)
            ax_pose.plot([0, 0], [0, 0], [0, axis_len], color="#1976d2", linewidth=2.0)
            ax_pose.text(axis_len, 0, 0, "X", color="#d32f2f", fontsize=10)
            ax_pose.text(0, axis_len, 0, "Y", color="#388e3c", fontsize=10)
            ax_pose.text(0, 0, axis_len, "Z", color="#1976d2", fontsize=10)

            errors = [entry[0].mean_error for entry in pose_entries]
            if errors:
                err_min = float(min(errors))
                err_max = float(max(errors))
                if abs(err_max - err_min) < 1e-6:
                    err_max = err_min + 1e-3
                norm = plt.Normalize(vmin=err_min, vmax=err_max)
            else:
                norm = plt.Normalize(0, 1)
            cmap = plt.get_cmap("viridis")

            for rec, polygon in pose_entries:
                color = cmap(norm(rec.mean_error))
                poly = Poly3DCollection([polygon], alpha=0.55, facecolor=color, edgecolor="#1b5e20", linewidth=0.8)
                ax_pose.add_collection3d(poly)
                center = np.mean(polygon, axis=0)
                ax_pose.scatter(center[0], center[1], center[2], color=color, s=30)
                ax_pose.text(center[0], center[1], center[2], rec.name, fontsize=7, color="#212121")

            all_points = np.vstack(board_points + [np.zeros((1, 3))])
            min_vals = all_points.min(axis=0)
            max_vals = all_points.max(axis=0)
            max_range = (max_vals - min_vals).max()
            mid = (max_vals + min_vals) / 2.0
            half = max_range / 2.0 if max_range > 0 else axis_len
            ax_pose.set_xlim(mid[0] - half, mid[0] + half)
            ax_pose.set_ylim(mid[1] - half, mid[1] + half)
            ax_pose.set_zlim(max(mid[2] - half, -axis_len), mid[2] + half)

            ax_pose.set_xlabel("X (mm)")
            ax_pose.set_ylabel("Y (mm)")
            ax_pose.set_zlabel("Z (mm)")
            ax_pose.view_init(elev=28, azim=-60)
            ax_pose.grid(True, alpha=0.2)

            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            color_label = "平均重投影误差 (px)" if HAS_CJK_FONT else "Mean reprojection error (px)"
            fig_pose.colorbar(mappable, ax=ax_pose, shrink=0.65, pad=0.1, label=color_label)
            title = "固定相机坐标系下的标定板位姿" if HAS_CJK_FONT else "Board poses in camera frame"
            ax_pose.set_title(title)
            fig_pose.tight_layout()
            pose_path = output_dir / "board_camera_pose_map.png"
            fig_pose.savefig(pose_path, dpi=180)
            plt.close(fig_pose)
            saved_figures.append(pose_path.name)

        image_size = None
        for rec in detection_records:
            if rec.resolution is not None:
                image_size = rec.resolution
                break

        point_coords: List[np.ndarray] = []
        point_errors: List[np.ndarray] = []
        for rec in calib_records:
            if not rec.residuals or not rec.image_points:
                continue
            points = np.asarray(rec.image_points, dtype=float)
            errors_arr = np.asarray(rec.residuals, dtype=float)
            if points.shape[0] != errors_arr.shape[0]:
                continue
            point_coords.append(points)
            point_errors.append(errors_arr)

        if image_size and point_coords:
            width, height = image_size
            pts = np.vstack(point_coords)
            errs = np.concatenate(point_errors)
            bins_x = max(12, int(width / 80))
            bins_y = max(12, int(height / 80))
            hist_sum, xedges, yedges = np.histogram2d(
                pts[:, 1], pts[:, 0],
                bins=[bins_y, bins_x],
                range=[[0, height], [0, width]],
                weights=errs,
            )
            hist_count, _, _ = np.histogram2d(
                pts[:, 1], pts[:, 0],
                bins=[bins_y, bins_x],
                range=[[0, height], [0, width]],
                weights=np.ones_like(errs),
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                heatmap_raw = np.divide(hist_sum, hist_count, out=np.zeros_like(hist_sum), where=hist_count > 0)

            # 对统计图进行柔和处理，避免离散块带来的锯齿
            if np.any(hist_count > 0):
                hist_sum_smooth = cv2.GaussianBlur(hist_sum.astype(np.float32), (0, 0), sigmaX=0.9, sigmaY=0.9)
                hist_count_smooth = cv2.GaussianBlur(hist_count.astype(np.float32), (0, 0), sigmaX=0.9, sigmaY=0.9)
                with np.errstate(invalid="ignore", divide="ignore"):
                    heatmap_smooth = np.divide(
                        hist_sum_smooth,
                        hist_count_smooth,
                        out=heatmap_raw.copy(),
                        where=hist_count_smooth > 1e-6,
                    )
            else:
                heatmap_smooth = heatmap_raw

            target_size = (int(width), int(height))
            if target_size[0] > 0 and target_size[1] > 0:
                heatmap_display = cv2.resize(
                    heatmap_smooth.astype(np.float32),
                    target_size,
                    interpolation=cv2.INTER_CUBIC,
                )
                heatmap_display = cv2.GaussianBlur(heatmap_display, (0, 0), sigmaX=1.2, sigmaY=1.2)
            else:
                heatmap_display = heatmap_smooth.astype(np.float32)

            heatmap_masked = np.ma.masked_invalid(heatmap_display)

            fig_heat, ax_heat = plt.subplots(figsize=(8.5, 6.5))
            im = ax_heat.imshow(
                heatmap_masked,
                extent=[0, width, height, 0],
                origin="upper",
                cmap="inferno",
                aspect="auto",
                interpolation="bicubic",
            )
            ax_heat.scatter(pts[:, 0], pts[:, 1], s=6, color="#ffffff", alpha=0.25, linewidths=0)
            ax_heat.set_xlabel("像素 X" if HAS_CJK_FONT else "Pixel X")
            ax_heat.set_ylabel("像素 Y" if HAS_CJK_FONT else "Pixel Y")
            title = "图像区域重投影误差热力图" if HAS_CJK_FONT else "Reprojection error heatmap"
            ax_heat.set_title(title)
            ax_heat.set_xlim(0, width)
            ax_heat.set_ylim(height, 0)
            ax_heat.grid(False)
            cbar = fig_heat.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.set_label("平均误差 (px)" if HAS_CJK_FONT else "Mean error (px)")
            fig_heat.tight_layout()
            heat_path = output_dir / "reprojection_error_heatmap.png"
            fig_heat.savefig(heat_path, dpi=180)
            plt.close(fig_heat)
            saved_figures.append(heat_path.name)

    if per_image_stats:
        sorted_stats = sorted(per_image_stats, key=lambda x: x.get("mean", 0.0), reverse=True)
        if sorted_stats:
            names = [str(item.get("name", "")) for item in sorted_stats]
            means = [float(item.get("mean", 0.0)) for item in sorted_stats]
            statuses = [item.get("status", "kept") for item in sorted_stats]
            y_positions = np.arange(len(sorted_stats))
            fig_height = max(6.0, len(sorted_stats) * 0.35)
            fig3, ax3 = plt.subplots(figsize=(12, fig_height))

            kept_indices = [idx for idx, status in enumerate(statuses) if status != "removed"]
            removed_indices = [idx for idx, status in enumerate(statuses) if status == "removed"]
            if kept_indices:
                ax3.barh(y_positions[kept_indices], [means[i] for i in kept_indices],
                         color="#42a5f5", alpha=0.85,
                         label="保留样本" if HAS_CJK_FONT else "Kept")
            if removed_indices:
                ax3.barh(y_positions[removed_indices], [means[i] for i in removed_indices],
                         color="#ef5350", alpha=0.85,
                         label="剔除样本" if HAS_CJK_FONT else "Removed")

            threshold_val = None
            if filter_history:
                threshold_val = filter_history[-1].get("threshold_mean_px")
                if threshold_val is not None:
                    ax3.axvline(float(threshold_val), color="#ff7043", linestyle="--",
                                linewidth=1.2,
                                label=("阈值" if HAS_CJK_FONT else "Threshold") +
                                      f" {float(threshold_val):.2f}px")

            ax3.set_yticks(y_positions)
            ax3.set_yticklabels(names)
            ax3.invert_yaxis()
            ax3.set_xlabel("平均重投影误差 (px)" if HAS_CJK_FONT else "Mean reprojection error (px)")
            ax3.set_title("样本重投影误差分布" if HAS_CJK_FONT else "Per-sample reprojection error")
            ax3.grid(axis="x", alpha=0.25)

            handles, labels = ax3.get_legend_handles_labels()
            if handles:
                ax3.legend(handles, labels, loc="lower right")

            fig3.tight_layout()
            fig3_path = output_dir / "per_image_reprojection.png"
            fig3.savefig(fig3_path, dpi=180)
            plt.close(fig3)
            saved_figures.append(fig3_path.name)

    if filter_history:
        iterations = [int(entry.get("iteration", idx + 1)) for idx, entry in enumerate(filter_history)]
        mean_means = [float(entry.get("mean_of_means_px", 0.0)) for entry in filter_history]
        max_means = [float(entry.get("max_of_means_px", 0.0)) for entry in filter_history]
        threshold_vals = [entry.get("threshold_mean_px") for entry in filter_history]
        has_threshold = any(val is not None for val in threshold_vals)

        fig4, ax4 = plt.subplots(figsize=(8, max(4.5, 3.5 + 0.7 * len(filter_history))))
        ax4.plot(iterations, mean_means, marker="o", color="#42a5f5",
                 label="平均重投影 (均值)" if HAS_CJK_FONT else "Mean of means")
        ax4.plot(iterations, max_means, marker="^", color="#8d6e63",
                 label="平均重投影 (最大)" if HAS_CJK_FONT else "Max of means")
        if has_threshold:
            thr_iters = [it for it, val in zip(iterations, threshold_vals) if val is not None]
            thr_vals = [float(val) for val in threshold_vals if val is not None]
            ax4.plot(thr_iters, thr_vals, linestyle="--", color="#ff7043",
                     label="阈值" if HAS_CJK_FONT else "Threshold")
        ax4.set_xticks(iterations)
        ax4.set_xlabel("迭代轮次" if HAS_CJK_FONT else "Iteration")
        ax4.set_ylabel("像素" if HAS_CJK_FONT else "Pixels")
        ax4.set_title("异常样本剔除过程" if HAS_CJK_FONT else "Outlier filtering progress")
        ax4.grid(alpha=0.3)
        ax4.legend(loc="best")

        fig4.tight_layout()
        fig4_path = output_dir / "outlier_filter_progress.png"
        fig4.savefig(fig4_path, dpi=180)
        plt.close(fig4)
        saved_figures.append(fig4_path.name)

    return saved_figures


def write_markdown_report(path: Path, detection_summary: dict, detection_records: List[DetectionRecord],
                          detection_config_info: Optional[dict], board_spec: BoardSpec,
                          calib_summary: dict, calib_records: List[CalibrationRecord],
                          samples: List[CalibrationRecord], translation_stats: Optional[dict],
                          cv_summary: Optional[dict], cv_records: List[CalibrationRecord],
                          cv_translation_stats: Optional[dict], cv_results: Optional[dict],
                          filter_report: Optional[Dict[str, object]],
                          per_image_stats: Optional[List[Dict[str, object]]],
                          figure_files: Optional[List[str]]) -> None:
    lines: List[str] = []
    lines.append("# 检测与标定评估报告")
    lines.append("")
    lines.append("## 检测表现")
    lines.append(f"- 总图片数: {detection_summary['total_images']}")
    lines.append(f"- 成功: {detection_summary['success']} | 失败: {detection_summary['failure']}")
    lines.append(f"- 成功率: {detection_summary['success_rate']:.1f}%")
    lines.append(f"- 耗时统计 (ms): 平均 {detection_summary['time_avg_ms']:.2f}, 最快 {detection_summary['time_min_ms']:.2f}, 最慢 {detection_summary['time_max_ms']:.2f}")

    lines.append("")
    lines.append("## 标定板参数")
    lines.append(f"- 小圆直径: {board_spec.small_diameter_mm:.2f} mm")
    lines.append(f"- 小圆半径: {board_spec.small_radius_mm:.2f} mm")
    lines.append(f"- 圆心间距: {board_spec.center_spacing_mm:.2f} mm")

    if detection_config_info:
        overrides = detection_config_info.get("overrides") or {}
        lines.append("")
        lines.append("### 检测参数配置")
        lines.append(f"- 预设: `{detection_config_info.get('preset', 'default')}`")
        if overrides:
            lines.append("- 覆盖参数:")
            for key in sorted(overrides):
                value = overrides[key]
                lines.append(f"  - `{key}` = `{value}`")
        else:
            lines.append("- 覆盖参数: 无")

    failures = [r for r in detection_records if not r.success]
    if failures:
        lines.append("")
        lines.append("### 失败样例摘要")
        for rec in failures[:10]:
            lines.append(f"- {rec.name}: {rec.reason} (耗时 {rec.elapsed_ms:.2f} ms)")
        if len(failures) > 10:
            lines.append(f"- ... 其余 {len(failures) - 10} 个略")

    if figure_files:
        lines.append("")
        lines.append("## 可视化概览")
        figure_titles = {
            "evaluation_overview.png": "检测概览" if HAS_CJK_FONT else "Detection overview",
            "pose_error_breakdown.png": "姿态误差拆解" if HAS_CJK_FONT else "Pose error breakdown",
            "reprojection_error_scatter.png": "重投影误差散点 (Δx, Δy)" if HAS_CJK_FONT else "Reprojection error scatter (Δx, Δy)",
            "board_camera_pose_map.png": "固定相机坐标系下的标定板位姿" if HAS_CJK_FONT else "Board poses in camera frame",
            "reprojection_error_heatmap.png": "图像区域重投影误差热力图" if HAS_CJK_FONT else "Reprojection error heatmap",
            "per_image_reprojection.png": "单样本重投影误差" if HAS_CJK_FONT else "Per-image reprojection error",
            "outlier_filter_progress.png": "迭代剔除进程" if HAS_CJK_FONT else "Outlier filtering progress",
            "translation_axis_offsets.png": "XYZ 位移估计分布" if HAS_CJK_FONT else "Translation axis offsets",
        }
        for name in figure_files:
            title = figure_titles.get(name, name)
            lines.append(f"![{title}](figures/{name})")

    if filter_report:
        lines.append("")
        lines.append("## 异常样本筛选")
        initial = filter_report.get("initial_samples")
        final = filter_report.get("final_samples")
        removed_list = filter_report.get("removed_samples", []) or []
        history = filter_report.get("history", []) or []
        lines.append(f"- 初始成功样本数: {initial}")
        lines.append(f"- 剩余样本数: {final}")
        lines.append(f"- 剔除数量: {len(removed_list)}")
        if history:
            last_hist = history[-1]
            mean_thr = last_hist.get("threshold_mean_px")
            max_thr = last_hist.get("threshold_max_px")
            lines.append(f"- 迭代轮数: {len(history)}")
            if mean_thr is not None:
                lines.append(f"- 最终平均误差阈值: {float(mean_thr):.2f} px")
            if max_thr is not None:
                lines.append(f"- 最终单点误差阈值: {float(max_thr):.2f} px")
        if removed_list:
            lines.append("")
            lines.append("### 剔除样本详情")
            lines.append("| 图像 | 平均 (px) | 最大 (px) | 中位数 (px) | 标准差 (px) | 点数 | 剔除轮次 |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for item in removed_list:
                lines.append(
                    f"| {item.get('name','')} | {float(item.get('mean', 0.0)):.3f} | {float(item.get('max', 0.0)):.3f} | "
                    f"{float(item.get('median', 0.0)):.3f} | {float(item.get('std', 0.0)):.3f} | "
                    f"{int(item.get('num_points', 0))} | {int(item.get('iteration_removed', 0))} |"
                )

    if per_image_stats:
        kept = [item for item in per_image_stats if item.get("status") == "kept"]
        removed = [item for item in per_image_stats if item.get("status") == "removed"]
        lines.append("")
        lines.append("## 样本重投影误差分布")
        if kept:
            top_kept = sorted(kept, key=lambda x: x.get("mean", 0.0), reverse=True)[:5]
            lines.append("### 剩余样本中重投影最高 TOP5")
            lines.append("| 图像 | 平均 (px) | 最大 (px) | 中位数 (px) | 标准差 (px) |")
            lines.append("| --- | --- | --- | --- | --- |")
            for item in top_kept:
                lines.append(
                    f"| {item.get('name','')} | {float(item.get('mean',0.0)):.3f} | {float(item.get('max',0.0)):.3f} | "
                    f"{float(item.get('median',0.0)):.3f} | {float(item.get('std',0.0)):.3f} |"
                )
        if removed and not filter_report:
            # 如果前面尚未展示剔除结果，这里给出简略表
            lines.append("### 被剔除样本")
            lines.append("| 图像 | 平均 (px) | 最大 (px) | 剔除轮次 |")
            lines.append("| --- | --- | --- | --- |")
            for item in removed:
                lines.append(
                    f"| {item.get('name','')} | {float(item.get('mean',0.0)):.3f} | {float(item.get('max',0.0)):.3f} | "
                    f"{int(item.get('iteration_removed',0))} |"
                )

    if calib_summary:
        lines.append("")
        lines.append("## 标定精度分析")
        lines.append(f"- 有效样本数: {calib_summary['num_samples']}")
        lines.append(f"- 平均重投影误差: {calib_summary['mean_reprojection_px']:.3f} px")
        lines.append(f"- 中位数重投影误差: {calib_summary['median_reprojection_px']:.3f} px")
        lines.append(f"- 最坏重投影误差: {calib_summary['max_reprojection_px']:.3f} px")
        lines.append(f"- 重投影标准差 (均值): {calib_summary['std_reprojection_px']:.3f} px")
        t_mean = calib_summary['translation_mean_mm']
        t_std = calib_summary['translation_std_mm']
        lines.append(f"- 平均相对位姿 (mm): X={t_mean[0]:.1f}±{t_std[0]:.1f}, Y={t_mean[1]:.1f}±{t_std[1]:.1f}, Z={t_mean[2]:.1f}±{t_std[2]:.1f}")
        r_mean = calib_summary['rotation_mean_deg']
        r_std = calib_summary['rotation_std_deg']
        lines.append(f"- 平均姿态 (deg): roll={r_mean[0]:.1f}±{r_std[0]:.1f}, pitch={r_mean[1]:.1f}±{r_std[1]:.1f}, yaw={r_mean[2]:.1f}±{r_std[2]:.1f}")
        if translation_stats:
            axis_mean = translation_stats["axis_error_mean"]
            axis_std = translation_stats["axis_error_std"]
            abs_mean = translation_stats["abs_error_mean"]
            abs_std = translation_stats["abs_error_std"]
            rel_mean = translation_stats["rel_error_mean"] * 100.0
            rel_std = translation_stats["rel_error_std"] * 100.0
            lines.append(f"- 轴向位移估计 (mm): ΔX={axis_mean[0]:.3f}±{axis_std[0]:.3f}, ΔY={axis_mean[1]:.3f}±{axis_std[1]:.3f}, ΔZ={axis_mean[2]:.3f}±{axis_std[2]:.3f}")
            lines.append(f"- 轴向绝对误差均值 (mm): X={abs_mean[0]:.3f}±{abs_std[0]:.3f}, Y={abs_mean[1]:.3f}±{abs_std[1]:.3f}, Z={abs_mean[2]:.3f}±{abs_std[2]:.3f}")
            lines.append(f"- 轴向相对误差均值 (%): X={rel_mean[0]:.3f}±{rel_std[0]:.3f}, Y={rel_mean[1]:.3f}±{rel_std[1]:.3f}, Z={rel_mean[2]:.3f}±{rel_std[2]:.3f}")

    if samples:
        lines.append("")
        lines.append("## 抽样详情")
        lines.append("| 图像 | 重投影均值 (px) | 重投影最大 (px) | 位姿 (mm) | 姿态 (deg) | 轴向位移估计 (mm) | 绝对误差 (mm) | 相对误差 (%) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for rec in samples:
            t = rec.translation
            r = rec.rotation_deg
            axis_err = rec.axis_error_mm or (0.0, 0.0, 0.0)
            abs_err = rec.abs_error_mm or (0.0, 0.0, 0.0)
            rel_err = rec.rel_error or (0.0, 0.0, 0.0)
            lines.append(
                f"| {rec.name} | {rec.mean_error:.3f} | {rec.max_error:.3f} | "
                f"({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) | ({r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}) | "
                f"({axis_err[0]:.3f}, {axis_err[1]:.3f}, {axis_err[2]:.3f}) | "
                f"({abs_err[0]:.3f}, {abs_err[1]:.3f}, {abs_err[2]:.3f}) | "
                f"({rel_err[0]*100:.3f}, {rel_err[1]*100:.3f}, {rel_err[2]*100:.3f}) |"
            )

    if cv_results or cv_summary or cv_records:
        lines.append("")
        lines.append("## 交叉验证与鲁棒标定")
        if cv_results:
            overall = cv_results.get("overall", {}) if cv_results else {}
            lines.append(f"- 折数: {len(cv_results.get('folds', []))}")
            if overall:
                lines.append(f"- 交叉验证平均重投影误差: {overall.get('mean_error', 0.0):.3f} px")
                lines.append(f"- 交叉验证最大误差: {overall.get('max_error', 0.0):.3f} px")
            folds = cv_results.get("folds", [])
            if folds:
                lines.append("")
                lines.append("| Fold | 训练样本 | 验证样本 | 均值 (px) | 中位数 (px) | 最大 (px) |")
                lines.append("| --- | --- | --- | --- | --- | --- |")
                for fold in folds:
                    lines.append(
                        f"| {fold['fold']} | {fold['train_size']} | {fold['val_size']} | "
                        f"{fold['mean_error']:.3f} | {fold['median_error']:.3f} | {fold['max_error']:.3f} |"
                    )

        if cv_summary:
            lines.append("")
            lines.append("### 全量鲁棒标定统计")
            lines.append(f"- 样本数: {cv_summary.get('num_samples', 0)}")
            if "mean_reprojection_px" in cv_summary:
                lines.append(f"- 平均重投影误差: {cv_summary['mean_reprojection_px']:.3f} px")
                lines.append(f"- 中位数重投影误差: {cv_summary['median_reprojection_px']:.3f} px")
                lines.append(f"- 最坏重投影误差: {cv_summary['max_reprojection_px']:.3f} px")
            if "translation_mean_mm" in cv_summary:
                t_mean = cv_summary['translation_mean_mm']
                t_std = cv_summary['translation_std_mm']
                lines.append(f"- 平均位姿 (mm): X={t_mean[0]:.1f}±{t_std[0]:.1f}, Y={t_mean[1]:.1f}±{t_std[1]:.1f}, Z={t_mean[2]:.1f}±{t_std[2]:.1f}")
            if "rotation_mean_deg" in cv_summary:
                r_mean = cv_summary['rotation_mean_deg']
                r_std = cv_summary['rotation_std_deg']
                lines.append(f"- 平均姿态 (deg): roll={r_mean[0]:.1f}±{r_std[0]:.1f}, pitch={r_mean[1]:.1f}±{r_std[1]:.1f}, yaw={r_mean[2]:.1f}±{r_std[2]:.1f}")
            if cv_translation_stats:
                axis_mean = cv_translation_stats['axis_error_mean']
                axis_std = cv_translation_stats['axis_error_std']
                abs_mean = cv_translation_stats['abs_error_mean']
                abs_std = cv_translation_stats['abs_error_std']
                rel_mean = cv_translation_stats['rel_error_mean'] * 100.0
                rel_std = cv_translation_stats['rel_error_std'] * 100.0
                lines.append(f"- 轴向位移估计 (mm): ΔX={axis_mean[0]:.3f}±{axis_std[0]:.3f}, ΔY={axis_mean[1]:.3f}±{axis_std[1]:.3f}, ΔZ={axis_mean[2]:.3f}±{axis_std[2]:.3f}")
                lines.append(f"- 轴向绝对误差均值 (mm): X={abs_mean[0]:.3f}±{abs_std[0]:.3f}, Y={abs_mean[1]:.3f}±{abs_std[1]:.3f}, Z={abs_mean[2]:.3f}±{abs_std[2]:.3f}")
                lines.append(f"- 轴向相对误差均值 (%): X={rel_mean[0]:.3f}±{rel_std[0]:.3f}, Y={rel_mean[1]:.3f}±{rel_std[1]:.3f}, Z={rel_mean[2]:.3f}±{rel_std[2]:.3f}")

        if cv_records:
            lines.append("")
            lines.append("### 鲁棒标定抽样")
            lines.append("| 图像 | 重投影均值 (px) | 重投影最大 (px) | 位姿 (mm) | 姿态 (deg) | 轴向位移估计 (mm) | 绝对误差 (mm) | 相对误差 (%) |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for rec in cv_records[:10]:
                t = rec.translation
                r = rec.rotation_deg
                axis_err = rec.axis_error_mm or (0.0, 0.0, 0.0)
                abs_err = rec.abs_error_mm or (0.0, 0.0, 0.0)
                rel_err = rec.rel_error or (0.0, 0.0, 0.0)
                lines.append(
                    f"| {rec.name} | {rec.mean_error:.3f} | {rec.max_error:.3f} | "
                    f"({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) | ({r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}) | "
                    f"({axis_err[0]:.3f}, {axis_err[1]:.3f}, {axis_err[2]:.3f}) | "
                    f"({abs_err[0]:.3f}, {abs_err[1]:.3f}, {abs_err[2]:.3f}) | "
                    f"({rel_err[0]*100:.3f}, {rel_err[1]*100:.3f}, {rel_err[2]*100:.3f}) |"
                )
            if len(cv_records) > 10:
                lines.append(f"| ... | | | | | | | 剩余 {len(cv_records) - 10} 个样本略 |")

    report = "\n".join(lines)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="板检测与标定质量评估")
    parser.add_argument("--images", default="data/raw/calibration/calib_25", help="输入图像目录")
    parser.add_argument("--output", default="outputs/reports/board_eval", help="评估输出目录")
    parser.add_argument("--calibration", default="outputs/calibration/latest/camera_calibration_improved.json",
                        help="标定结果 JSON 文件路径")
    parser.add_argument("--export-calibration", default=None,
                        help="鲁棒标定结果 JSON 输出路径，默认写入 <output>/calibration/camera_calibration_robust.json")
    parser.add_argument("--small-diameter", type=float, default=DEFAULT_BOARD_SPEC.small_diameter_mm,
                        help="标定板小圆直径 (mm)")
    parser.add_argument("--circle-spacing", type=float, default=DEFAULT_BOARD_SPEC.center_spacing_mm,
                        help="标定板小圆圆心间距 (mm)")
    parser.add_argument("--sample-size", type=int, default=5, help="抽样展示数量")
    parser.add_argument("--random-seed", type=int, default=2025, help="抽样随机种子")
    parser.add_argument("--folds", type=int, default=5, help="交叉验证的折数")
    parser.add_argument("--cv-seed", type=int, default=2025, help="交叉验证随机种子")
    parser.add_argument("--exts", default="*.png,*.jpg,*.jpeg,*.bmp,*.tif,*.tiff,*.dng,*.DNG",
                        help="匹配的图像扩展名列表")
    parser.add_argument("--save-viz", action="store_true", help="保存可视化调试图像")
    parser.add_argument("--skip-calibration", action="store_true", help="仅评估检测，不执行交叉验证标定")
    parser.add_argument("--max-mean-error", type=float, default=8.0,
                        help="允许的平均重投影误差上限 (px)，<=0 时禁用")
    parser.add_argument("--max-point-error", type=float, default=45.0,
                        help="允许的单点最大误差上限 (px)，<=0 时禁用")
    parser.add_argument("--max-filter-iterations", type=int, default=3,
                        help="自动剔除异常样本的迭代次数")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="剔除后保留的最少样本数")
    parser.add_argument("--det-preset", choices=["default", "high_recall"], default="default",
                        help="检测参数预设，可选 default/high_recall")
    parser.add_argument("--det-override", action="append", default=[], metavar="KEY=VALUE",
                        help="覆写检测配置字段，可多次使用，例如 blob_min_area=320.0")
    args = parser.parse_args()

    image_dir = Path(args.images)
    output_root = Path(args.output)
    ensure_dir(output_root)
    viz_dir = output_root / "visualizations"
    ensure_dir(viz_dir)

    patterns = [e.strip() for e in args.exts.split(",") if e.strip()]
    img_paths = collect_image_paths(image_dir, patterns)
    if not img_paths:
        logging.error("未找到输入图像，请检查 --images 与 --exts")
        return

    board_spec = BoardSpec(small_diameter_mm=float(args.small_diameter),
                           center_spacing_mm=float(args.circle_spacing))

    logging.info("标定板参数：小圆直径=%.2f mm，圆心间距=%.2f mm",
                 board_spec.small_diameter_mm, board_spec.center_spacing_mm)

    det_config, det_override_map = build_detection_config(args.det_preset, args.det_override)
    detection_config_info = {
        "preset": args.det_preset,
        "overrides": det_override_map,
        "resolved": asdict(det_config),
    }
    logging.info("检测参数 preset=%s overrides=%s", args.det_preset,
                 det_override_map if det_override_map else "<无>")

    calibrator = Calibrator(config=det_config)
    visualizer = Visualizer(
        str(viz_dir),
        calibration_path=args.calibration,
        board_spacing=board_spec.center_spacing_mm,
    ) if args.save_viz else None

    expected_small = 41
    expected_big = 4
    detection_records: List[DetectionRecord] = []

    progress = ProgressPrinter(len(img_paths))
    for idx, img_path in enumerate(img_paths, 1):
        base = img_path.stem
        gray = read_image_robust(img_path)
        if gray is None:
            detection_records.append(DetectionRecord(
                name=base,
                success=False,
                reason="读图失败",
                elapsed_ms=0.0,
                small_count=0,
                big_count=0,
                image_points=None,
                object_points=None,
                resolution=None,
            ))
            logging.warning("[FAIL] %s 读图失败", base)
            progress.update(idx)
            continue

        t0 = time.perf_counter()
        debug_info: Dict[str, Any] = {}
        result = calibrator.process_gray(gray, debug=debug_info)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        small_cnt = len(result.small_numbered) if (result and result.small_numbered) else 0
        big_cnt = len(result.big_circles) if result else 0
        success = bool(result) and small_cnt == expected_small and big_cnt == expected_big

        reason = "成功" if success else "检测/编号失败"
        if result and result.small_numbered and success:
            reason = "成功"
        elif result and result.small_numbered:
            reason = (f"圆数量不匹配 (小圆={small_cnt}/{expected_small}, "
                      f"大圆={big_cnt}/{expected_big})")
        elif not success:
            fail_reason = debug_info.get("fail_reason")
            if fail_reason == "quad_not_found":
                reason = "未找到有效四边形"
            elif fail_reason == "no_blobs":
                reason = "未检测到圆形候选"

        image_points = extract_image_points_ordered(result) if success else None
        object_points = (create_circle_board_object_points(expected_small, board_spec=board_spec)
                         if success else None)

        detection_records.append(DetectionRecord(
            name=base,
            success=success,
            reason=reason,
            elapsed_ms=elapsed_ms,
            small_count=small_cnt,
            big_count=big_cnt,
            image_points=image_points,
            object_points=object_points,
            resolution=(gray.shape[1], gray.shape[0]),
        ))

        if success:
            logging.info("[OK] %s 有效耗时 = %.2f ms (小圆=%d，大圆=%d)",
                         base, elapsed_ms, small_cnt, big_cnt)
        else:
            logging.warning("[FAIL] %s %s (耗时 %.2f ms)", base, reason, elapsed_ms)

        if args.save_viz:
            if visualizer and result is not None:
                try:
                    visualizer.save_all(gray, base, result)
                except Exception as exc:  # pylint: disable=broad-except
                    logging.warning("[WARN] 可视化保存异常 %s: %s", base, exc)
            save_debug_artifacts(viz_dir, base, gray, debug_info, result,
                                 expected_small, expected_big, success)

        progress.update(idx)

    detection_summary = summarize_detection(detection_records)
    logging.info("=== 检测汇总 ===")
    logging.info("总图片：%d | 成功：%d | 失败：%d | 成功率：%.1f%%",
                 detection_summary["total_images"], detection_summary["success"],
                 detection_summary["failure"], detection_summary["success_rate"])
    logging.info("耗时（ms） avg=%.2f, min=%.2f, max=%.2f",
                 detection_summary["time_avg_ms"], detection_summary["time_min_ms"], detection_summary["time_max_ms"])

    successful_records = [rec for rec in detection_records if rec.success]

    if successful_records:
        preprocess_stats = preprocess_detection_records(successful_records)
        logging.info("RANSAC 过滤：平均保留 %.1f/%d 点 (最小比例 %.2f)",
                     preprocess_stats.get("avg_inliers", 0.0),
                     expected_small,
                     preprocess_stats.get("min_ratio", 1.0))

    calibration_records: List[CalibrationRecord] = []
    calib_summary: dict = {}
    external_translation_stats: Optional[dict] = None
    cv_results: Optional[dict] = None
    cv_model: Optional[dict] = None
    cv_summary: dict = {}
    cv_translation_stats: Optional[dict] = None
    cv_calibration_records: List[CalibrationRecord] = []
    filtered_records: List[DetectionRecord] = successful_records
    filter_report: Dict[str, object] = {}
    per_image_stats: List[Dict[str, object]] = []
    filter_history: List[Dict[str, object]] = []
    image_size: Optional[Tuple[int, int]] = None
    axis_error_logs: Dict[str, dict] = {}

    calib_path = Path(args.calibration)
    if args.export_calibration:
        export_calib_path: Optional[Path] = Path(args.export_calibration)
    else:
        export_calib_path = output_root / "calibration" / "camera_calibration_robust.json"
    if calib_path.exists():
        calib_data = json.loads(calib_path.read_text(encoding="utf-8"))
        camera_matrix = np.array(calib_data.get("camera_matrix")) if calib_data.get("camera_matrix") is not None else None
        dist_coeffs = np.array(calib_data.get("distortion_coefficients")) if calib_data.get("distortion_coefficients") is not None else None
        if camera_matrix is None or dist_coeffs is None:
            logging.warning("标定文件缺少 camera_matrix 或 distortion_coefficients，跳过标定精度评估")
        else:
            for rec in detection_records:
                if not rec.success or rec.object_points is None or rec.image_points is None:
                    continue
                cal_rec = solve_pnp_metrics(rec.object_points, rec.image_points, camera_matrix, dist_coeffs)
                if cal_rec is None:
                    logging.warning("solvePnP 失败: %s", rec.name)
                    continue
                cal_rec.name = rec.name
                calibration_records.append(cal_rec)

            if calibration_records:
                calib_summary = summarize_calibration(calibration_records)
                det_lookup = {rec.name: rec for rec in detection_records}
                external_translation_stats = compute_translation_error_stats(calibration_records, camera_matrix, det_lookup)
                if external_translation_stats:
                    calib_summary.update({
                        "translation_axis_error_mean_mm": external_translation_stats["axis_error_mean"].round(3).tolist(),
                        "translation_axis_error_std_mm": external_translation_stats["axis_error_std"].round(3).tolist(),
                        "translation_abs_error_mean_mm": external_translation_stats["abs_error_mean"].round(3).tolist(),
                        "translation_abs_error_std_mm": external_translation_stats["abs_error_std"].round(3).tolist(),
                        "translation_rel_error_mean_pct": (external_translation_stats["rel_error_mean"] * 100).round(3).tolist(),
                        "translation_rel_error_std_pct": (external_translation_stats["rel_error_std"] * 100).round(3).tolist(),
                    })
                    axis_error_logs["provided_calibration"] = format_translation_stats_for_log(external_translation_stats)

                logging.info("=== 标定精度 ===")
                logging.info("样本数：%d", calib_summary["num_samples"])
                logging.info("平均重投影误差：%.3f px", calib_summary["mean_reprojection_px"])
                logging.info("中位数重投影误差：%.3f px", calib_summary["median_reprojection_px"])
                logging.info("最大重投影误差：%.3f px", calib_summary["max_reprojection_px"])
                if external_translation_stats:
                    axis_mean = external_translation_stats["axis_error_mean"]
                    axis_std = external_translation_stats["axis_error_std"]
                    abs_mean = external_translation_stats["abs_error_mean"]
                    rel_mean = external_translation_stats["rel_error_mean"] * 100.0
                    logging.info("轴向位移估计 (mm)：ΔX=%.3f±%.3f, ΔY=%.3f±%.3f, ΔZ=%.3f±%.3f",
                                 axis_mean[0], axis_std[0], axis_mean[1], axis_std[1], axis_mean[2], axis_std[2])
                    logging.info("轴向绝对误差均值 (mm)：X=%.3f, Y=%.3f, Z=%.3f",
                                 abs_mean[0], abs_mean[1], abs_mean[2])
                    logging.info("轴向相对误差均值 (%%)：X=%.3f, Y=%.3f, Z=%.3f",
                                 rel_mean[0], rel_mean[1], rel_mean[2])
            else:
                logging.warning("没有可用于标定分析的成功样本")
    else:
        logging.warning("未找到标定结果文件：%s", calib_path)

    if not args.skip_calibration and successful_records:
        first_resolution = next((rec.resolution for rec in successful_records if rec.resolution), None)
        if first_resolution:
            image_size = (int(first_resolution[0]), int(first_resolution[1]))
            filter_result = iterative_calibration_with_outlier_filter(
                successful_records,
                image_size,
                max_mean_error=args.max_mean_error,
                max_point_error=args.max_point_error,
                max_iterations=args.max_filter_iterations,
                min_samples=args.min_samples,
            )

            filtered_records = filter_result.get("records", successful_records)
            per_image_stats = filter_result.get("per_image_stats", [])
            filter_history = filter_result.get("history", [])
            removed_entries = filter_result.get("removed", [])

            filter_report = {
                "initial_samples": len(successful_records),
                "final_samples": len(filtered_records),
                "removed_samples": removed_entries,
                "history": filter_history,
            }
            if filter_history:
                filter_report["final_threshold_mean_px"] = filter_history[-1].get("threshold_mean_px")
                filter_report["final_threshold_max_px"] = filter_history[-1].get("threshold_max_px")

            if removed_entries:
                removed_names = ", ".join(entry.get("name", "") for entry in removed_entries if entry.get("name"))
                logging.info("自动剔除异常样本：%s", removed_names)
            logging.info("鲁棒剔除后样本数：%d -> %d", len(successful_records), len(filtered_records))

            cv_model = filter_result.get("model")
            cv_calibration_records = filter_result.get("calibration_records", []) or []
            cv_summary = filter_result.get("summary", {}) or {}
            cv_translation_stats = filter_result.get("translation_stats")
            if cv_translation_stats:
                axis_error_logs["robust_calibration"] = format_translation_stats_for_log(cv_translation_stats)

            kept_stats = [item for item in per_image_stats if item.get("status") == "kept"]
            if kept_stats:
                worst_kept = max(kept_stats, key=lambda x: x["mean"])
                logging.info("剔除后最大平均重投影误差：%.3f px (%s)", worst_kept["mean"], worst_kept["name"])

            if cv_model:
                logging.info("=== 全量鲁棒标定（剔除后） ===")
                logging.info("样本数：%d | RMS：%.3f", cv_model["num_samples"], cv_model["rms"])
            if cv_summary:
                logging.info("平均重投影误差：%.3f px | 中位数：%.3f px | 最大：%.3f px",
                             cv_summary.get("mean_reprojection_px", 0.0),
                             cv_summary.get("median_reprojection_px", 0.0),
                             cv_summary.get("max_reprojection_px", 0.0))
            if cv_translation_stats is not None:
                axis_mean_cv = cv_translation_stats["axis_error_mean"]
                axis_std_cv = cv_translation_stats["axis_error_std"]
                abs_mean_cv = cv_translation_stats["abs_error_mean"]
                rel_mean_cv = cv_translation_stats["rel_error_mean"] * 100.0
                logging.info("轴向位移估计 (mm)：ΔX=%.3f±%.3f, ΔY=%.3f±%.3f, ΔZ=%.3f±%.3f",
                             axis_mean_cv[0], axis_std_cv[0], axis_mean_cv[1], axis_std_cv[1],
                             axis_mean_cv[2], axis_std_cv[2])
                logging.info("轴向绝对误差均值 (mm)：X=%.3f, Y=%.3f, Z=%.3f",
                             abs_mean_cv[0], abs_mean_cv[1], abs_mean_cv[2])
                logging.info("轴向相对误差均值 (%%)：X=%.3f, Y=%.3f, Z=%.3f",
                             rel_mean_cv[0], rel_mean_cv[1], rel_mean_cv[2])

            if filtered_records:
                cv_results = perform_kfold_cross_validation(filtered_records, args.folds, args.cv_seed, image_size)
                if cv_results:
                    logging.info("=== %d-fold 交叉验证 ===", len(cv_results["folds"]))
                    logging.info("整体平均重投影误差：%.3f px", cv_results["overall"]["mean_error"])
                    for fold in cv_results["folds"]:
                        logging.info(
                            "Fold %d -> train=%d, val=%d, mean=%.3f px, median=%.3f px, max=%.3f px",
                            fold["fold"], fold["train_size"], fold["val_size"],
                            fold["mean_error"], fold["median_error"], fold["max_error"],
                        )
            else:
                logging.warning("剔除后无可用样本，跳过交叉验证")
        else:
            logging.warning("成功样本缺少分辨率信息，无法执行交叉验证标定")
    elif not args.skip_calibration:
        logging.warning("没有成功样本，跳过交叉验证标定")

    if not filter_report:
        filter_report = {
            "initial_samples": len(successful_records),
            "final_samples": len(filtered_records),
            "removed_samples": [],
            "history": [],
        }

    samples = select_samples(calibration_records, args.sample_size, args.random_seed)

    figures_dir = output_root / "figures"
    viz_calib_records = cv_calibration_records if cv_calibration_records else calibration_records
    viz_translation_stats = cv_translation_stats if cv_translation_stats else external_translation_stats
    exported_calibration_payload: Optional[dict] = None
    exported_calibration_path: Optional[Path] = None

    figure_files = generate_visualizations(
        figures_dir,
        detection_records,
        viz_calib_records,
        viz_translation_stats,
        per_image_stats or None,
        filter_history or None,
    )

    if export_calib_path and cv_model and filtered_records:
        try:
            export_summary = cv_summary if cv_summary else calib_summary
            export_translation_stats = cv_translation_stats if cv_translation_stats else external_translation_stats
            export_records = cv_calibration_records if cv_calibration_records else viz_calib_records
            exported_calibration_payload = build_calibration_export_payload(
                cv_model,
                board_spec,
                image_size,
                filtered_records,
                per_image_stats,
                export_records,
                export_summary,
                export_translation_stats,
                filter_report,
            )
            ensure_dir(export_calib_path.parent)
            export_calib_path.write_text(
                json.dumps(exported_calibration_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logging.info("鲁棒标定结果已导出：%s", export_calib_path)
            exported_calibration_path = export_calib_path
        except Exception as exc:  # pylint: disable=broad-except
            logging.error("导出鲁棒标定结果失败：%s", exc)
    elif export_calib_path and cv_model is None and not args.skip_calibration:
        logging.warning("未获取鲁棒标定模型，跳过鲁棒标定导出")

    translation_payload = None
    if external_translation_stats:
        translation_payload = {
            "mean_translation_mm": external_translation_stats["mean_translation"].round(3).tolist(),
            "axis_error_mean_mm": external_translation_stats["axis_error_mean"].round(3).tolist(),
            "axis_error_std_mm": external_translation_stats["axis_error_std"].round(3).tolist(),
            "abs_error_mean_mm": external_translation_stats["abs_error_mean"].round(3).tolist(),
            "abs_error_std_mm": external_translation_stats["abs_error_std"].round(3).tolist(),
            "rel_error_mean_pct": (external_translation_stats["rel_error_mean"] * 100).round(3).tolist(),
            "rel_error_std_pct": (external_translation_stats["rel_error_std"] * 100).round(3).tolist(),
            "per_sample": [
                {
                    "name": entry["name"],
                    "axis_error_mm": [round(val, 4) for val in entry["axis_error_mm"]],
                    "abs_error_mm": [round(val, 4) for val in entry["abs_error_mm"]],
                    "rel_error_pct": [round(val * 100.0, 4) for val in entry["rel_error"]],
                }
                for entry in external_translation_stats.get("per_sample", [])
            ],
        }

    axis_error_log_path = write_axis_error_log(output_root, axis_error_logs)

    report_json = {
        "detection_summary": detection_summary,
        "detection_config": detection_config_info,
        "board_spec": board_spec.to_dict(),
        "detection_records": [
            {
                "name": rec.name,
                "success": rec.success,
                "reason": rec.reason,
                "elapsed_ms": rec.elapsed_ms,
                "small_count": rec.small_count,
                "big_count": rec.big_count,
            }
            for rec in detection_records
        ],
        "calibration_summary": calib_summary,
        "calibration_records": [asdict(rec) for rec in calibration_records],
        "samples": [asdict(rec) for rec in samples],
        "translation_stats": translation_payload,
        "cross_validation": cv_results,
        "cv_model": {
            "rms": cv_model["rms"] if cv_model else None,
            "num_samples": cv_model["num_samples"] if cv_model else None,
            "camera_matrix": cv_model["camera_matrix"].tolist() if cv_model else None,
            "dist_coeffs": cv_model["dist_coeffs"].ravel().tolist() if cv_model else None,
            "summary": cv_summary,
            "translation_stats": {
                "mean_translation_mm": cv_translation_stats["mean_translation"].round(3).tolist() if cv_translation_stats else None,
                "axis_error_mean_mm": cv_translation_stats["axis_error_mean"].round(3).tolist() if cv_translation_stats else None,
                "axis_error_std_mm": cv_translation_stats["axis_error_std"].round(3).tolist() if cv_translation_stats else None,
                "abs_error_mean_mm": cv_translation_stats["abs_error_mean"].round(3).tolist() if cv_translation_stats else None,
                "abs_error_std_mm": cv_translation_stats["abs_error_std"].round(3).tolist() if cv_translation_stats else None,
                "rel_error_mean_pct": (cv_translation_stats["rel_error_mean"] * 100).round(3).tolist() if cv_translation_stats else None,
                "rel_error_std_pct": (cv_translation_stats["rel_error_std"] * 100).round(3).tolist() if cv_translation_stats else None,
                "per_sample": [
                    {
                        "name": entry["name"],
                        "axis_error_mm": [round(val, 4) for val in entry["axis_error_mm"]],
                        "abs_error_mm": [round(val, 4) for val in entry["abs_error_mm"]],
                        "rel_error_pct": [round(val * 100.0, 4) for val in entry["rel_error"]],
                    }
                    for entry in cv_translation_stats.get("per_sample", [])
                ] if cv_translation_stats else None,
            } if cv_translation_stats else None,
        },
        "cv_calibration_records": [asdict(rec) for rec in cv_calibration_records],
        "filter_report": filter_report,
        "per_image_stats": per_image_stats,
        "filtered_sample_names": [rec.name for rec in filtered_records],
        "exported_calibration_path": str(exported_calibration_path) if exported_calibration_path else None,
        "robust_calibration": exported_calibration_payload,
        "figure_files": figure_files,
        "axis_error_log_path": str(axis_error_log_path) if axis_error_log_path else None,
    }
    (output_root / "evaluation_report.json").write_text(json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8")

    write_markdown_report(output_root / "evaluation_report.md", detection_summary, detection_records,
                          detection_config_info, board_spec,
                          calib_summary, calibration_records, samples, external_translation_stats,
                          cv_summary, cv_calibration_records, cv_translation_stats, cv_results,
                          filter_report, per_image_stats or None, figure_files)


if __name__ == "__main__":
    main()