# calib/viz/viz_steps.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from ..core.types import BoardResult, Pt

__all__ = ["Visualizer"]


# ---------- small helpers (独立于core，避免循环依赖) ----------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def to_bgr(g: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def annotate_text(img, text, xy, color=(0, 0, 255), font_scale=0.95, thick=2):
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.putText(img, str(text), (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, str(text), (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)

def H_inv_point(H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    p = np.array([x, y, 1.0], np.float32)
    q = np.linalg.inv(H) @ p
    return (float(q[0] / q[2]), float(q[1] / q[2]))

def draw_poly_fancy(img, pts, color_a, color_b, width=2, alpha=0.9):
    if len(pts) < 2:
        return
    overlay = img.copy()
    n = len(pts) - 1
    def lerp(a, b, t): return a + (b - a) * t
    for k in range(n):
        t0 = k / float(max(1, n)); t1 = (k + 1) / float(max(1, n))
        c0 = tuple(int(lerp(color_a[i], color_b[i], t0)) for i in range(3))
        c1 = tuple(int(lerp(color_a[i], color_b[i], t1)) for i in range(3))
        cv2.line(overlay, (int(pts[k][0]), int(pts[k][1])),
                          (int(pts[k+1][0]), int(pts[k+1][1])), c0, width + 4, cv2.LINE_AA)
        cv2.line(overlay, (int(pts[k][0]), int(pts[k][1])),
                          (int(pts[k+1][0]), int(pts[k+1][1])), c1, width, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_dashed(img, p, q, color=(200,200,200), width=1, dash=9, gap=8, alpha=0.5):
    p = np.array(p, np.float32); q = np.array(q, np.float32)
    v = q - p; L = np.linalg.norm(v) + 1e-9; d = v / L
    nseg = int(L // (dash + gap)) + 1
    overlay = img.copy()
    for i in range(nseg):
        a = p + d * (i * (dash + gap))
        b = p + d * (i * (dash + gap) + dash)
        b = p + d * min(np.linalg.norm(b - p), L)
        cv2.line(overlay, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 color, width, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


class Visualizer:
    """
    只做“可视化与落盘”，不参与主流程计时。
    依赖 BoardResult 里的字段进行还原绘制，避免与 core 互相引用。
    """

    def __init__(
        self,
        out_root: str,
        calibration_path: Optional[str] = None,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        board_spacing: float = 25.0,
    ):
        self.out_root = out_root
        ensure_dir(out_root)

        self.board_spacing = float(board_spacing)
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self._calibration_path: Optional[Path] = None

        if camera_matrix is not None:
            self.camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
            if dist_coeffs is not None:
                coeffs = np.asarray(dist_coeffs, dtype=np.float32)
                self.dist_coeffs = coeffs.reshape(-1, 1) if coeffs.ndim == 1 else coeffs.astype(np.float32)
            else:
                self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        else:
            candidate = None
            if calibration_path:
                candidate = Path(calibration_path)
            else:
                candidate = self._guess_default_calibration()

            if candidate and candidate.is_file():
                self._load_calibration(candidate)

        # 行彩虹调色板
        self.palette = [
            (255, 64,  64),   # R
            (255, 180, 72),   # O
            (240, 240, 64),   # Y
            (64,  220, 64),   # G
            (64,  220, 220),  # C
            (64,   96, 255),  # B
            (255,  64, 255),  # M
        ]

    def _guess_default_calibration(self) -> Optional[Path]:
        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "outputs" / "calibration" / "latest" / "camera_calibration_improved.json",
            root / "outputs" / "calibration" / "latest" / "camera_calibration.json",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand
        return None

    def _load_calibration(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("[Visualizer] 读取标定文件失败 %s: %s", path, exc)
            return

        matrix = data.get("camera_matrix") or data.get("cameraMatrix")
        if matrix is not None:
            mat = np.asarray(matrix, dtype=np.float32)
            if mat.shape == (3, 3):
                self.camera_matrix = mat
            else:
                logging.warning("[Visualizer] camera_matrix 形状异常: %s", mat.shape)
        else:
            logging.warning("[Visualizer] 标定文件缺少 camera_matrix 字段: %s", path)

        coeffs = (data.get("distortion_coefficients")
                  or data.get("dist_coeffs")
                  or data.get("distortion"))
        if coeffs is not None:
            arr = np.asarray(coeffs, dtype=np.float32)
            self.dist_coeffs = arr.reshape(-1, 1) if arr.ndim == 1 else arr.astype(np.float32)
        else:
            if self.camera_matrix is not None:
                self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        if self.camera_matrix is not None:
            self._calibration_path = path

    def _draw_axes_2d(self, canvas: np.ndarray, H: np.ndarray, O: np.ndarray,
                      x_hat: np.ndarray, y_hat: np.ndarray, L: float) -> None:
        X0, Y0 = H_inv_point(H, O[0], O[1])
        Xx, Yx = H_inv_point(H, O[0] + x_hat[0] * L, O[1] + x_hat[1] * L)
        Xy, Yy = H_inv_point(H, O[0] + y_hat[0] * L, O[1] + y_hat[1] * L)

        origin = (int(round(X0)), int(round(Y0)))
        end_x = (int(round(Xx)), int(round(Yx)))
        end_y = (int(round(Xy)), int(round(Yy)))

        cv2.arrowedLine(canvas, origin, end_x, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.08)
        cv2.arrowedLine(canvas, origin, end_y, (0, 200, 0), 3, cv2.LINE_AA, tipLength=0.08)
        annotate_text(canvas, "x", (Xx, Yx), (0, 0, 255), 0.9, 2)
        annotate_text(canvas, "y", (Xy, Yy), (0, 200, 0), 0.9, 2)
    # ---- 对单张图生成所有关键可视化，与 v3.1 的命名保持一致 ----
    def save_all(self, gray: np.ndarray, base: str, result: BoardResult) -> None:
        """
        参数：
          gray: 原始灰度图
          base: 文件基名（不含扩展名）
          result: BoardResult（核心算法产出）
        输出：
          out_root 下生成:
            {base}_0_raw.png
            {base}_1_quad.png
            {base}_2_rect.png
            {base}_3_rect_refined.png
            {base}_5_rect_numbered.png
            {base}_6_raw_numbered.png
        """
        ensure_dir(self.out_root)

        # 0) 原图
        cv2.imwrite(os.path.join(self.out_root, f"{base}_0_raw.png"), to_bgr(gray))

        # 1) raw + quad
        raw_quad = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        quad = np.asarray(result.quad, dtype=np.float32)
        if quad.shape == (4, 2):
            cv2.polylines(raw_quad, [quad.astype(np.int32)], True, (0, 200, 255), 2)
        cv2.imwrite(os.path.join(self.out_root, f"{base}_1_quad.png"), raw_quad)

        # 2) rect（用 homography 还原）
        H = np.asarray(result.homography, dtype=np.float32)
        # 粗估 rect 尺寸（四点的包围盒宽高）
        W = int(max(400, np.linalg.norm(quad[1] - quad[0]))) if quad.shape == (4,2) else gray.shape[1]
        Hh = int(max(400, np.linalg.norm(quad[3] - quad[0]))) if quad.shape == (4,2) else gray.shape[0]
        dst = cv2.warpPerspective(gray, H, (W, Hh))
        # 简单增强（与核心一致）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        rect_pre = cv2.GaussianBlur(clahe.apply(dst), (3, 3), 0)
        cv2.imwrite(os.path.join(self.out_root, f"{base}_2_rect.png"), to_bgr(rect_pre))

        # 3) rect_refined（把小/大圆的 refined 结果画出来）
        rect_refined = cv2.cvtColor(rect_pre, cv2.COLOR_GRAY2BGR)
        for it in result.small_circles_rect:
            cx, cy = int(round(it.x)), int(round(it.y))
            cv2.circle(rect_refined, (cx, cy), 2, (0, 255, 0), -1, cv2.LINE_AA)
        for it in result.big_circles:
            cx, cy = int(round(it.x)), int(round(it.y))
            cv2.circle(rect_refined, (cx, cy), 2, (0, 128, 255), -1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.out_root, f"{base}_3_rect_refined.png"), rect_refined)

        # 5) rect_numbered（坐标轴 + 序号 + 行连线）
        rect_num = cv2.cvtColor(rect_pre, cv2.COLOR_GRAY2BGR)

        # 坐标轴
        O = np.array(result.axis_origin_rect, np.float32)
        x_hat = np.array(result.axis_x_hat, np.float32)
        y_hat = np.array(result.axis_y_hat, np.float32)
        L = 180.0
        p0 = (float(O[0]), float(O[1]))
        px = (float(O[0] + x_hat[0] * L), float(O[1] + x_hat[1] * L))
        py = (float(O[0] + y_hat[0] * L), float(O[1] + y_hat[1] * L))
        cv2.arrowedLine(rect_num, (int(p0[0]), int(p0[1])), (int(px[0]), int(px[1])), (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.08)
        cv2.arrowedLine(rect_num, (int(p0[0]), int(p0[1])), (int(py[0]), int(py[1])), (0, 200, 0), 3, cv2.LINE_AA, tipLength=0.08)
        annotate_text(rect_num, "x", px, (0, 0, 255), 0.9, 2)
        annotate_text(rect_num, "y", py, (0, 200, 0), 0.9, 2)

        # 序号 + 按行连线
        # rows: 由 small_numbered 的 row 分组，并且保证行内仍按 seq 顺序
        K = 7
        rows_pts: List[List[Tuple[float, float]]] = [[] for _ in range(K)]
        for d in sorted(result.small_numbered, key=lambda t: (t.row, t.col)):
            annotate_text(rect_num, d.seq, (d.x + 5, d.y - 6), (0, 0, 255), 0.9, 2)
            cv2.circle(rect_num, (int(round(d.x)), int(round(d.y))), 2, (0, 255, 0), -1, cv2.LINE_AA)
            if 0 <= d.row < K:
                rows_pts[d.row].append((float(d.x), float(d.y)))

        for r, pts in enumerate(rows_pts):
            if len(pts) >= 2:
                color_a = self.palette[r % len(self.palette)]
                color_b = (min(255, color_a[0] + 60), min(255, color_a[1] + 40), min(255, color_a[2] + 60))
                draw_poly_fancy(rect_num, pts, color_a, color_b, width=3, alpha=0.85)
            if r < len(rows_pts) - 1 and pts and rows_pts[r + 1]:
                draw_dashed(rect_num, pts[-1], rows_pts[r + 1][0],
                            (220, 220, 220), width=1, dash=10, gap=9, alpha=0.45)

        cv2.imwrite(os.path.join(self.out_root, f"{base}_5_rect_numbered.png"), rect_num)

        # 6) raw_numbered（把 rect 标注回投到 raw）
        raw_num = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        numbered_sorted = sorted(result.small_numbered, key=lambda t: (t.row, t.col))
        raw_coords_map: Dict[int, Tuple[float, float]] = {}
        rows_pts_raw: List[List[Tuple[float, float]]] = [[] for _ in range(K)]
        for d in numbered_sorted:
            X, Y = H_inv_point(H, d.x, d.y)
            raw_coords_map[d.seq] = (float(X), float(Y))
            if 0 <= d.row < K:
                rows_pts_raw[d.row].append((float(X), float(Y)))

        self._draw_axes_2d(raw_num, H, O, x_hat, y_hat, L)

        for d in numbered_sorted:
            X, Y = raw_coords_map[d.seq]
            annotate_text(raw_num, d.seq, (X + 5, Y - 6), (0, 0, 255), 0.9, 2)
            cv2.circle(raw_num, (int(round(X)), int(round(Y))), 2, (0, 255, 0), -1, cv2.LINE_AA)

        for r, pts in enumerate(rows_pts_raw):
            if len(pts) >= 2:
                color_a = self.palette[r % len(self.palette)]
                color_b = (min(255, color_a[0] + 60), min(255, color_a[1] + 40), min(255, color_a[2] + 60))
                draw_poly_fancy(raw_num, pts, color_a, color_b, width=3, alpha=0.85)
            if r < len(rows_pts_raw) - 1 and pts and rows_pts_raw[r + 1]:
                draw_dashed(raw_num, pts[-1], rows_pts_raw[r + 1][0],
                            (220, 220, 220), width=1, dash=10, gap=9, alpha=0.45)

        cv2.imwrite(os.path.join(self.out_root, f"{base}_6_raw_numbered.png"), raw_num)