#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调试标定板四点检测的工具脚本。

用法：
    python tests/debug_quad_detection.py --image data/raw/calibration/calib_25/IMG_8139.DNG --out outputs/tests/debug_quad

输出：
    - *_raw.png 原始灰度图
    - *_edges.png 霍夫检测使用的边缘图
    - *_hough_quad.png 霍夫候选四边形覆盖图
    - *_white_mask.png 连通域检测得到的二值掩码
    - *_white_quad.png 连通域候选四边形覆盖图
    - *_selected_quad.png 最终选择的四边形+外扩四边形
    - *_rect.png 透视展开后的图像
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 项目内导入
from calib.core.calib_core import (
    detect_by_hough_search,
    detect_by_white_region,
    warp_by_quad,
    expand_quad,
)
from calib.utils.images import read_image_robust


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def draw_quad(img: np.ndarray, quad: Optional[np.ndarray], color=(0, 255, 255), thickness=2) -> np.ndarray:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    if quad is not None:
        q = quad.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [q], True, color, thickness, cv2.LINE_AA)
    return vis


def save(path: str, image: np.ndarray) -> None:
    cv2.imwrite(path, image)


def main() -> None:
    parser = argparse.ArgumentParser(description="调试标定板四点检测")
    parser.add_argument("--image", required=True, help="输入图像路径（支持 DNG）")
    parser.add_argument("--out", default="debug_quad", help="输出目录")
    parser.add_argument("--expand-scale", type=float, default=1.05, help="四边形缩放系数")
    parser.add_argument("--expand-offset", type=float, default=16.0, help="四边形向外偏移像素")
    parser.add_argument("--min-short", type=int, default=1400, help="透视展开最短边长度")
    args = parser.parse_args()

    ensure_dir(args.out)
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    gray = read_image_robust(args.image)
    if gray is None:
        raise RuntimeError(f"无法读取图像: {args.image}")

    save(os.path.join(args.out, f"{base_name}_raw.png"), gray)

    quad_h, edges = detect_by_hough_search(gray)
    if edges is not None:
        save(os.path.join(args.out, f"{base_name}_edges.png"), edges)
    if quad_h is not None:
        save(
            os.path.join(args.out, f"{base_name}_hough_quad.png"),
            draw_quad(gray, quad_h, color=(0, 220, 255), thickness=2),
        )

    quad_w, mask = detect_by_white_region(gray)
    if mask is not None:
        save(os.path.join(args.out, f"{base_name}_white_mask.png"), mask)
    if quad_w is not None:
        save(
            os.path.join(args.out, f"{base_name}_white_quad.png"),
            draw_quad(gray, quad_w, color=(0, 255, 0), thickness=2),
        )

    # 选择优先级：霍夫成功优先，否则使用白块
    quad_selected = quad_h if quad_h is not None else quad_w
    if quad_selected is None:
        raise RuntimeError("未检测到四边形")

    quad_expanded = expand_quad(quad_selected, scale=args.expand_scale, offset=args.expand_offset)

    vis_final = draw_quad(gray, quad_selected, color=(255, 128, 0), thickness=2)
    vis_final = draw_quad(vis_final, quad_expanded, color=(0, 0, 255), thickness=2)
    save(os.path.join(args.out, f"{base_name}_selected_quad.png"), vis_final)

    rect, _ = warp_by_quad(gray, quad_expanded, min_short=args.min_short)
    save(os.path.join(args.out, f"{base_name}_rect.png"), rect)

    print(f"✅ 完成，调试输出保存在: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
