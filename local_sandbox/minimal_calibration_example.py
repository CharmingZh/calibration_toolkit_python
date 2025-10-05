"""最小化标定示例脚本。

用法：
    python local_sandbox/minimal_calibration_example.py \
        --dataset /path/to/calib_images \
        --output outputs/local_sandbox_calibration \
        --limit 12

脚本会执行以下操作：
1. 遍历标定图像目录，读取灰度图并运行 Calibrator。
2. 过滤检测失败的图像，仅保留通过质量检查的样本。
3. 使用 OpenCV 的 calibrateCamera 计算相机内参与畸变系数。
4. 将结果写入指定输出目录，并打印关键指标。

需要提前设置 PYTHONPATH 指向项目根目录，或从项目根目录运行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

from calib.core.board_spec import DEFAULT_BOARD_SPEC
from calib.core.calib_core import Calibrator
from calib.utils.board import (
    create_circle_board_object_points_adaptive,
    extract_image_points_ordered,
)
from calib.utils.images import read_image_robust

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dng"}


def collect_images(dataset_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in dataset_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_SUFFIXES and p.is_file()
    )


def run_minimal_calibration(dataset: Path, output_dir: Path, limit: int | None) -> Path:
    calibrator = Calibrator()
    image_paths = collect_images(dataset)

    if limit is not None:
        image_paths = image_paths[:limit]

    if len(image_paths) < 5:
        raise RuntimeError("至少需要 5 张标定图像，当前可用图像不足。")

    object_points_list: List[np.ndarray] = []
    image_points_list: List[np.ndarray] = []
    used_images: list[str] = []
    image_size: tuple[int, int] | None = None

    for path in image_paths:
        gray = read_image_robust(str(path))
        if gray is None:
            print(f"跳过无法读取的文件: {path.name}")
            continue

        result = calibrator.process_gray(gray)
        if not (result and result.small_numbered):
            print(f"未检测到有效标定板: {path.name}")
            continue

        image_points = extract_image_points_ordered(result)
        if image_points is None:
            print(f"点提取失败: {path.name}")
            continue

        object_points = create_circle_board_object_points_adaptive(
            len(image_points), spacing=DEFAULT_BOARD_SPEC.center_spacing_mm
        )

        object_points_list.append(object_points)
        image_points_list.append(image_points)
        used_images.append(path.name)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

    if len(object_points_list) < 5:
        raise RuntimeError("有效图像不足，至少需要 5 张才能完成标定")

    assert image_size is not None

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points_list,
        image_points_list,
        image_size,
        None,
        None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "camera_calibration_minimal.json"

    calib_result = {
        "success": bool(ret),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.flatten().tolist(),
        "image_size": image_size,
        "num_images": len(object_points_list),
        "used_images": used_images,
    }

    result_path.write_text(json.dumps(calib_result, indent=2), encoding="utf-8")

    print("标定完成，写入:", result_path)
    print(
        "fx = %.2f, fy = %.2f, cx = %.2f, cy = %.2f"
        % (
            camera_matrix[0, 0],
            camera_matrix[1, 1],
            camera_matrix[0, 2],
            camera_matrix[1, 2],
        )
    )
    print("使用图像数量:", len(object_points_list))

    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="最小化相机标定示例")
    parser.add_argument("--dataset", required=True, help="标定图像所在目录")
    parser.add_argument(
        "--output",
        default="outputs/local_sandbox_calibration",
        help="输出目录 (默认: outputs/local_sandbox_calibration)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多使用多少张图像进行示例，默认使用全部图像",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        raise SystemExit(f"数据集目录不存在: {dataset_dir}")

    output_dir = Path(args.output)
    run_minimal_calibration(dataset_dir, output_dir, args.limit)


if __name__ == "__main__":
    main()
