# -*- coding: utf-8 -*-
"""
Improved Camera Calibration Script
改进的相机标定脚本

主要改进：
1. 更精确的圆心检测
2. 自动调整标定板参数
3. 更严格的质量控制
4. 多轮标定优化

使用方法：
    python tools/calibration/calibrate_intrinsics.py \
        --input data/raw/calibration/calib_25 \
        --output outputs/calibration/latest
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

from calib.core.board_spec import BoardSpec, DEFAULT_BOARD_SPEC
from calib.core.calib_core import Calibrator
from calib.utils.board import (
    create_circle_board_object_points_adaptive,
    extract_image_points_ordered,
)
from calib.utils.images import read_image_robust
from calib.viz.viz_steps import Visualizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO)


def find_project_root(start_dir: str, marker_rel: str = os.path.join("calib", "__init__.py")) -> str:
    cur = os.path.abspath(start_dir)
    last = None
    while cur and cur != last:
        if os.path.isfile(os.path.join(cur, marker_rel)):
            return cur
        last = cur
        cur = os.path.dirname(cur)
    return None


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = find_project_root(_THIS_DIR)
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def analyze_detection_quality(result, image_name: str) -> dict:
    """分析检测质量"""
    if result is None or not result.small_numbered:
        return {'valid': False, 'reason': '未检测到圆'}

    num_circles = len(result.small_numbered)
    expected_circles = 41

    if num_circles != expected_circles:
        return {
            'valid': False,
            'reason': f'圆数量不正确：检测到{num_circles}，期望{expected_circles}'
        }

    num_big_circles = len(result.big_circles)
    if num_big_circles != 4:
        return {
            'valid': False,
            'reason': f'大圆数量不正确：检测到{num_big_circles}，期望4'
        }

    x_coords = [c.x for c in result.small_numbered]
    y_coords = [c.y for c in result.small_numbered]

    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)

    if x_range < 500 or y_range < 500:
        return {
            'valid': False,
            'reason': f'圆分布范围太小：x_range={x_range:.0f}, y_range={y_range:.0f}'
        }

    return {'valid': True, 'reason': '检测质量良好'}


def analyze_calibration_quality(calib_result: dict) -> dict:
    """
    分析标定质量并生成详细评价报告
    """

    print("\n" + "="*70)
    print("📊 相机标定质量评价报告")
    print("="*70)

    image_size = calib_result['image_size']
    width, height = image_size
    fx = calib_result['focal_length_x']
    fy = calib_result['focal_length_y']
    cx = calib_result['principal_point_x']
    cy = calib_result['principal_point_y']
    reprojection_error = calib_result['reprojection_error']

    print("\n📸 基本信息:")
    print(f"   图像尺寸: {width} × {height} 像素")
    print(f"   使用图像数量: {calib_result['num_images']}")
    print(f"   重投影误差: {reprojection_error:.4f} 像素")

    print("\n🔍 相机内参分析:")
    print(f"   焦距 fx: {fx:.2f} 像素")
    print(f"   焦距 fy: {fy:.2f} 像素")
    print(f"   主点 cx: {cx:.2f} 像素 (中心偏移: {cx - width/2:.1f})")
    print(f"   主点 cy: {cy:.2f} 像素 (中心偏移: {cy - height/2:.1f})")

    aspect_ratio = fx / fy if fy != 0 else 0
    print(f"   焦距比例 fx/fy: {aspect_ratio:.4f}")

    aspect_ratio_quality = "good"
    if abs(aspect_ratio - 1.0) > 0.05:
        print("   ⚠️  焦距比例偏离1.0较多，可能存在像素不是正方形")
        aspect_ratio_quality = "warning"
    else:
        print("   ✅ 焦距比例接近1.0，像素近似正方形")

    cx_offset = abs(cx - width/2)
    cy_offset = abs(cy - height/2)
    print(f"   主点偏移: ({cx_offset:.1f}, {cy_offset:.1f}) 像素")

    principal_point_quality = "good"
    if cx_offset > width * 0.05 or cy_offset > height * 0.05:
        print("   ⚠️  主点偏离图像中心较远")
        principal_point_quality = "warning"
    else:
        print("   ✅ 主点接近图像中心")

    print("\n👁️  视场角估算:")
    sensor_width_mm = 5.76
    sensor_height_mm = sensor_width_mm * height / width

    focal_length_mm_x = fx * sensor_width_mm / width if width != 0 else 0
    focal_length_mm_y = fy * sensor_height_mm / height if height != 0 else 0

    print(f"   估算传感器尺寸: {sensor_width_mm:.2f} × {sensor_height_mm:.2f} mm")
    print(f"   等效焦距: {focal_length_mm_x:.2f} mm (水平), {focal_length_mm_y:.2f} mm (垂直)")

    fov_x = 2 * math.atan(sensor_width_mm / (2 * focal_length_mm_x)) * 180 / math.pi if focal_length_mm_x != 0 else 0
    fov_y = 2 * math.atan(sensor_height_mm / (2 * focal_length_mm_y)) * 180 / math.pi if focal_length_mm_y != 0 else 0

    print(f"   水平视场角: {fov_x:.1f}°")
    print(f"   垂直视场角: {fov_y:.1f}°")

    print("\n📐 畸变系数分析:")
    dist_coeffs = calib_result['distortion_coefficients']
    k1, k2, p1, p2, k3 = dist_coeffs[:5] if len(dist_coeffs) >= 5 else dist_coeffs + [0] * (5 - len(dist_coeffs))

    print(f"   径向畸变 k1: {k1:.6f}")
    print(f"   径向畸变 k2: {k2:.6f}")
    print(f"   径向畸变 k3: {k3:.6f}")
    print(f"   切向畸变 p1: {p1:.6f}")
    print(f"   切向畸变 p2: {p2:.6f}")

    radial_distortion = abs(k1) + abs(k2) + abs(k3)
    tangential_distortion = abs(p1) + abs(p2)

    distortion_quality = "good"
    print(f"\n   畸变严重程度:")
    if radial_distortion > 1.0:
        print("   ⚠️  径向畸变较严重，建议进行畸变矫正")
        distortion_quality = "high"
    elif radial_distortion > 0.3:
        print("   ⚡ 径向畸变中等，可能需要矫正")
        distortion_quality = "medium"
    else:
        print("   ✅ 径向畸变较小")

    if tangential_distortion > 0.1:
        print("   ⚠️  切向畸变较明显")
        if distortion_quality == "good":
            distortion_quality = "medium"
    else:
        print("   ✅ 切向畸变很小")

    print("\n📊 标定质量综合评估:")

    if reprojection_error < 0.5:
        error_quality = "excellent"
        quality_icon = "🌟"
        quality_text = "优秀"
    elif reprojection_error < 1.0:
        error_quality = "good"
        quality_icon = "✅"
        quality_text = "良好"
    elif reprojection_error < 2.0:
        error_quality = "fair"
        quality_icon = "⚡"
        quality_text = "一般"
    elif reprojection_error < 10.0:
        error_quality = "poor"
        quality_icon = "⚠️"
        quality_text = "较差"
    else:
        error_quality = "very_poor"
        quality_icon = "❌"
        quality_text = "很差"

    print(f"   {quality_icon} 重投影误差: {reprojection_error:.4f} 像素")
    print(f"   标定质量等级: {quality_text}")

    print("\n💡 应用建议:")
    if error_quality in ["excellent", "good"]:
        print("   ✅ 可用于高精度应用:")
        print("      - 精密3D测量")
        print("      - 工业检测")
        print("      - 机器人视觉导航")
        print("      - AR/VR应用")
    elif error_quality == "fair":
        print("   ⚡ 适用于一般精度应用:")
        print("      - 图像畸变矫正")
        print("      - 基础3D重建")
        print("      - 一般计算机视觉任务")
        print("   ❌ 不推荐用于高精度测量")
    else:
        print("   ⚠️  当前标定质量较低，建议:")
        print("      - 重新拍摄标定图像")
        print("      - 检查标定板物理尺寸设置")
        print("      - 增加更多高质量标定图像")
        print("      - 确保标定板在图像中完整清晰")

    if error_quality not in ["excellent", "good"]:
        print("\n🔧 改进建议:")
        print("   1. 图像采集改进:")
        print("      - 确保标定板平整无弯曲")
        print("      - 使用三脚架固定相机避免抖动")
        print("      - 均匀光照，避免反光和阴影")
        print("      - 标定板应占图像面积的20-50%")

        print("\n   2. 拍摄角度多样化:")
        print("      - 标定板应出现在图像的不同位置")
        print("      - 包含不同的倾斜角度")
        print("      - 拍摄15-30张不同角度的图像")

        print("\n   3. 参数检查:")
        print("      - 精确测量标定板圆心间距")
        print("      - 确认标定板类型和尺寸设置")

    print("\n" + "="*70)
    print("💻 畸变矫正示例代码")
    print("="*70)

    camera_matrix = calib_result['camera_matrix']
    dist_coeffs = calib_result['distortion_coefficients']

    print("```python")
    print("import cv2")
    print("import numpy as np")
    print()
    print("# 相机内参和畸变系数")
    print(f"camera_matrix = np.array({camera_matrix})")
    print(f"dist_coeffs = np.array({dist_coeffs})")
    print(f"image_size = {image_size}  # (width, height)")
    print()
    print("# 读取需要矫正的图像")
    print("img = cv2.imread('your_image.jpg')")
    print()
    print("# 计算最优的新相机矩阵")
    print("new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(")
    print("    camera_matrix, dist_coeffs, image_size, 1, image_size)")
    print()
    print("# 畸变矫正")
    print("undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)")
    print()
    print("# 裁剪有效区域（可选）")
    print("x, y, w, h = roi")
    print("undistorted_img = undistorted_img[y:y+h, x:x+w]")
    print()
    print("# 保存结果")
    print("cv2.imwrite('undistorted_image.jpg', undistorted_img)")
    print("```")

    evaluation = {
        'overall_quality': error_quality,
        'reprojection_error_quality': error_quality,
        'aspect_ratio_quality': aspect_ratio_quality,
        'principal_point_quality': principal_point_quality,
        'distortion_quality': distortion_quality,
        'recommended_applications': [],
        'improvement_suggestions': []
    }

    if error_quality in ["excellent", "good"]:
        evaluation['recommended_applications'] = [
            "high_precision_measurement", "industrial_inspection",
            "robot_vision", "ar_vr_applications"
        ]
    elif error_quality == "fair":
        evaluation['recommended_applications'] = [
            "distortion_correction", "basic_3d_reconstruction",
            "general_computer_vision"
        ]

    if error_quality not in ["excellent", "good"]:
        evaluation['improvement_suggestions'] = [
            "retake_calibration_images", "check_board_dimensions",
            "add_more_quality_images", "ensure_board_visibility"
        ]

    print("="*70)

    return evaluation


def perform_multi_stage_calibration(object_points_list: List[np.ndarray],
                                   image_points_list: List[np.ndarray],
                                   image_size: Tuple[int, int]) -> dict:
    """多阶段标定，提高精度"""

    logging.info(f"开始多阶段相机标定，使用 {len(object_points_list)} 张图像...")

    logging.info("阶段1: 标准标定...")
    ret1, camera_matrix1, dist_coeffs1, rvecs1, tvecs1 = cv2.calibrateCamera(
        object_points_list, image_points_list, image_size, None, None
    )

    if not ret1:
        return {'success': False, 'error': '标准标定失败'}

    errors = []
    for i in range(len(object_points_list)):
        projected_points, _ = cv2.projectPoints(
            object_points_list[i], rvecs1[i], tvecs1[i],
            camera_matrix1, dist_coeffs1
        )
        projected_points = projected_points.reshape(-1, 2)
        error = cv2.norm(image_points_list[i], projected_points, cv2.NORM_L2)
        errors.append(error / len(object_points_list[i]))

    sorted_indices = sorted(range(len(errors)), key=lambda idx: errors[idx])
    keep_count = max(3, int(len(errors) * 0.8))
    good_indices = sorted_indices[:keep_count]

    logging.info(f"筛选出 {len(good_indices)}/{len(errors)} 张高质量图像")

    if len(good_indices) >= 3:
        logging.info("阶段2: 使用筛选图像重新标定...")
        filtered_obj_points = [object_points_list[i] for i in good_indices]
        filtered_img_points = [image_points_list[i] for i in good_indices]

        ret2, camera_matrix2, dist_coeffs2, rvecs2, tvecs2 = cv2.calibrateCamera(
            filtered_obj_points, filtered_img_points, image_size,
            camera_matrix1, dist_coeffs1
        )

        if ret2:
            camera_matrix, dist_coeffs, rvecs, tvecs = camera_matrix2, dist_coeffs2, rvecs2, tvecs2
            used_images = len(good_indices)
        else:
            camera_matrix, dist_coeffs, rvecs, tvecs = camera_matrix1, dist_coeffs1, rvecs1, tvecs1
            used_images = len(object_points_list)
    else:
        camera_matrix, dist_coeffs, rvecs, tvecs = camera_matrix1, dist_coeffs1, rvecs1, tvecs1
        used_images = len(object_points_list)

    total_error = 0
    total_points = 0
    final_obj_points = [object_points_list[i] for i in good_indices] if len(good_indices) >= 3 else object_points_list
    final_img_points = [image_points_list[i] for i in good_indices] if len(good_indices) >= 3 else image_points_list

    for i in range(len(final_obj_points)):
        projected_points, _ = cv2.projectPoints(
            final_obj_points[i], rvecs[i], tvecs[i],
            camera_matrix, dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        error = cv2.norm(final_img_points[i], projected_points, cv2.NORM_L2)
        total_error += error * error
        total_points += len(final_obj_points[i])

    mean_error = np.sqrt(total_error / total_points)

    result = {
        'success': True,
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.flatten().tolist(),
        'image_size': image_size,
        'num_images': used_images,
        'reprojection_error': float(mean_error),
        'focal_length_x': float(camera_matrix[0, 0]),
        'focal_length_y': float(camera_matrix[1, 1]),
        'principal_point_x': float(camera_matrix[0, 2]),
        'principal_point_y': float(camera_matrix[1, 2]),
    }

    logging.info(f"多阶段标定完成！重投影误差: {mean_error:.4f} 像素")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/calibration/calib_25", help="输入图像文件夹")
    ap.add_argument("--output", default="outputs/calibration/latest", help="输出结果文件夹")
    ap.add_argument("--small-diameter", "--small_diameter", dest="small_diameter",
                    type=float, default=DEFAULT_BOARD_SPEC.small_diameter_mm, help="小圆直径(mm)")
    ap.add_argument("--circle-spacing", "--circle_spacing", dest="circle_spacing",
                    type=float, default=DEFAULT_BOARD_SPEC.center_spacing_mm, help="圆心间距离(mm)")
    ap.add_argument("--save_viz", action="store_true", help="保存可视化结果")
    ap.add_argument("--min_images", type=int, default=5, help="最少需要的有效图像数")
    args = ap.parse_args()

    board_spec = BoardSpec(
        small_diameter_mm=args.small_diameter,
        center_spacing_mm=args.circle_spacing,
    )

    logging.info(
        "使用标定板规格: 小圆直径 %.2f mm, 圆心距离 %.2f mm",
        board_spec.small_diameter_mm,
        board_spec.center_spacing_mm,
    )

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.dng", "*.DNG"]
    image_paths = []
    for ext in image_extensions:
        pattern = os.path.join(args.input, ext)
        image_paths.extend(glob.glob(pattern))
    image_paths = sorted(image_paths)

    if not image_paths:
        logging.error(f"未找到图像文件在 {args.input}")
        return

    logging.info(f"找到 {len(image_paths)} 张图像")

    ensure_dir(args.output)

    calibrator = Calibrator()
    if args.save_viz:
        visualizer = Visualizer(args.output)

    successful_images = []
    object_points_list = []
    image_points_list = []
    image_size = None
    quality_reports = []

    for i, image_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        logging.info(f"处理 [{i+1}/{len(image_paths)}] {base_name}...")

        gray = read_image_robust(image_path)
        if gray is None:
            continue

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        try:
            result = calibrator.process_gray(gray)

            quality = analyze_detection_quality(result, base_name)
            quality_reports.append((base_name, quality))

            if not quality['valid']:
                logging.warning(f"质量检查失败 {base_name}: {quality['reason']}")
                continue

            image_points = extract_image_points_ordered(result)
            if image_points is None:
                continue

            object_points = create_circle_board_object_points_adaptive(
                len(image_points), spacing=args.circle_spacing, board_spec=board_spec
            )

            object_points_list.append(object_points)
            image_points_list.append(image_points)
            successful_images.append(base_name)

            logging.info(f"✅ {base_name}: 通过质量检查")

            if args.save_viz:
                try:
                    visualizer.save_all(gray, base_name, result)
                except Exception as exc:
                    logging.warning(f"可视化保存失败 {base_name}: {exc}")

        except Exception as exc:
            logging.error(f"处理图像失败 {base_name}: {exc}")

    if len(successful_images) < args.min_images:
        logging.error(f"有效图像太少 ({len(successful_images)})，至少需要{args.min_images}张")
        return

    logging.info(f"通过质量检查的图像: {len(successful_images)} 张")

    calib_result = perform_multi_stage_calibration(
        object_points_list, image_points_list, image_size
    )

    if not calib_result.get('success', False):
        logging.error("相机标定失败")
        return

    evaluation = analyze_calibration_quality(calib_result)

    result_file = os.path.join(args.output, "camera_calibration_improved.json")
    calib_result['board_spec'] = board_spec.to_dict()
    calib_result['successful_images'] = successful_images
    calib_result['quality_reports'] = quality_reports
    calib_result['quality_evaluation'] = evaluation

    with open(result_file, 'w') as f:
        json.dump(calib_result, f, indent=2)

    logging.info(f"改进的标定结果已保存到: {result_file}")

    print("\n" + "🎯 " + "="*60)
    print("最终标定总结")
    print("="*66)

    error = calib_result['reprojection_error']
    if error < 1.0:
        status = "🎉 标定成功 - 质量优秀"
    elif error < 2.0:
        status = "✅ 标定成功 - 质量良好"
    elif error < 10.0:
        status = "⚡ 标定完成 - 质量一般，建议改进"
    else:
        status = "⚠️ 标定完成 - 质量较差，强烈建议重新标定"

    print(f"状态: {status}")
    print(f"有效图像: {calib_result['num_images']} 张")
    print(f"重投影误差: {error:.3f} 像素")

    print("\n关键内参:")
    print(f"  fx = {calib_result['focal_length_x']:.1f}")
    print(f"  fy = {calib_result['focal_length_y']:.1f}")
    print(f"  cx = {calib_result['principal_point_x']:.1f}")
    print(f"  cy = {calib_result['principal_point_y']:.1f}")

    print("\n📁 输出文件:")
    print(f"  标定参数: {result_file}")
    if args.save_viz:
        print(f"  可视化结果: {args.output}/<图像名>_*.png")

    print("="*66)


if __name__ == "__main__":
    main()
