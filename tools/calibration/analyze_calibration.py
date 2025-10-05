# -*- coding: utf-8 -*-
"""Post-calibration analysis helpers."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np


def analyze_camera_parameters(calib_data):
    """分析相机参数"""

    print("="*70)
    print("相机标定结果详细分析")
    print("="*70)

    image_size = calib_data['image_size']
    width, height = image_size
    fx = calib_data['focal_length_x']
    fy = calib_data['focal_length_y']
    cx = calib_data['principal_point_x']
    cy = calib_data['principal_point_y']

    print("\n📸 基本信息:")
    print(f"   图像尺寸: {width} × {height} 像素")
    print(f"   使用图像数量: {calib_data['num_images']}")
    print(f"   重投影误差: {calib_data['reprojection_error']:.4f} 像素")

    print("\n🔍 相机内参分析:")
    print(f"   焦距 fx: {fx:.2f} 像素")
    print(f"   焦距 fy: {fy:.2f} 像素")
    print(f"   主点 cx: {cx:.2f} 像素 (中心偏移: {cx - width/2:.1f})")
    print(f"   主点 cy: {cy:.2f} 像素 (中心偏移: {cy - height/2:.1f})")

    aspect_ratio = fx / fy
    print(f"   焦距比例 fx/fy: {aspect_ratio:.4f}")
    if abs(aspect_ratio - 1.0) > 0.05:
        print("   ⚠️  焦距比例偏离1.0较多，可能存在像素不是正方形")
    else:
        print("   ✅ 焦距比例接近1.0，像素近似正方形")

    cx_offset = abs(cx - width/2)
    cy_offset = abs(cy - height/2)
    print(f"   主点偏移: ({cx_offset:.1f}, {cy_offset:.1f}) 像素")

    if cx_offset > width * 0.05 or cy_offset > height * 0.05:
        print("   ⚠️  主点偏离图像中心较远")
    else:
        print("   ✅ 主点接近图像中心")

    print("\n👁️  视场角分析:")

    sensor_width_mm = 5.76
    sensor_height_mm = sensor_width_mm * height / width

    focal_length_mm_x = fx * sensor_width_mm / width
    focal_length_mm_y = fy * sensor_height_mm / height

    print(f"   估算传感器尺寸: {sensor_width_mm:.2f} × {sensor_height_mm:.2f} mm")
    print(f"   等效焦距: {focal_length_mm_x:.2f} mm (水平), {focal_length_mm_y:.2f} mm (垂直)")

    fov_x = 2 * math.atan(sensor_width_mm / (2 * focal_length_mm_x)) * 180 / math.pi
    fov_y = 2 * math.atan(sensor_height_mm / (2 * focal_length_mm_y)) * 180 / math.pi
    fov_diagonal = 2 * math.atan(math.sqrt(sensor_width_mm**2 + sensor_height_mm**2) /
                                (2 * math.sqrt(focal_length_mm_x**2 + focal_length_mm_y**2))) * 180 / math.pi

    print(f"   水平视场角: {fov_x:.1f}°")
    print(f"   垂直视场角: {fov_y:.1f}°")
    print(f"   对角视场角: {fov_diagonal:.1f}°")

    print("\n📐 畸变系数分析:")
    dist_coeffs = calib_data['distortion_coefficients']
    k1, k2, p1, p2, k3 = dist_coeffs

    print(f"   径向畸变 k1: {k1:.6f}")
    print(f"   径向畸变 k2: {k2:.6f}")
    print(f"   径向畸变 k3: {k3:.6f}")
    print(f"   切向畸变 p1: {p1:.6f}")
    print(f"   切向畸变 p2: {p2:.6f}")

    radial_distortion = abs(k1) + abs(k2) + abs(k3)
    tangential_distortion = abs(p1) + abs(p2)

    print(f"\n   畸变严重程度:")
    if radial_distortion > 1.0:
        print("   ⚠️  径向畸变较严重，建议进行畸变矫正")
    elif radial_distortion > 0.3:
        print("   ⚡ 径向畸变中等，可能需要矫正")
    else:
        print("   ✅ 径向畸变较小")

    if tangential_distortion > 0.1:
        print("   ⚠️  切向畸变较明显")
    else:
        print("   ✅ 切向畸变很小")

    print("\n📊 标定质量评估:")
    reprojection_error = calib_data['reprojection_error']

    if reprojection_error < 0.5:
        quality = "优秀"
        quality_icon = "🌟"
    elif reprojection_error < 1.0:
        quality = "良好"
        quality_icon = "✅"
    elif reprojection_error < 2.0:
        quality = "一般"
        quality_icon = "⚡"
    else:
        quality = "较差"
        quality_icon = "⚠️"

    print(f"   {quality_icon} 重投影误差: {reprojection_error:.4f} 像素")
    print(f"   标定质量: {quality}")

    if reprojection_error > 2.0:
        print("\n💡 改进建议:")
        print("   - 增加更多不同角度和位置的标定图像")
        print("   - 确保标定板在图像中清晰可见")
        print("   - 检查标定板的物理尺寸设置是否正确")
        print("   - 考虑使用更高精度的标定板")

    return {
        'fov_horizontal': fov_x,
        'fov_vertical': fov_y,
        'focal_length_mm': (focal_length_mm_x + focal_length_mm_y) / 2,
        'quality': quality,
        'distortion_level': 'high' if radial_distortion > 1.0 else 'medium' if radial_distortion > 0.3 else 'low'
    }


def generate_undistortion_example(calib_data):
    """生成畸变矫正示例代码"""

    print("\n" + "="*70)
    print("畸变矫正示例代码")
    print("="*70)

    camera_matrix = calib_data['camera_matrix']
    dist_coeffs = calib_data['distortion_coefficients']
    image_size = calib_data['image_size']

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', default='outputs/calibration/latest/camera_calibration_improved.json',
                       help='相机标定结果JSON文件路径')
    args = parser.parse_args()

    try:
        with open(args.result, 'r') as f:
            calib_data = json.load(f)

        if not calib_data.get('success', False):
            print("❌ 标定结果显示失败，请检查数据")
            return

        analyze_camera_parameters(calib_data)
        generate_undistortion_example(calib_data)

        print("\n" + "="*70)
        print("✅ 分析完成！您现在可以使用这些内参进行:")
        print("   📷 图像畸变矫正")
        print("   📏 3D测量和重建")
        print("   🎯 物体检测和跟踪")
        print("   📐 增强现实应用")
        print("="*70)

    except FileNotFoundError:
        print(f"❌ 找不到标定结果文件: {args.result}")
    except json.JSONDecodeError:
        print(f"❌ 无法解析JSON文件: {args.result}")
    except Exception as exc:
        print(f"❌ 分析过程出错: {exc}")


if __name__ == "__main__":
    main()
