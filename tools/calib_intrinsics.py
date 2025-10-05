# -*- coding: utf-8 -*-
"""
Camera Calibration Script using Modular Code
使用模块化代码进行相机内参标定

功能：
1. 读取 data/raw/calibration/calib_25 文件夹中的所有图像
2. 使用模块化代码检测圆形标定板
3. 提取圆心坐标作为角点
4. 执行 OpenCV 相机标定
5. 输出内参矩阵、畸变系数等

运行方法：
    python tools/calib_intrinsics.py \
        --input data/raw/calibration/calib_25 \
        --output outputs/calibration/basic
"""

import os
import sys
import glob
import argparse
import logging
import cv2
import numpy as np
from typing import List, Tuple
import json

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO)

# 项目路径设置
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

# 导入模块
try:
    from calib.core.calib_core import Calibrator
    from calib.viz.viz_steps import Visualizer
    from calib.utils.board import (
        create_circle_board_object_points,
        extract_image_points_ordered,
    )
    from calib.utils.images import read_image_robust
except ImportError as e:
    raise SystemExit(f"[FATAL] 无法导入模块：{e}")


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def perform_camera_calibration(object_points_list: List[np.ndarray], 
                             image_points_list: List[np.ndarray],
                             image_size: Tuple[int, int]) -> dict:
    """
    执行相机标定
    
    Args:
        object_points_list: 每张图的物理坐标点列表
        image_points_list: 每张图的图像坐标点列表  
        image_size: 图像尺寸 (width, height)
    
    Returns:
        标定结果字典
    """
    logging.info(f"开始相机标定，使用 {len(object_points_list)} 张图像...")
    
    # OpenCV相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list, 
        image_points_list, 
        image_size, 
        None, None
    )
    
    if not ret:
        logging.error("相机标定失败")
        return None
    
    # 计算重投影误差
    total_error = 0
    total_points = 0
    for i in range(len(object_points_list)):
        projected_points, _ = cv2.projectPoints(
            object_points_list[i], rvecs[i], tvecs[i], 
            camera_matrix, dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        error = cv2.norm(image_points_list[i], projected_points, cv2.NORM_L2)
        total_error += error * error
        total_points += len(object_points_list[i])
    
    mean_error = np.sqrt(total_error / total_points)
    
    # 整理结果
    result = {
        'success': True,
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.flatten().tolist(),
        'image_size': image_size,
        'num_images': len(object_points_list),
        'reprojection_error': float(mean_error),
        'focal_length_x': float(camera_matrix[0, 0]),
        'focal_length_y': float(camera_matrix[1, 1]),
        'principal_point_x': float(camera_matrix[0, 2]),
        'principal_point_y': float(camera_matrix[1, 2]),
    }
    
    logging.info(f"标定成功！重投影误差: {mean_error:.4f} 像素")
    logging.info(f"相机矩阵:")
    logging.info(f"  fx: {result['focal_length_x']:.2f}")
    logging.info(f"  fy: {result['focal_length_y']:.2f}")  
    logging.info(f"  cx: {result['principal_point_x']:.2f}")
    logging.info(f"  cy: {result['principal_point_y']:.2f}")
    
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/calibration/calib_25", help="输入图像文件夹")
    ap.add_argument("--output", default="outputs/calibration/basic", help="输出结果文件夹")
    ap.add_argument("--circle_spacing", type=float, default=30.0, 
                    help="圆心间距离(mm)，默认30.0")
    ap.add_argument("--save_viz", action="store_true", 
                    help="保存可视化结果")
    args = ap.parse_args()
    
    # 收集图像文件
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
    
    # 确保输出目录存在
    ensure_dir(args.output)
    
    # 初始化
    calibrator = Calibrator()
    if args.save_viz:
        visualizer = Visualizer(args.output)
    
    # 处理每张图像
    successful_images = []
    object_points_list = []
    image_points_list = []
    image_size = None
    
    for i, image_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        logging.info(f"处理 [{i+1}/{len(image_paths)}] {base_name}...")
        
        # 读取图像
        gray = read_image_robust(image_path)
        if gray is None:
            logging.warning(f"跳过无效图像: {image_path}")
            continue
        
        # 记录图像尺寸
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)
        
        # 使用模块化代码检测标定板
        try:
            result = calibrator.process_gray(gray)
            if result is None or not result.small_numbered:
                logging.warning(f"未检测到标定板: {base_name}")
                continue
            
            # 检查检测到的圆数量
            num_circles = len(result.small_numbered)
            expected_circles = 41  # 7*6 - 1 (中心空位)
            
            if num_circles != expected_circles:
                logging.warning(f"圆数量不正确 {base_name}: 检测到{num_circles}，期望{expected_circles}")
                continue
            
            # 提取图像坐标点
            image_points = extract_image_points_ordered(result)
            if image_points is None or len(image_points) != expected_circles:
                logging.warning(f"提取图像坐标失败: {base_name}")
                continue
            
            # 创建对应的物理坐标点
            object_points = create_circle_board_object_points(
                count=len(image_points), spacing=args.circle_spacing
            )
            
            # 确保点数匹配
            if len(object_points) != len(image_points):
                logging.warning(f"坐标点数量不匹配 {base_name}: "
                              f"物理{len(object_points)} vs 图像{len(image_points)}")
                continue
            
            # 添加到标定数据
            object_points_list.append(object_points)
            image_points_list.append(image_points)
            successful_images.append(base_name)
            
            logging.info(f"✓ {base_name}: 检测到 {num_circles} 个圆")
            
            # 保存可视化结果
            if args.save_viz:
                try:
                    visualizer.save_all(gray, base_name, result)
                except Exception as e:
                    logging.warning(f"可视化保存失败 {base_name}: {e}")
        
        except Exception as e:
            logging.error(f"处理图像失败 {base_name}: {e}")
            continue
    
    # 检查是否有足够的成功图像
    if len(successful_images) < 3:
        logging.error(f"成功图像太少 ({len(successful_images)})，至少需要3张进行标定")
        return
    
    logging.info(f"成功处理 {len(successful_images)} 张图像，开始标定...")
    
    # 执行相机标定
    calib_result = perform_camera_calibration(
        object_points_list, image_points_list, image_size
    )
    
    if calib_result is None:
        logging.error("相机标定失败")
        return
    
    # 保存标定结果
    result_file = os.path.join(args.output, "camera_calibration.json")
    with open(result_file, 'w') as f:
        json.dump(calib_result, f, indent=2)
    
    logging.info(f"标定结果已保存到: {result_file}")
    
    # 打印摘要
    print("\\n" + "="*60)
    print("相机标定结果摘要")
    print("="*60)
    print(f"使用图像数量: {calib_result['num_images']}")
    print(f"图像尺寸: {calib_result['image_size']}")
    print(f"重投影误差: {calib_result['reprojection_error']:.4f} 像素")
    print()
    print("相机内参:")
    print(f"  焦距 fx: {calib_result['focal_length_x']:.2f}")
    print(f"  焦距 fy: {calib_result['focal_length_y']:.2f}")
    print(f"  主点 cx: {calib_result['principal_point_x']:.2f}")
    print(f"  主点 cy: {calib_result['principal_point_y']:.2f}")
    print()
    print(f"相机矩阵:")
    cm = np.array(calib_result['camera_matrix'])
    for row in cm:
        print(f"  [{row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f}]")
    print()
    print(f"畸变系数: {calib_result['distortion_coefficients']}")
    print("="*60)


if __name__ == "__main__":
    main()