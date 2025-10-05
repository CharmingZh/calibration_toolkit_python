#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate photometrically and geometrically diverse calibration images."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_exposure_gamma(img: np.ndarray, exposure: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    imgf = img.astype(np.float32) * float(exposure)
    if gamma <= 0:
        return to_uint8(imgf)
    inv = 1.0 / float(gamma)
    table = (np.arange(256, dtype=np.float32) / 255.0) ** inv * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(to_uint8(imgf), table)


def kelvin_to_gains(kelvin: float) -> np.ndarray:
    temp = float(np.clip(kelvin, 2000, 12000)) / 100.0
    if temp <= 66:
        red = 255.0
    else:
        red = 329.698727446 * ((temp - 60) ** -0.1332047592)
    if temp <= 66:
        green = 99.4708025861 * np.log(temp) - 161.1195681661
    else:
        green = 288.1221695283 * ((temp - 60) ** -0.0755148492)
    if temp >= 66:
        blue = 255.0
    elif temp <= 19:
        blue = 0.0
    else:
        blue = 138.5177312231 * np.log(temp - 10) - 305.0447927307

    gains = np.array([blue, green, red], dtype=np.float32)
    gains /= gains.mean() if gains.mean() != 0 else 1.0
    return gains


def apply_color_temp(img: np.ndarray, kelvin: float = 6500.0) -> np.ndarray:
    gains = kelvin_to_gains(kelvin).reshape(1, 1, 3)
    return to_uint8(img.astype(np.float32) * gains)


def add_poisson_gaussian_noise(img: np.ndarray, sigma: float = 3.0, poisson_scale: float = 0.0) -> np.ndarray:
    imgf = img.astype(np.float32)
    if poisson_scale > 0:
        x = np.clip(imgf / 255.0, 0, 1)
        lam = np.clip(x * poisson_scale, 0, None)
        x_noisy = np.random.poisson(lam) / max(poisson_scale, 1e-6)
        imgf = x_noisy * 255.0
    if sigma > 0:
        imgf += np.random.normal(0, sigma, img.shape).astype(np.float32)
    return to_uint8(imgf)


def add_vignette(img: np.ndarray, strength: float = 0.12) -> np.ndarray:
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
    mask = (1 - strength * r).clip(0, 1).astype(np.float32)
    mask = cv2.merge([mask, mask, mask])
    return to_uint8(img.astype(np.float32) * mask)


def add_flicker_banding(img: np.ndarray, amp: float = 12.0, period_px: int = 32, axis: str = "h") -> np.ndarray:
    if amp <= 0:
        return img
    h, w = img.shape[:2]
    if axis == "h":
        coords = np.arange(h, dtype=np.float32).reshape(h, 1)
        modulation = 1 + (amp / 255.0) * np.sin(2 * np.pi * coords / period_px)
        mask = np.repeat(modulation, w, axis=1)
    else:
        coords = np.arange(w, dtype=np.float32).reshape(1, w)
        modulation = 1 + (amp / 255.0) * np.sin(2 * np.pi * coords / period_px)
        mask = np.repeat(modulation, h, axis=0)
    mask = np.expand_dims(mask, axis=2)
    return to_uint8(img.astype(np.float32) * mask)


def reencode_jpeg(img: np.ndarray, quality: int = 80) -> np.ndarray:
    success, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 10, 100))])
    if not success:
        return img
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else img


def motion_blur(img: np.ndarray, ksize: int = 9, angle_deg: float = 0.0) -> np.ndarray:
    if ksize <= 1:
        return img
    kernel = np.zeros((ksize, ksize), np.float32)
    kernel[ksize // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (ksize, ksize))
    kernel /= kernel.sum() if kernel.sum() != 0 else 1.0
    return cv2.filter2D(img, -1, kernel)


def defocus_blur(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 1:
        return img
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def resize_keep_ratio(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp)


def small_pose_homography(
    img: np.ndarray,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    fov_pad: float = 0.05,
) -> np.ndarray:
    h, w = img.shape[:2]
    cvt = cv2.getRotationMatrix2D((w / 2, h / 2), roll_deg, 1.0)
    rotated = cv2.warpAffine(img, cvt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    dx = (w * 0.15) * np.tanh(yaw)
    dy = (h * 0.15) * np.tanh(pitch)
    padw = int(w * fov_pad)
    padh = int(h * fov_pad)

    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [0 + dx + padw, 0 + dy + padh],
            [w - 1 - dx - padw, 0 + dy + padh],
            [w - 1 + dx - padw, h - 1 - dy - padh],
            [0 - dx + padw, h - 1 - dy - padh],
        ]
    )
    homography = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rotated, homography, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_lens_distortion(
    img: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> np.ndarray:
    h, w = img.shape[:2]
    fx = 0.8 * w
    fy = 0.8 * h
    cx = w / 2
    cy = h / 2
    x = (np.arange(w) - cx) / fx
    y = (np.arange(h) - cy) / fy
    xv, yv = np.meshgrid(x, y)
    r2 = xv * xv + yv * yv
    x_distort = xv * (1 + k1 * r2 + k2 * r2 * r2) + 2 * p1 * xv * yv + p2 * (r2 + 2 * xv * xv)
    y_distort = yv * (1 + k1 * r2 + k2 * r2 * r2) + p1 * (r2 + 2 * yv * yv) + 2 * p2 * xv * yv
    map_x = (x_distort * fx + cx).astype(np.float32)
    map_y = (y_distort * fy + cy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------------------
# Scenario definitions
# --------------------------------------------------------------------------------------

SCENARIOS = [
    (
        "lab_nominal",
        dict(exposure=1.0, gamma=1.0, kelvin=6500, vign=0.06, noise_sigma=2, noise_poi=0),
        dict(motion_k=0, motion_ang=0, defocus_k=0),
        dict(
            scale=1.0,
            yaw=0,
            pitch=0,
            roll=0,
            k1=0.0,
            k2=0.0,
            p1=0.0,
            p2=0.0,
            flicker_amp=0,
            flicker_per=0,
            flicker_axis="h",
        ),
        dict(jpeg_q=95),
    ),
    (
        "overexp_half_stop",
        dict(exposure=1.41, gamma=0.95, kelvin=6500, vign=0.05, noise_sigma=2, noise_poi=0),
        dict(motion_k=0, motion_ang=0, defocus_k=0),
        dict(
            scale=1.0,
            yaw=2,
            pitch=1,
            roll=2,
            k1=0.01,
            k2=0.0,
            p1=0.0,
            p2=0.0,
            flicker_amp=0,
            flicker_per=0,
            flicker_axis="h",
        ),
        dict(jpeg_q=92),
    ),
    (
        "underexp_1stop_dark_warehouse",
        dict(exposure=0.5, gamma=1.15, kelvin=6500, vign=0.12, noise_sigma=6, noise_poi=10),
        dict(motion_k=0, motion_ang=0, defocus_k=0),
        dict(
            scale=1.0,
            yaw=0,
            pitch=2,
            roll=0,
            k1=-0.02,
            k2=0.0,
            p1=0.0,
            p2=0.0,
            flicker_amp=10,
            flicker_per=28,
            flicker_axis="h",
        ),
        dict(jpeg_q=85),
    ),
    (
        "warm_led_3500k",
        dict(exposure=1.0, gamma=1.0, kelvin=3500, vign=0.08, noise_sigma=3, noise_poi=0),
        dict(motion_k=0, motion_ang=0, defocus_k=0),
        dict(
            scale=1.0,
            yaw=0,
            pitch=1,
            roll=2,
            k1=0.015,
            k2=0.0,
            p1=0.001,
            p2=-0.001,
            flicker_amp=8,
            flicker_per=36,
            flicker_axis="h",
        ),
        dict(jpeg_q=90),
    ),
    (
        "cool_daylight_7500k",
        dict(exposure=1.0, gamma=1.0, kelvin=7500, vign=0.07, noise_sigma=3, noise_poi=0),
        dict(motion_k=0, motion_ang=0, defocus_k=0),
        dict(
            scale=1.0,
            yaw=-3,
            pitch=2,
            roll=-2,
            k1=-0.015,
            k2=0.0,
            p1=-0.001,
            p2=0.001,
            flicker_amp=5,
            flicker_per=40,
            flicker_axis="h",
        ),
        dict(jpeg_q=90),
    ),
    (
        "conveyor_motion_blur",
        dict(exposure=1.0, gamma=1.0, kelvin=6500, vign=0.06, noise_sigma=2, noise_poi=0),
        dict(motion_k=9, motion_ang=0, defocus_k=0),
        dict(
            scale=1.0,
            yaw=0,
            pitch=1,
            roll=0,
            k1=0.01,
            k2=0.0,
            p1=0.0,
            p2=0.0,
            flicker_amp=0,
            flicker_per=0,
            flicker_axis="h",
        ),
        dict(jpeg_q=90),
    ),
    (
        "slight_defocus",
        dict(exposure=1.0, gamma=1.0, kelvin=6500, vign=0.06, noise_sigma=2, noise_poi=0),
        dict(motion_k=0, motion_ang=0, defocus_k=5),
        dict(
            scale=1.0,
            yaw=0,
            pitch=0,
            roll=0,
            k1=0.012,
            k2=0.0,
            p1=0.0,
            p2=0.0,
            flicker_amp=0,
            flicker_per=0,
            flicker_axis="h",
        ),
        dict(jpeg_q=95),
    ),
    (
        "pose_tilt_small",
        dict(exposure=1.0, gamma=1.0, kelvin=6500, vign=0.06, noise_sigma=2, noise_poi=0),
        dict(motion_k=0, motion_ang=0, defocus_k=0),
        dict(
            scale=1.0,
            yaw=6,
            pitch=-6,
            roll=5,
            k1=0.01,
            k2=0.0,
            p1=0.0,
            p2=0.0,
            flicker_amp=0,
            flicker_per=0,
            flicker_axis="v",
        ),
        dict(jpeg_q=92),
    ),
    (
        "pose_tilt_medium",
        dict(exposure=1.0, gamma=1.0, kelvin=6500, vign=0.06, noise_sigma=3, noise_poi=0),
        dict(motion_k=0, motion_ang=0, defocus_k=0),
        dict(
            scale=1.0,
            yaw=10,
            pitch=10,
            roll=-8,
            k1=-0.02,
            k2=0.0,
            p1=0.001,
            p2=0.001,
            flicker_amp=0,
            flicker_per=0,
            flicker_axis="v",
        ),
        dict(jpeg_q=90),
    ),
]

RESOLUTIONS = [0.6, 1.0, 1.4]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dng", ".DNG"}


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def collect_inputs(input_dir: Path) -> List[Path]:
    return [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix in IMAGE_EXTS]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成多场景真实风格的数据增强集")
    parser.add_argument("--input-dir", default="data/raw/test_samples", help="原始输入图像目录")
    parser.add_argument("--output-dir", default="data/augmented/realistic", help="增强图像输出目录")
    parser.add_argument("--manifest", default=None, help="manifest CSV 文件路径（默认写入输出目录）")
    args = parser.parse_args()

    random.seed(2025)
    np.random.seed(2025)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"输入目录不存在: {input_dir}")

    input_files = collect_inputs(input_dir)
    if not input_files:
        raise SystemExit(f"输入目录中未找到图像文件: {input_dir}")

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.csv"
    ensure_dir(manifest_path.parent)

    total = len(input_files) * len(SCENARIOS) * len(RESOLUTIONS)
    counter = 1

    with open(manifest_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "idx",
                "src",
                "scenario",
                "scale",
                "exposure",
                "gamma",
                "kelvin",
                "vignette",
                "noise_sigma",
                "noise_poisson",
                "flicker_amp",
                "flicker_period",
                "flicker_axis",
                "yaw",
                "pitch",
                "roll",
                "k1",
                "k2",
                "p1",
                "p2",
                "motion_k",
                "motion_ang",
                "defocus_k",
                "jpeg_q",
                "width",
                "height",
                "outfile",
            ]
        )

        with tqdm(total=total, desc="生成真实风格增强集") as progress:
            for src in input_files:
                image = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"[跳过] 无法读取 {src}")
                    progress.update(len(SCENARIOS) * len(RESOLUTIONS))
                    continue

                for scenario_name, phot, blur, geom, comp in SCENARIOS:
                    for scale in RESOLUTIONS:
                        augmented = image.copy()
                        augmented = resize_keep_ratio(augmented, scale)
                        augmented = apply_exposure_gamma(augmented, phot["exposure"], phot["gamma"])
                        augmented = apply_color_temp(augmented, phot["kelvin"])
                        augmented = add_poisson_gaussian_noise(
                            augmented, sigma=phot["noise_sigma"], poisson_scale=phot["noise_poi"]
                        )
                        if phot["vign"] > 0:
                            augmented = add_vignette(augmented, strength=phot["vign"])
                        if geom["flicker_amp"] > 0:
                            augmented = add_flicker_banding(
                                augmented,
                                amp=geom["flicker_amp"],
                                period_px=max(geom["flicker_per"], 16),
                                axis=geom["flicker_axis"],
                            )

                        augmented = small_pose_homography(
                            augmented,
                            yaw_deg=geom["yaw"],
                            pitch_deg=geom["pitch"],
                            roll_deg=geom["roll"],
                        )
                        augmented = apply_lens_distortion(
                            augmented,
                            k1=geom["k1"],
                            k2=geom["k2"],
                            p1=geom["p1"],
                            p2=geom["p2"],
                        )

                        if blur["motion_k"] > 1:
                            augmented = motion_blur(augmented, ksize=blur["motion_k"], angle_deg=blur["motion_ang"])
                        if blur["defocus_k"] > 1:
                            augmented = defocus_blur(augmented, kernel_size=blur["defocus_k"])

                        augmented = reencode_jpeg(augmented, quality=comp["jpeg_q"])

                        outfile = out_dir / f"{counter:04d}.png"
                        cv2.imwrite(str(outfile), augmented)
                        h, w = augmented.shape[:2]

                        writer.writerow(
                            [
                                counter,
                                str(src),
                                scenario_name,
                                scale,
                                phot["exposure"],
                                phot["gamma"],
                                phot["kelvin"],
                                phot["vign"],
                                phot["noise_sigma"],
                                phot["noise_poi"],
                                geom["flicker_amp"],
                                geom["flicker_per"],
                                geom["flicker_axis"],
                                geom["yaw"],
                                geom["pitch"],
                                geom["roll"],
                                geom["k1"],
                                geom["k2"],
                                geom["p1"],
                                geom["p2"],
                                blur["motion_k"],
                                blur["motion_ang"],
                                blur["defocus_k"],
                                comp["jpeg_q"],
                                w,
                                h,
                                str(outfile),
                            ]
                        )
                        counter += 1
                        progress.update(1)

    print(f"完成：输出目录 = {out_dir}, 共 {counter - 1} 张；参数清单 = {manifest_path}")


if __name__ == "__main__":
    main()
