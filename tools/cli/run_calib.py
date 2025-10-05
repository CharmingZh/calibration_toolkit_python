# -*- coding: utf-8 -*-
import os, glob, json, argparse
import sys
from typing import List, Tuple
from pathlib import Path
import numpy as np
import cv2
import yaml
from tqdm import tqdm


def find_project_root(start_dir: Path, marker_rel: Path = Path("calib") / "__init__.py") -> Path | None:
    cur = start_dir
    last = None
    while cur != last:
        if (cur / marker_rel).is_file():
            return cur
        last = cur
        cur = cur.parent
    return None


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = find_project_root(_THIS_DIR)
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from calib.core.calib_core import Calibrator
from calib.core.types import BoardResult
from calib.viz.viz_steps import draw_rect_steps, draw_raw_steps

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# === 生成世界坐标（mm）——与你的板一致：7 行 × 6 列，步距 25，跳过 (r=3,c=3) ===
def make_object_points_mm(spacing_mm: float = 25.0) -> np.ndarray:
    pts=[]
    for r in range(7):
        for c in range(6):
            if r==3 and c==3:   # 中心空位
                continue
            X = c*spacing_mm
            Y = r*spacing_mm
            pts.append([X, Y, 0.0])
    return np.asarray(pts, np.float32)  # 41x3

# === 标定：给所有图片的 2D-3D 对应，跑 pinhole 模型 ===
def calibrate_camera(object_points: List[np.ndarray],
                     image_points: List[np.ndarray],
                     image_size: Tuple[int,int]):
    flags = (cv2.CALIB_RATIONAL_MODEL)  # k1..k6 + p1 p2（更稳）
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
        criteria=criteria
    )
    # 每幅图的 reproj error
    per_image_err=[]
    for i,(obj, imgp, r,t) in enumerate(zip(object_points, image_points, rvecs, tvecs)):
        proj,_ = cv2.projectPoints(obj, r, t, K, dist)
        proj = proj.reshape(-1,2)
        err = np.sqrt(np.mean(np.sum((proj - imgp)**2, axis=1)))
        per_image_err.append(float(err))
    return dict(rms=float(ret), K=K, dist=dist, rvecs=rvecs, tvecs=tvecs, per_image_err=per_image_err)

def save_yaml(path, K, dist, image_size):
    data = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": {"rows":3,"cols":3,"data":K.reshape(-1).tolist()},
        "distortion_coefficients": {"rows":1,"cols":len(dist.reshape(-1)),"data":dist.reshape(-1).tolist()},
        "distortion_model": "rational_polynomial",
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/raw/calibration/calib_25")
    ap.add_argument("--out",   default="outputs/calibration/cli")
    ap.add_argument("--spacing_mm", type=float, default=25.0, help="小圆网格步距（mm）")
    ap.add_argument("--save_undist", action="store_true", help="保存去畸变预览")
    args = ap.parse_args()

    ensure_dir(args.out)
    img_paths = sorted(sum([glob.glob(os.path.join(args.indir, ext))
                            for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif")], []))
    if not img_paths:
        print("未找到输入图像"); return

    calib = Calibrator(min_short=1400)
    obj_pts_ref = make_object_points_mm(args.spacing_mm)   # 41x3

    all_obj_pts=[]; all_img_pts=[]
    per_image_json=[]

    for p in tqdm(img_paths, desc="[Calib]"):
        base = os.path.splitext(os.path.basename(p))[0]
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None: 
            print(f"[WARN] 读图失败: {p}"); 
            continue

        br: BoardResult = calib.process_gray(gray)
        if br is None:
            print(f"[FAIL] {base} 未检测到板")
            continue

        # —— 取 41 个小圆（raw 坐标，顺序 0..40）
        img_pts_raw = []
        for d in br.small_numbered_rect:
            X,Y = br.rect_to_raw(d.x, d.y)
            img_pts_raw.append([X, Y])
        img_pts_raw = np.asarray(img_pts_raw, np.float32)

        # 收集标定对应
        all_obj_pts.append(obj_pts_ref.copy())
        all_img_pts.append(img_pts_raw.copy())

        # —— 保存 JSON（复用/调试）
        result_json = {
            "quad": br.quad_raw,
            "homography": br.homography_rect_from_raw,
            "axis_origin_rect": list(br.axis.origin_rect),
            "axis_x_hat": list(br.axis.x_hat),
            "axis_y_hat": list(br.axis.y_hat),
            "small_numbered_raw": img_pts_raw.tolist(),   # 直接给 raw 2D，用于标定
        }
        with open(os.path.join(args.out, f"{base}.json"), "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)

        # —— 步骤图
        rect_gray = np.array(br.rect_gray, dtype=np.uint8)
        rect_img = draw_rect_steps(rect_gray, br)
        raw_img  = draw_raw_steps(gray, br)
        cv2.imwrite(os.path.join(args.out, f"{base}_rect.png"), rect_img)
        cv2.imwrite(os.path.join(args.out, f"{base}_raw.png"),  raw_img)

        per_image_json.append(dict(name=base, n_pts=int(len(img_pts_raw))))

    # === 相机初步标定 ===
    H, W = gray.shape[:2]
    calib_out = calibrate_camera(all_obj_pts, all_img_pts, (W,H))
    K, dist = calib_out["K"], calib_out["dist"]
    rms = calib_out["rms"]; per_err = calib_out["per_image_err"]

    # —— 汇总保存
    summary = {
        "num_images": len(all_img_pts),
        "rms_reproj_error": rms,
        "per_image_error_px": per_err,
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "image_size": [int(W), int(H)],
    }
    with open(os.path.join(args.out, "calib_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    save_yaml(os.path.join(args.out, "calibration.yaml"), K, dist, (W,H))

    # —— 可选：保存去畸变预览
    if args.save_undist:
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (W,H), cv2.CV_16SC2)
        for p in img_paths:
            base = os.path.splitext(os.path.basename(p))[0]
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None: continue
            und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(args.out, f"{base}_undist.png"), und)

    print(f"[OK] 标定完成：RMS={rms:.4f} px，结果已写入 {args.out}")

if __name__ == "__main__":
    main()