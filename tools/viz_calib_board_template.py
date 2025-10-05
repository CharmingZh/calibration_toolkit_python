# -*- coding: utf-8 -*-
"""
Calibration board world coordinates (explicit points; one-folder outputs)

用途 / What this script does
----------------------------
1) 可视化标定板的世界坐标点（小圆 φ≈5mm，按你给的表；大圆 φ≈8mm，按你估计值）。
2) 生成两套图：Matplotlib 矢量/位图（SVG/PNG）、OpenCV 位图（PNG）。
3) 导出点数据为 CSV 与 JSON。
4) ——以上**所有输出**统一保存在同一目录 OUT_DIR 下，避免路径分散或父目录不存在报错。

注意
----
- 未改变功能与数据，仅将输出路径统一到 OUT_DIR，并在保存前确保该目录存在。
- 坐标系仍为“左下角为原点”的世界坐标（单位 mm）。大圆中心不是规则网格点，保持你的估计值不变。
- OpenCV 绘图是可选分支；若未安装 OpenCV，会被 try/except 忽略并给出提示。
"""

from typing import List, Tuple, Literal
import os, math, json
import matplotlib.pyplot as plt
import pandas as pd

# =========== 0) 统一输出目录 ===========
OUT_DIR = "data/templates/calib_board"   # 你可以按需修改；所有输出会集中到这里
os.makedirs(OUT_DIR, exist_ok=True)

Kind = Literal["small", "big"]

# =========== 1) 显式坐标（你的小圆 + 估算的大圆） ===========
# 结构: (X_mm, Y_mm, index, kind, diameter_mm)
POINTS: List[Tuple[float, float, int, Kind, float]] = [
    # ---- 小圆: φ≈5 mm ----
    (0,   0,   0,  "small", 5.0),
    (25,  0,   1,  "small", 5.0),
    (50,  0,   2,  "small", 5.0),
    (75,  0,   3,  "small", 5.0),
    (100, 0,   4,  "small", 5.0),
    (125, 0,   5,  "small", 5.0),

    (0,   25,  6,  "small", 5.0),
    (25,  25,  7,  "small", 5.0),
    (50,  25,  8,  "small", 5.0),
    (75,  25,  9,  "small", 5.0),
    (100, 25,  10, "small", 5.0),
    (125, 25,  11, "small", 5.0),

    (0,   50,  12, "small", 5.0),
    (25,  50,  13, "small", 5.0),
    (50,  50,  14, "small", 5.0),
    (75,  50,  15, "small", 5.0),
    (100, 50,  16, "small", 5.0),
    (125, 50,  17, "small", 5.0),

    (0,   75,  18, "small", 5.0),
    (25,  75,  19, "small", 5.0),
    (50,  75,  20, "small", 5.0),
    # 空缺
    (100, 75,  21, "small", 5.0),
    (125, 75,  22, "small", 5.0),

    (0,   100, 23, "small", 5.0),
    (25,  100, 24, "small", 5.0),
    (50,  100, 25, "small", 5.0),
    (75,  100, 26, "small", 5.0),
    (100, 100, 27, "small", 5.0),
    (125, 100, 28, "small", 5.0),

    (0,   125, 29, "small", 5.0),
    (25,  125, 30, "small", 5.0),
    (50,  125, 31, "small", 5.0),
    (75,  125, 32, "small", 5.0),
    (100, 125, 33, "small", 5.0),
    (125, 125, 34, "small", 5.0),

    (0,   150, 35, "small", 5.0),
    (25,  150, 36, "small", 5.0),
    (50,  150, 37, "small", 5.0),
    (75,  150, 38, "small", 5.0),
    (100, 150, 39, "small", 5.0),
    (125, 150, 40, "small", 5.0),

    # ---- 大圆: φ≈8 mm ----
    (70.6, 61.8, 41, "big", 8.0),
    (58.4, 61.8, 42, "big", 8.0),
    (70.6, 74.6, 43, "big", 8.0),
    (58.4, 86.3, 44, "big", 8.0),
]

# =========== 2) 可视化（Matplotlib） ===========
def plot_world_points(
    points: List[Tuple[float, float, int, Kind, float]],
    out_png: str = None,
    out_svg: str = None,
    annotate: bool = True
):
    # 统一输出文件名（若未指定）
    if out_png is None:
        out_png = os.path.join(OUT_DIR, "board_points.png")
    if out_svg is None:
        out_svg = os.path.join(OUT_DIR, "board_points.svg")

    # 确保目录存在
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    pad = 12.5
    X0, X1 = x_min - pad, x_max + pad
    Y0, Y1 = y_min - pad, y_max + pad

    fig = plt.figure(figsize=(8, 6), dpi=130)
    ax  = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Calibration Board (small=5 mm, big=8 mm)")

    # 网格（主50、次25）
    def ticks(a0, a1, step):
        s = int(math.floor(a0/step)*step)
        e = int(math.ceil(a1/step)*step)
        return list(range(s, e+step, step))
    ax.set_xticks(ticks(X0, X1, 50))
    ax.set_yticks(ticks(Y0, Y1, 50))
    ax.set_xticks(ticks(X0, X1, 25), minor=True)
    ax.set_yticks(ticks(Y0, Y1, 25), minor=True)
    ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.35)

    # marker 尺度（视觉区分，不按真实尺寸画比例）
    def msize(d_mm: float) -> float:
        return 28 if d_mm >= 7.0 else 18

    # 绘制
    for (x, y, idx, kind, d) in points:
        if kind == "big":
            ax.scatter([x], [y], marker="s", s=msize(d), facecolors="none",
                       edgecolors="k", linewidths=1.2, zorder=3)
        else:
            ax.scatter([x], [y], marker="o", s=msize(d), facecolors="none",
                       edgecolors="k", linewidths=0.9, zorder=2)
        if annotate:
            ax.annotate(str(idx), (x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=9)

    # 原点参考线
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="k", linewidth=0.8, alpha=0.5)

    # 图例
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker='o', linestyle='None', mfc='none', mec='k', label='small (5 mm)'),
        Line2D([0],[0], marker='s', linestyle='None', mfc='none', mec='k', label='big (8 mm)'),
    ]
    ax.legend(handles=handles, loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_svg)
    print(f"[OK] 保存: {os.path.abspath(out_png)} / {os.path.abspath(out_svg)}")
    plt.close(fig)

# =========== 3) (可选) OpenCV 输出 ===========
def draw_with_opencv(points, out_png=None, scale_px_per_mm=2.0):
    # 统一输出文件名（若未指定）
    if out_png is None:
        out_png = os.path.join(OUT_DIR, "board_points_cv.png")

    # 确保目录存在
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    import numpy as np, cv2
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    pad = 12.5
    X0, X1 = min(xs)-pad, max(xs)+pad
    Y0, Y1 = min(ys)-pad, max(ys)+pad
    W = int((X1 - X0) * scale_px_per_mm) + 1
    H = int((Y1 - Y0) * scale_px_per_mm) + 1
    img = np.full((H, W, 3), 255, np.uint8)

    def w2p(x, y):
        u = int((x - X0) * scale_px_per_mm)
        v = int((Y1 - y) * scale_px_per_mm)
        return u, v

    # 网格
    def grid(step, color, thick):
        x = math.ceil(X0/step)*step
        while x <= X1+1e-6:
            u0,v0 = w2p(x, Y0); u1,v1 = w2p(x, Y1)
            cv2.line(img,(u0,v0),(u1,v1),color,thick,cv2.LINE_AA)
            x += step
        y = math.ceil(Y0/step)*step
        while y <= Y1+1e-6:
            u0,v0 = w2p(X0, y); u1,v1 = w2p(X1, y)
            cv2.line(img,(u0,v0),(u1,v1),color,thick,cv2.LINE_AA)
            y += step
    grid(50,(200,200,200),1); grid(25,(230,230,230),1)

    # 点
    for (x,y,idx,kind,d) in points:
        u,v = w2p(x,y)
        if kind == "big":
            # 用小矩形表示大圆（视觉区分用；不改变功能）
            import cv2
            cv2.rectangle(img,(u-4,v-4),(u+4,v+4),(0,0,0),1,cv2.LINE_AA)
        else:
            cv2.circle(img,(u,v),4,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(img,str(idx),(u+6,v-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1,cv2.LINE_AA)

    import cv2
    cv2.imwrite(out_png, img)
    print(f"[OK] 保存: {os.path.abspath(out_png)}")

# =========== 4) 导出（给标定脚本） ===========
def export_points(points, out_dir=None):
    # 统一导出到 OUT_DIR
    if out_dir is None:
        out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(points, columns=["X_mm","Y_mm","index","kind","diameter_mm"])
    csv_p  = os.path.join(out_dir, "board_points.csv")
    json_p = os.path.join(out_dir, "board_points.json")

    df.to_csv(csv_p, index=False)
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    print(f"[OK] 导出: {os.path.abspath(csv_p)}")
    print(f"[OK] 导出: {os.path.abspath(json_p)}")

# =========== 5) 运行 ===========
if __name__ == "__main__":
    plot_world_points(POINTS, annotate=True)
    try:
        draw_with_opencv(POINTS)
    except Exception as e:
        print(f"[WARN] OpenCV 绘制失败（可忽略）：{e}")
    export_points(POINTS)