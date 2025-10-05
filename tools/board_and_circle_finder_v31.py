# -*- coding: utf-8 -*-
"""
BoardAndCircleFinder v3.1 (Axis-Locked Numbering & Fancy Row Lines)

变化摘要：
- 用 4 个大圆稳健估计坐标轴（只允许旋转，不会镜像）：
  * 找出“远离另外三个”的点记为 TL
  * 其余三个里找近似直角点记为 BR
  * 剩下两点与 BR 组成两条正交轴（红=x, 绿=y），若叉积<0则交换保证右手系
- 小圆编号：将小圆投影到 (x̂, ŷ)，v 方向 KMeans=7 聚类得 7 行，行内按 u 排序；
  * 第 4 行（r==3）跳过 col=3（中心空位），得到 0..40 稳定编号（行优先）
- 可视化：
  * 在 rect/raw 上以大号字体标注编号
  * 每一行用彩虹渐变连线；并用半透明虚线把行与行“尾接头”，更连贯
  * 用大圆画坐标轴箭头（红=x，绿=y）
- 其余检测/精修逻辑沿用 v2.8；修正 np.eye 调用方式

依赖：
  pip install opencv-python numpy tqdm
"""

import os, glob, argparse, logging, json
import numpy as np
import cv2
from itertools import combinations
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO)

# --------------------- Utils ---------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def to_bgr(g): return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def order_quad(pts4):
    pts4 = np.asarray(pts4, np.float32)
    s = pts4.sum(1); d = np.diff(pts4,axis=1).reshape(-1)
    tl = pts4[np.argmin(s)]; br = pts4[np.argmax(s)]
    tr = pts4[np.argmin(d)]; bl = pts4[np.argmax(d)]
    return np.array([tl,tr,br,bl], np.float32)

def warp_by_quad(gray, quad, min_short=1400):
    W = int(max(400, np.linalg.norm(quad[1]-quad[0])))
    H = int(max(400, np.linalg.norm(quad[3]-quad[0])))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
    Hm = cv2.getPerspectiveTransform(quad, dst)
    rect = cv2.warpPerspective(gray, Hm, (W, H))
    short = min(W,H)
    if short < min_short:
        s = float(min_short)/short
        rect = cv2.resize(rect, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        Hm = np.array([[s,0,0],[0,s,0],[0,0,1]], np.float32) @ Hm
    return rect, Hm

def H_inv_point(H, x, y):
    p = np.array([x,y,1.0], np.float32)
    q = np.linalg.inv(H) @ p
    return (float(q[0]/q[2]), float(q[1]/q[2]))

def annotate_text(img, text, xy, color=(0,0,255), font_scale=0.95, thick=2):
    x,y = int(round(xy[0])), int(round(xy[1]))
    cv2.putText(img, str(text), (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)

# --------------------- Hough helpers (找板) ---------------------
def seg_to_normal_form(x1,y1,x2,y2):
    vx, vy = x2-x1, y2-y1
    L = np.hypot(vx,vy) + 1e-9
    nx, ny = -vy/L, vx/L
    c = -(nx*x1 + ny*y1)
    rho   = -c
    theta = (np.arctan2(ny, nx) + np.pi) % np.pi
    return (nx, ny, c, theta, rho)

def intersect(Lm, Ln):
    a1,b1,c1,_,_ = Lm; a2,b2,c2,_,_ = Ln
    D = a1*b2 - a2*b1
    if abs(D) < 1e-8: return None
    x = (b1*c2 - b2*c1)/D
    y = (c1*a2 - c2*b1)/D
    return np.array([x,y], np.float32)

def line_angle_deg(L): return (np.degrees(L[3]) - 90.0) % 180.0
def deg_diff(a,b): return abs((a-b+90) % 180 - 90)

def bilinear(im, xy):
    x=xy[:,0]; y=xy[:,1]
    x0=np.clip(np.floor(x).astype(np.int32),0,im.shape[1]-2)
    y0=np.clip(np.floor(y).astype(np.int32),0,im.shape[0]-2)
    dx=(x-x0); dy=(y-y0)
    return (1-dx)*(1-dy)*im[y0,x0] + dx*(1-dy)*im[y0,x0+1] + (1-dx)*dy*im[y0+1,x0] + dx*dy*im[y0+1,x0+1]

def edge_contrast(gray, p, q, half=6, ns=48):
    v = q - p
    L = np.linalg.norm(v) + 1e-9
    t = np.linspace(0,1,ns)[:,None]
    pts = p[None,:] + t*v[None,:]
    n = np.array([-v[1], v[0]], np.float32)/L
    p_in  = pts - n*half
    p_out = pts + n*half
    return float(np.mean(bilinear(gray, p_in) - bilinear(gray, p_out)))

def quad_score(gray, quad):
    h,w = gray.shape
    margin = 0.03
    if np.any(quad[:,0] < margin*w) or np.any(quad[:,0] > (1-margin)*w) \
       or np.any(quad[:,1] < margin*h) or np.any(quad[:,1] > (1-margin)*h):
        return False, -1e9
    area = cv2.contourArea(quad.astype(np.float32))
    if area < 0.15*h*w or area > 0.80*h*w: return False, -1e9
    wlen = 0.5*(np.linalg.norm(quad[1]-quad[0])+np.linalg.norm(quad[2]-quad[3]))
    hlen = 0.5*(np.linalg.norm(quad[3]-quad[0])+np.linalg.norm(quad[2]-quad[1]))
    ratio = max(wlen,hlen)/max(1.0,min(wlen,hlen))
    if not (0.90 <= ratio <= 1.35): return False, -1e9
    csum = 0.0
    for i in range(4):
        v = edge_contrast(gray, quad[i], quad[(i+1)%4], half=6, ns=48)
        if v < 0.5: return False, -1e9
        csum += v
    return True, csum + 300.0*(area/(h*w))

def detect_by_hough_search(gray):
    h,w = gray.shape
    g = cv2.GaussianBlur(gray, (0,0), 1.0)
    med = np.median(g)
    lo = int(max(10, 0.66*med)); hi = lo*2
    edges = cv2.Canny(g, lo, hi)
    edges = cv2.dilate(edges, np.ones((3,3),np.uint8), 1)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, int(0.006*min(h,w)),
                            minLineLength=int(0.30*min(h,w)), maxLineGap=int(0.03*min(h,w)))
    if lines is None or len(lines)<4: return None, edges
    Ls = [seg_to_normal_form(*xy) for xy in lines.reshape(-1,4)]

    feats = np.array([[np.cos(2*L[3]), np.sin(2*L[3])] for L in Ls], np.float32)
    crit  = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _, labels, _ = cv2.kmeans(feats, 2, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    groups = [[],[]]
    for L, lb in zip(Ls, labels.ravel()):
        groups[lb].append(L)
    if len(groups[0])<2 or len(groups[1])<2: return None, edges

    def nms_1d(lines, rho_thr):
        kept=[]
        for L in sorted(lines,key=lambda t:t[4]):
            if all(abs(L[4]-K[4])>rho_thr for K in kept): kept.append(L)
        return kept
    G0 = nms_1d(groups[0],0.02*max(h,w))
    G1 = nms_1d(groups[1],0.02*max(h,w))

    best=None; best_score=-1e9
    for L0a,L0b in combinations(G0,2):
        if deg_diff(line_angle_deg(L0a),line_angle_deg(L0b))>5: continue
        for L1a,L1b in combinations(G1,2):
            if deg_diff(line_angle_deg(L1a),line_angle_deg(L1b))>5: continue
            if abs(deg_diff(line_angle_deg(L0a),line_angle_deg(L1a))-90)>10: continue
            pts=[intersect(L0a,L1a),intersect(L0b,L1a),intersect(L0b,L1b),intersect(L0a,L1b)]
            if any(p is None for p in pts): continue
            quad=order_quad(np.stack(pts,0))
            ok, sc=quad_score(gray, quad)
            if ok and sc>best_score:
                best_score=sc; best=quad
    return best, edges

def detect_by_white_region(gray):
    g = cv2.GaussianBlur(gray,(0,0),1.2)
    _, th = cv2.threshold(g, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), 1)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if num<=1: return None, th
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.zeros_like(th); mask[lab==idx] = 255
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, mask
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    quad = cv2.boxPoints(rect).astype(np.float32)
    return order_quad(quad), mask

# --------------------- Blob（覆盖小/大圆） ---------------------
def make_blob_detector(min_area=500.0, max_area=15000.0, dark_blobs=True, min_circ=0.50, min_conv=0.50):
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = 5; p.maxThreshold = 220; p.thresholdStep = 5
    p.filterByArea = True; p.minArea = float(min_area); p.maxArea = float(max_area)
    p.filterByCircularity = True; p.minCircularity = float(min_circ)
    p.filterByInertia = True; p.minInertiaRatio = 0.04
    p.filterByConvexity = True; p.minConvexity = float(min_conv)
    p.filterByColor = True; p.blobColor = 0 if dark_blobs else 255
    p.minDistBetweenBlobs = 10.0
    return cv2.SimpleBlobDetector_create(p)

# --------------------- 组件分割（精细化的前置） ---------------------
def segment_component(patch, seed, ksize=3):
    g = cv2.GaussianBlur(patch,(ksize,ksize),0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th = cv2.medianBlur(th, 3)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    H,W = th.shape
    sx, sy = int(round(seed[0])), int(round(seed[1]))
    sx = np.clip(sx,0,W-1); sy = np.clip(sy,0,H-1)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if num <= 1:
        return np.zeros_like(th), []
    lbl = lab[sy, sx]
    if lbl == 0:
        ys, xs = np.nonzero(th)
        if len(xs)==0: return np.zeros_like(th), []
        d = (xs - sx)**2 + (ys - sy)**2
        j = int(np.argmin(d))
        lbl = lab[ys[j], xs[j]]
        if lbl == 0:
            return np.zeros_like(th), []

    mask = np.zeros_like(th); mask[lab==lbl]=255
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return mask, []
    cnt = max(cnts, key=cv2.contourArea)
    return mask, cnt

# --------------------- 多源中心估计 ---------------------
def center_by_moments(cnt):
    m = cv2.moments(cnt)
    if abs(m["m00"]) < 1e-6: return None
    return (float(m["m10"]/m["m00"]), float(m["m01"]/m["m00"]))

def center_by_ellipse(cnt):
    if len(cnt) < 5: return None, None
    ell = cv2.fitEllipse(cnt)
    (cx,cy),(ma,Mi),ang = ell
    return (float(cx),float(cy)), ((float(ma),float(Mi)), float(ang), ell)

def center_by_dt_peak(mask, approx=(3,3), win=9):
    if mask.max()==0: return None
    inv = (mask>0).astype(np.uint8)*255
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    y,x = np.unravel_index(np.argmax(dt), dt.shape)
    c0 = np.array([[float(x), float(y)]], np.float32)
    dt8 = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    term=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    try:
        cv2.cornerSubPix(dt8, c0, approx, (-1,-1), term)
    except cv2.error:
        pass
    return (float(c0[0,0]), float(c0[0,1]))

def average_radius_to_edge(center, cnt):
    cx,cy = center
    pts = cnt.reshape(-1,2).astype(np.float32)
    rr = np.hypot(pts[:,0]-cx, pts[:,1]-cy)
    return float(np.median(rr))

def fuse_centers(patch, mask, cnt, cand, ell_c, ell_info, dt_c):
    scores=[]; cands=[]; tags=[]
    if cand is not None:
        cands.append(cand); tags.append("seed"); scores.append(0.5)
    if cnt is not None:
        m_c = center_by_moments(cnt)
        if m_c is not None:
            cands.append(m_c); tags.append("moments"); scores.append(0.9)
    if ell_c is not None and ell_info is not None:
        (ma,Mi), ang, ell = ell_info
        ar = min(ma,Mi)/max(ma,Mi) if max(ma,Mi)>1e-6 else 0
        area_ok = (np.pi*0.25*ma*Mi) > 50
        qual = 0.6 + 0.4*ar if area_ok else 0.3*ar
        cands.append(ell_c); tags.append("ellipse"); scores.append(float(qual))
    if dt_c is not None:
        cands.append(dt_c); tags.append("dt"); scores.append(0.95)

    Cs = np.array(cands, np.float32)
    w = np.array(scores, np.float32)
    if len(Cs)==1: return cands[0], tags[0], 1.0

    w /= (w.sum()+1e-9)
    mu = (Cs*w[:,None]).sum(0)
    d = np.linalg.norm(Cs - mu[None,:], axis=1)
    sigma = (np.median(d)+1e-6)
    damp = np.exp(-(d/(3*sigma))**2)
    w2 = w * damp
    w2 /= (w2.sum()+1e-9)
    fused = (Cs*w2[:,None]).sum(0)
    conf = float(w2.max())
    tag = tags[int(np.argmax(w2))]
    return (float(fused[0]), float(fused[1])), tag, conf

# --------------------- 数学兜底（圆最小二乘） ---------------------
def circle_fit_least_squares(xy):
    x = xy[:,0]; y = xy[:,1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x*x + y*y
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, d = c
    r = np.sqrt(max(1e-6, cx*cx + cy*cy + d))
    return float(cx), float(cy), float(r)

# --------------------- 主精修：对单个候选圆 ---------------------
def refine_center_on_patch(rect, c0, r0, gate=0.6):
    h,w = rect.shape
    x0,y0 = c0
    win = int(np.clip(3.0*r0, 30, 220))
    x1 = max(0, int(x0-win)); x2 = min(w, int(x0+win))
    y1 = max(0, int(y0-win)); y2 = min(h, int(y0+win))
    patch = rect[y1:y2, x1:x2]
    if patch.size==0:
        return (x0,y0), r0, "seed", None, None, None

    seed_local = (x0 - x1, y0 - y1)

    mask, cnt = segment_component(patch, seed_local, ksize=3)
    if len(cnt)==0:
        edges = cv2.Canny(cv2.GaussianBlur(patch,(0,0),1.0), 30, 90)
        ys,xs = np.nonzero(edges)
        if len(xs)<15:
            return (x0,y0), r0, "seed", None, None, None
        pts = np.stack([xs,ys],1).astype(np.float32)
        cx,cy,r = circle_fit_least_squares(pts)
        return (x1+cx, y1+cy), r, "lsq_circle", pts, None, None

    ell_c, ell_info = center_by_ellipse(cnt)
    mom_c = center_by_moments(cnt)
    dt_c  = center_by_dt_peak(mask)

    fused_c, tag, conf = fuse_centers(patch, mask, cnt, seed_local, ell_c, ell_info, dt_c)
    r_est = average_radius_to_edge(fused_c, cnt) if len(cnt)>0 else r0

    cx,cy = fused_c
    cx += x1; cy += y1

    ell_draw = None
    if ell_info is not None and ell_c is not None:
        (ma,Mi), ang, ell = ell_info
        ell_draw = ((x1+ell_c[0], y1+ell_c[1]), (ma,Mi), ang)
    area = float(cv2.contourArea(cnt)) if isinstance(cnt, np.ndarray) and len(cnt)>0 else float(np.pi*r_est*r_est)

    return (cx,cy), r_est, tag, cnt + np.array([x1,y1]), ell_draw, dict(seed=(x0,y0), moments=mom_c, dt=dt_c, conf=conf, area=area)

# --------------------- KMeans helpers ---------------------
def kmeans_1d_sizes(sizes):
    v = np.array(sizes, np.float32)[:,None]
    crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,1e-4)
    _, labels, centers = cv2.kmeans(v, 2, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    if centers[0] > centers[1]:
        labels = 1 - labels
        centers = centers[::-1]
    return labels.ravel(), centers.ravel()

# --------------------- 方向估计（由 4 个大圆） ---------------------
def axes_from_big4(points_4):
    """
    输入：4 个大圆的 (x,y)
    返回：原点 O=BR，单位轴向量 x_hat, y_hat, 以及 TL,TR,BR,BL 的索引
    逻辑：
      - 求每点到其他三点距离之和，和最大者记为 TL（离群）
      - 其余三点中，夹角最接近 90° 的顶点记为 BR
      - 剩下两点与 BR 构成两条轴；通过交换保证右手系（cross>0）
      - x 轴指向“TR”，y 轴指向“BL”
    """
    P = np.asarray(points_4, np.float32)  # (4,2)
    D = np.linalg.norm(P[:,None,:]-P[None,:,:], axis=2)
    s = D.sum(1)
    i_tl = int(np.argmax(s))
    others = [i for i in range(4) if i!=i_tl]

    # 在 others 里找直角顶点
    def angle_at(i, j, k):  # 夹角 j-i-k at i
        v1 = P[j]-P[i]; v2 = P[k]-P[i]
        v1/= (np.linalg.norm(v1)+1e-9); v2/= (np.linalg.norm(v2)+1e-9)
        c = np.clip(np.dot(v1,v2), -1, 1)
        return float(np.degrees(np.arccos(c)))
    angles = []
    for i in others:
        j,k = [x for x in others if x!=i]
        angles.append((abs(angle_at(i,j,k)-90.0), i))
    angles.sort()
    i_br = angles[0][1]
    i_tr, i_bl = [x for x in others if x!=i_br]

    BR = P[i_br]; TR = P[i_tr]; BL = P[i_bl]
    x_vec = TR - BR
    y_vec = BL - BR

    # 正交化（尽量）
    x_hat = x_vec / (np.linalg.norm(x_vec)+1e-9)
    y_vec = y_vec - x_hat*np.dot(y_vec, x_hat)
    y_hat = y_vec / (np.linalg.norm(y_vec)+1e-9)

    # 右手系：2D 叉积 z = x×y
    z = x_hat[0]*y_hat[1] - x_hat[1]*y_hat[0]
    if z < 0:  # 反了就交换
        x_hat, y_hat = y_hat, x_hat
        i_tr, i_bl = i_bl, i_tr

    return BR, x_hat, y_hat, (i_tl, i_tr, i_br, i_bl)

# --------------------- 按轴对小圆进行行列归属与编号 ---------------------
def number_smalls_with_axes(smalls_xy, O, x_hat, y_hat):
    """
    输入：小圆中心 rect 坐标 [(x,y),...]；O, x_hat, y_hat 来自大圆
    输出：list(dict(seq,row,col,x,y))，以及每一行的点坐标列表 rows_pts
    """
    P = np.asarray(smalls_xy, np.float32)
    rel = P - O[None,:]
    u = rel @ x_hat  # 投到 x
    v = rel @ y_hat  # 投到 y

    # ---------- v 方向做 7 类聚类（行） ----------
    K = 7
    v2 = v[:, None].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)

    def kmeans_pp_1d(v2):
        # OpenCV kmeans，kmeans++ 初始化，不需要 bestLabels
        _, labels, centers = cv2.kmeans(
            v2, K, None, crit, 10, cv2.KMEANS_PP_CENTERS
        )
        return labels.ravel(), centers.ravel()

    def lloyd_1d(v):
        # 自写 1D Lloyd 迭代（稳定回退）
        centers = np.linspace(float(v.min()), float(v.max()), K, dtype=np.float32)
        for _ in range(25):
            # 分配
            idx = np.argmin((v[:,None] - centers[None,:])**2, axis=1)
            new_c = centers.copy()
            moved = 0.0
            for k in range(K):
                if np.any(idx == k):
                    nk = v[idx == k].mean()
                    moved = max(moved, abs(nk - centers[k]))
                    new_c[k] = nk
            centers = new_c
            if moved < 1e-4:
                break
        return idx.astype(np.int32), centers

    try:
        labels, centers = kmeans_pp_1d(v2)
    except cv2.error:
        labels, centers = lloyd_1d(v)

    # 若因极端分布导致某些簇为空，做一次 Lloyd 清洗
    uniq = np.unique(labels)
    if len(uniq) < K:
        labels, centers = lloyd_1d(v)

    # 从小到大排序（沿 ŷ 由上到下）
    order_rows = np.argsort(centers)
    rows = [[] for _ in range(K)]
    for i in range(len(P)):
        r_raw = int(labels[i])
        r = int(np.where(order_rows == r_raw)[0][0])  # 0..6
        rows[r].append(i)

    numbered = []
    rows_pts = []

    def k_to_c(r, k):
        # 第 4 行跳过 col=3（中心空位）
        return k if (r != 3 or k < 3) else k + 1

    # 行优先；行内按 u 从小到大（沿 +x̂ 左→右）
    for r in range(K):
        idxs = rows[r]
        if not idxs:
            rows_pts.append([])
            continue
        idxs.sort(key=lambda i: u[i])
        pts_row = []
        col_k = 0
        for i in idxs:
            c = k_to_c(r, col_k)
            if c <= 5:  # 容错：最多 6 列
                numbered.append(dict(
                    seq=0,  # 先占位，稍后统一重写
                    row=int(r), col=int(c),
                    x=float(P[i, 0]), y=float(P[i, 1]),
                    u=float(u[i]), v=float(v[i])
                ))
                pts_row.append((float(P[i, 0]), float(P[i, 1])))
            col_k += 1
        rows_pts.append(pts_row)

    # 保障顺序：按 (row,col) 排并把 seq 设为 0..40
    numbered.sort(key=lambda d: (d["row"], d["col"]))
    for i, d in enumerate(numbered):
        d["seq"] = int(i)

    return numbered, rows_pts
# --------------------- 彩虹渐变与连线 ---------------------
def lerp(a,b,t): return a + (b-a)*t

def draw_poly_fancy(img, pts, color_a, color_b, width=2, alpha=0.9):
    """用两端颜色做线性渐变，按折线段绘制"""
    if len(pts)<2: return
    overlay = img.copy()
    n = len(pts)-1
    for k in range(n):
        t0 = k/float(max(1,n)); t1 = (k+1)/float(max(1,n))
        c0 = tuple(int(lerp(color_a[i], color_b[i], t0)) for i in range(3))
        c1 = tuple(int(lerp(color_a[i], color_b[i], t1)) for i in range(3))
        # 先画底色光晕
        cv2.line(overlay, (int(pts[k][0]),int(pts[k][1])),
                          (int(pts[k+1][0]),int(pts[k+1][1])), c0, width+4, cv2.LINE_AA)
        # 再画主线
        cv2.line(overlay, (int(pts[k][0]),int(pts[k][1])),
                          (int(pts[k+1][0]),int(pts[k+1][1])), c1, width, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_dashed(img, p, q, color=(200,200,200), width=1, dash=9, gap=8, alpha=0.5):
    p = np.array(p, np.float32); q = np.array(q, np.float32)
    v = q-p; L = np.linalg.norm(v)+1e-9; d = v/L
    nseg = int(L//(dash+gap))+1
    overlay = img.copy()
    for i in range(nseg):
        a = p + d*(i*(dash+gap))
        b = p + d*(i*(dash+gap)+dash)
        b = p + d*min(np.linalg.norm(b-p), L)
        cv2.line(overlay, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), color, width, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

# --------------------- 面积筛选（强制 41/4） ---------------------
def pick_by_area_and_count(items, target_n, area_range=None, relax=0.12, label="small"):
    if len(items)==0: return []
    areas = np.array([it.get("area", np.pi*it["r"]*it["r"]) for it in items], np.float32)
    if area_range is None:
        med = float(np.median(areas))
        mad = float(np.median(np.abs(areas - med)) + 1e-6)
        lo, hi = med - 2.5*mad, med + 2.5*mad
    else:
        lo, hi = float(area_range[0]), float(area_range[1])
    for _ in range(8):
        sel = [it for it,a in zip(items, areas) if lo <= a <= hi]
        if len(sel) >= target_n:
            s_areas = np.array([it.get("area", np.pi*it["r"]*it["r"]) for it in sel], np.float32)
            med = float(np.median(s_areas))
            score = np.abs(s_areas - med)
            order = np.argsort(score)
            sel = [sel[i] for i in order[:target_n]]
            return sel
        else:
            width = hi - lo
            lo -= width*relax*0.5
            hi += width*relax*0.5
    order = np.argsort(np.abs(areas - float(np.median(areas))))
    return [items[i] for i in order[:target_n]]

# --------------------- 主流程（单图） ---------------------
def process_one(path, out_dir):
    base = os.path.splitext(os.path.basename(path))[0]
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        logging.error(f"读图失败: {path}")
        return
    ensure_dir(out_dir)
    cv2.imwrite(os.path.join(out_dir,f"{base}_0_raw.png"), to_bgr(gray))

    # 1) 找板
    quad, ed = detect_by_hough_search(gray)
    if quad is None:
        quad, _ = detect_by_white_region(gray)
    if quad is None:
        logging.warning(f"[FAIL] {base} 未找到标定板"); return

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, [quad.astype(np.int32)], True, (0,200,255), 2)
    cv2.imwrite(os.path.join(out_dir,f"{base}_1_quad.png"), vis)

    # 2) 透视成 rect（短边提升到 >=1400）
    rect, H = warp_by_quad(gray, quad, min_short=1400)

    # 2.1) 预处理：CLAHE + 轻度高斯
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    rect_eq = clahe.apply(rect)
    rect_pre = cv2.GaussianBlur(rect_eq, (3,3), 0)
    cv2.imwrite(os.path.join(out_dir,f"{base}_2_rect.png"), to_bgr(rect_pre))

    # 3) 宽窗Blob候选（更宽容的参数）
    det = make_blob_detector(
        min_area=450.0,
        max_area=26000.0,
        dark_blobs=True,
        min_circ=0.45,
        min_conv=0.45
    )
    kps = det.detect(rect_pre)
    logging.info(f"{base} 候选(宽窗) = {len(kps)}")
    if len(kps)==0:
        logging.warning(f"[FAIL] {base} Blob 候选为 0"); return

    # --- KMeans 按 size 初分簇 ---
    sizes = [kp.size for kp in kps]
    if len(kps) >= 2:
        labels, centers = kmeans_1d_sizes(sizes)
    else:
        labels, centers = np.array([0]), np.array([sizes[0], sizes[0]])

    # 多者为小圆簇
    idx0 = np.where(labels==0)[0]; idx1 = np.where(labels==1)[0]
    if len(idx0) >= len(idx1):
        small_idx, big_idx = idx0, idx1
    else:
        small_idx, big_idx = idx1, idx0

    small_raw = [kps[i] for i in small_idx]
    big_raw   = [kps[i] for i in big_idx]
    logging.info(f"{base} 预分簇：小圆候选={len(small_raw)} 大圆候选={len(big_raw)}  (centers={centers})")

    # 4) 精细中心提取
    small_refined=[]; big_refined=[]
    rect_vis = cv2.cvtColor(rect_pre, cv2.COLOR_GRAY2BGR)

    def draw_one(vis, center, cnt, ell, tag, color_dot=(0,255,0)):
        cx,cy = center
        cv2.circle(vis, (int(round(cx)),int(round(cy))), 2, color_dot, -1, cv2.LINE_AA)
        if isinstance(cnt, np.ndarray) and len(cnt)>0:
            cv2.drawContours(vis, [cnt.astype(np.int32)], -1, (255,255,0), 1, cv2.LINE_AA)
        if ell is not None:
            cv2.ellipse(vis,
                        ( (int(ell[0][0]), int(ell[0][1])),
                          (int(ell[1][0]), int(ell[1][1])), float(ell[2]) ),
                        (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(vis, tag, (int(cx)+6, int(cy)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_dot, 1, cv2.LINE_AA)

    for group, out_list, cname in [(small_raw, small_refined, "S"), (big_raw, big_refined, "B")]:
        for i, kp in enumerate(group):
            (x,y), r0 = kp.pt, kp.size * 0.5
            (cx,cy), r_est, tag, cnt, ell, debug = refine_center_on_patch(rect_pre, (x,y), r0, gate=0.6)
            area = float(debug["area"]) if (debug and "area" in debug) else float(np.pi*r_est*r_est)
            out_list.append(dict(
                id=f"{cname}{i}", x=float(cx), y=float(cy), r=float(r_est),
                tag=tag, area=area, debug=debug
            ))
            draw_one(rect_vis, (cx,cy), cnt, ell, f"{cname}{i}:{tag}",
                     (0,255,0) if cname=="S" else (0,128,255))

    # --- 极端回退 ---
    all_refined = small_refined + big_refined
    if len(all_refined) >= 8 and (len(small_refined) < 30 or len(big_refined) < 2):
        areas = np.array([it["area"] for it in all_refined], np.float32)
        order = np.argsort(-areas)
        big_pick = [all_refined[i] for i in order[:min(6, len(all_refined))]]
        big_refined = pick_by_area_and_count(big_pick, 4, area_range=None, relax=0.20, label="big-re")
        big_ids = set(id(it) for it in big_refined)
        small_refined = [it for it in all_refined if id(it) not in big_ids]
        logging.info(f"{base} 触发面积回退：小圆={len(small_refined)} 大圆={len(big_refined)} (从 {len(all_refined)} 候选重分)")

    # 4.1) 面积一致化筛选，强制 41/4
    small_refined = pick_by_area_and_count(small_refined, 41, area_range=None, relax=0.14, label="small")
    big_refined   = pick_by_area_and_count(big_refined,   4, area_range=None, relax=0.14, label="big")
    logging.info(f"{base} 筛选后：小圆={len(small_refined)} 大圆={len(big_refined)}")

    cv2.imwrite(os.path.join(out_dir,f"{base}_3_rect_refined.png"), rect_vis)

    # 5) —— 用 4 大圆确定坐标轴（只旋转不镜像）
    if len(big_refined) != 4:
        logging.warning(f"[WARN] {base} 大圆不足 4 个，跳过轴锁定；将使用粗略的 y→x 排序可视化。")
        O = np.array([rect_pre.shape[1]*0.5, rect_pre.shape[0]*0.5], np.float32)
        x_hat = np.array([1.0, 0.0], np.float32)
        y_hat = np.array([0.0, 1.0], np.float32)
    else:
        big_pts = [(it["x"], it["y"]) for it in big_refined]
        O, x_hat, y_hat, idx4 = axes_from_big4(big_pts)

    # 6) —— 按轴对小圆编号（0..40；行优先；r==3 跳过 c==3）
    small_xy = [(it["x"], it["y"]) for it in small_refined]
    small_numbered, rows_pts = number_smalls_with_axes(small_xy, O, x_hat, y_hat)

    # 7) 可视化：rect 上 —— 坐标轴 + 编号 + 行连线 + 行间虚线衔接
    rect_anno = cv2.cvtColor(rect_pre, cv2.COLOR_GRAY2BGR)

    # 7.1 坐标轴（红=x，绿=y）
    def draw_arrow(img, p, v, color, text):
        p = tuple(map(int, np.round(p)))
        q = tuple(map(int, np.round(p[0] + v[0],))), tuple(map(int, np.round(p[1] + v[1],)))
    L = 180.0
    p0 = (float(O[0]), float(O[1]))
    px = (float(O[0]+x_hat[0]*L), float(O[1]+x_hat[1]*L))
    py = (float(O[0]+y_hat[0]*L), float(O[1]+y_hat[1]*L))
    cv2.arrowedLine(rect_anno, (int(p0[0]),int(p0[1])), (int(px[0]),int(px[1])), (0,0,255), 3, cv2.LINE_AA, tipLength=0.08)
    cv2.arrowedLine(rect_anno, (int(p0[0]),int(p0[1])), (int(py[0]),int(py[1])), (0,200,0), 3, cv2.LINE_AA, tipLength=0.08)
    annotate_text(rect_anno, "x", px, (0,0,255), 0.9, 2)
    annotate_text(rect_anno, "y", py, (0,200,0), 0.9, 2)

    # 7.2 小圆编号
    for d in small_numbered:
        annotate_text(rect_anno, d["seq"], (d["x"]+5, d["y"]-6), (0,0,255), 0.9, 2)
        cv2.circle(rect_anno, (int(round(d["x"])),int(round(d["y"]))), 2, (0,255,0), -1, cv2.LINE_AA)

    # 7.3 行连线（彩虹）+ 行间虚线衔接
    palette = [
        (255, 64, 64),   # R
        (255,180, 72),   # O
        (240,240, 64),   # Y
        ( 64,220, 64),   # G
        ( 64,220,220),   # C
        ( 64, 96,255),   # B
        (255, 64,255),   # M
    ]
    # rows_pts 已按 0..6 存储；每行内部已是从左到右
    for r, pts in enumerate(rows_pts):
        if len(pts) >= 2:
            color_a = palette[r % len(palette)]
            color_b = (min(255, color_a[0]+60), min(255, color_a[1]+40), min(255, color_a[2]+60))
            draw_poly_fancy(rect_anno, pts, color_a, color_b, width=3, alpha=0.85)
        # 行间虚线衔接
        if r < len(rows_pts)-1 and pts and rows_pts[r+1]:
            p_tail = pts[-1]
            p_head = rows_pts[r+1][0]
            draw_dashed(rect_anno, p_tail, p_head, (220,220,220), width=1, dash=10, gap=9, alpha=0.45)

    cv2.imwrite(os.path.join(out_dir,f"{base}_5_rect_numbered.png"), rect_anno)

    # 8) raw 上回投标注（同样风格）
    raw_anno = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 原点与轴回投
    X0,Y0 = H_inv_point(H, O[0], O[1])
    Xx,Yx = H_inv_point(H, O[0]+x_hat[0]*L, O[1]+x_hat[1]*L)
    Xy,Yy = H_inv_point(H, O[0]+y_hat[0]*L, O[1]+y_hat[1]*L)
    cv2.arrowedLine(raw_anno, (int(X0),int(Y0)), (int(Xx),int(Yx)), (0,0,255), 3, cv2.LINE_AA, tipLength=0.08)
    cv2.arrowedLine(raw_anno, (int(X0),int(Y0)), (int(Xy),int(Yy)), (0,200,0), 3, cv2.LINE_AA, tipLength=0.08)
    annotate_text(raw_anno, "x", (Xx,Yx), (0,0,255), 0.9, 2)
    annotate_text(raw_anno, "y", (Xy,Yy), (0,200,0), 0.9, 2)

    # 小圆编号 + 连线
    rows_pts_raw=[]
    for r, pts in enumerate(rows_pts):
        pr=[]
        for (x,y) in pts:
            X,Y = H_inv_point(H, x, y)
            pr.append((X,Y))
        rows_pts_raw.append(pr)
    for d in small_numbered:
        X,Y = H_inv_point(H, d["x"], d["y"])
        annotate_text(raw_anno, d["seq"], (X+5, Y-6), (0,0,255), 0.9, 2)
        cv2.circle(raw_anno, (int(round(X)),int(round(Y))), 2, (0,255,0), -1, cv2.LINE_AA)
    for r, pts in enumerate(rows_pts_raw):
        if len(pts) >= 2:
            color_a = palette[r % len(palette)]
            color_b = (min(255, color_a[0]+60), min(255, color_a[1]+40), min(255, color_a[2]+60))
            draw_poly_fancy(raw_anno, pts, color_a, color_b, width=3, alpha=0.85)
        if r < len(rows_pts_raw)-1 and pts and rows_pts_raw[r+1]:
            draw_dashed(raw_anno, pts[-1], rows_pts_raw[r+1][0], (220,220,220), width=1, dash=10, gap=9, alpha=0.45)

    cv2.imwrite(os.path.join(out_dir,f"{base}_6_raw_numbered.png"), raw_anno)

    # 9) JSON 输出
    out_json = {
        "quad": np.asarray(quad).tolist(),
        "homography": np.asarray(H).tolist(),
        "big_circles": big_refined,
        "small_circles": small_refined,
        "axis_origin_rect": [float(O[0]), float(O[1])],
        "axis_x_hat": [float(x_hat[0]), float(x_hat[1])],
        "axis_y_hat": [float(y_hat[0]), float(y_hat[1])],
        "small_numbered": small_numbered,  # seq,row,col,x,y,u,v
    }
    with open(os.path.join(out_dir,f"{base}_circles.json"),"w",encoding="utf-8") as f:
        json.dump(out_json,f,indent=2,ensure_ascii=False)

    logging.info(f"[OK] {base}: 小圆={len(small_refined)} 大圆={len(big_refined)} 已按轴编号并绘制。")

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/augmented/realistic",
                    help="输入图像目录")
    ap.add_argument("--out",   default="outputs/board_finder_v31",
                    help="输出结果目录")
    args = ap.parse_args()
    ensure_dir(args.out)
    imgs = sorted(sum([glob.glob(os.path.join(args.indir, ext))
                       for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.dng","*.DNG")], []))
    if not imgs:
        logging.error("未找到输入图像"); return
    for p in tqdm(imgs, desc="[BoardAndCircleFinder v3.1]"):
        process_one(p, args.out)

if __name__ == "__main__":
    main()