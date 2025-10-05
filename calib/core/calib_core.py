# -*- coding: utf-8 -*-
"""
Calibrator：输入灰度图（uint8），输出 BoardResult（41 小圆编号、4 大圆、轴、H 等）
不负责画图/保存；仅算法与数据。
"""
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class DetectionConfig:
    # Quad/warp parameters
    quad_expand_scale: float = 1.03
    quad_expand_offset: float = 12.0
    warp_min_short: int = 1400
    warp_min_dim: int = 400
    warp_interp: int = cv2.INTER_CUBIC

    # Hough search parameters
    hough_gaussian_sigma: float = 1.0
    hough_canny_low_ratio: float = 0.66
    hough_canny_low_min: int = 10
    hough_canny_high_ratio: float = 2.0
    hough_dilate_kernel: int = 3
    hough_dilate_iterations: int = 1
    hough_votes_ratio: float = 0.006
    hough_min_line_ratio: float = 0.30
    hough_max_gap_ratio: float = 0.03
    hough_orientation_tol: float = 5.0
    hough_orthogonality_tol: float = 10.0
    hough_rho_nms_ratio: float = 0.02
    hough_kmeans_max_iter: int = 100
    hough_kmeans_eps: float = 1e-4
    hough_kmeans_attempts: int = 8

    # Quadrilateral scoring
    quad_margin: float = 0.03
    quad_area_min_ratio: float = 0.15
    quad_area_max_ratio: float = 0.80
    quad_aspect_min: float = 0.90
    quad_aspect_max: float = 1.35
    quad_edge_half: int = 6
    quad_edge_samples: int = 48
    quad_edge_min_contrast: float = 0.5
    quad_area_bonus: float = 300.0

    # White-region fallback
    white_gaussian_sigma: float = 1.2
    white_morph_kernel: int = 11
    white_morph_iterations: int = 1
    white_approx_eps_ratio: float = 0.0125
    white_approx_expand: float = 1.3
    white_approx_shrink: float = 0.7

    # CLAHE & rect preprocessing
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    rect_blur_kernel: Tuple[int, int] = (3, 3)

    # Blob detector
    blob_min_area: float = 450.0
    blob_max_area: float = 26000.0
    blob_dark: bool = True
    blob_min_circularity: float = 0.45
    blob_min_convexity: float = 0.45
    blob_min_inertia: float = 0.04
    blob_min_threshold: float = 5.0
    blob_max_threshold: float = 220.0
    blob_threshold_step: float = 5.0
    blob_min_dist: float = 10.0

    # Refinement
    refine_gate: float = 0.6
    refine_win_scale: float = 3.0
    refine_win_min: float = 30.0
    refine_win_max: float = 220.0
    refine_segment_ksize: int = 3
    refine_open_kernel: Tuple[int, int] = (3, 3)
    fallback_canny_low: int = 30
    fallback_canny_high: int = 90

    # Area selection
    area_relax_default: float = 0.12
    area_relax_small: float = 0.14
    area_relax_big: float = 0.14
    area_relax_reassign_big: float = 0.20
    area_iterations: int = 8


def create_detection_config(**overrides) -> DetectionConfig:
    """Create a DetectionConfig with selective overrides for convenient tuning."""
    cfg = DetectionConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown detection config field: {key}")
        setattr(cfg, key, value)
    return cfg


DEFAULT_CONFIG = DetectionConfig()

HIGH_RECALL_CONFIG = create_detection_config(
    quad_expand_scale=1.05,
    quad_expand_offset=16.0,
    warp_min_short=1600,
    hough_votes_ratio=0.0045,
    hough_min_line_ratio=0.25,
    hough_max_gap_ratio=0.05,
    hough_orientation_tol=7.5,
    hough_orthogonality_tol=14.0,
    white_morph_kernel=13,
    white_approx_shrink=0.6,
    blob_min_area=320.0,
    blob_max_area=32000.0,
    area_relax_small=0.18,
    area_relax_big=0.18,
    area_relax_reassign_big=0.26,
)

from .types import Axis, Circle, SmallNumbered, BoardResult, Pt

# --------------------- Utils ---------------------
def order_quad(pts4):
    pts4 = np.asarray(pts4, np.float32)
    s = pts4.sum(1); d = np.diff(pts4,axis=1).reshape(-1)
    tl = pts4[np.argmin(s)]; br = pts4[np.argmax(s)]
    tr = pts4[np.argmin(d)]; bl = pts4[np.argmax(d)]
    return np.array([tl,tr,br,bl], np.float32)


def expand_quad(quad: np.ndarray, scale: Optional[float] = None, offset: Optional[float] = None,
               cfg: DetectionConfig = DEFAULT_CONFIG) -> np.ndarray:
    """沿着中心方向外扩四边形，避免透视下裁剪到边缘"""
    q = np.asarray(quad, np.float32)
    centroid = q.mean(axis=0, keepdims=True)
    vec = q - centroid
    if scale is None:
        scale = cfg.quad_expand_scale
    if offset is None:
        offset = cfg.quad_expand_offset
    if scale != 1.0:
        q = centroid + vec * float(scale)
    if offset != 0.0:
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        dirs = np.divide(vec, np.maximum(norms, 1e-6))
        q = q + dirs * float(offset)
    return q

def warp_by_quad(gray, quad, min_short: Optional[int] = None,
                 cfg: DetectionConfig = DEFAULT_CONFIG):
    if min_short is None:
        min_short = cfg.warp_min_short
    min_dim = int(cfg.warp_min_dim)
    W = int(max(min_dim, np.linalg.norm(quad[1]-quad[0])))
    H = int(max(min_dim, np.linalg.norm(quad[3]-quad[0])))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
    Hm = cv2.getPerspectiveTransform(quad, dst)
    rect = cv2.warpPerspective(gray, Hm, (W, H), flags=cfg.warp_interp)
    short = min(W,H)
    if short < min_short:
        s = float(min_short)/short
        rect = cv2.resize(rect, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        Hm = np.array([[s,0,0],[0,s,0],[0,0,1]], np.float32) @ Hm
    return rect, Hm

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

def quad_score(gray, quad, cfg: DetectionConfig = DEFAULT_CONFIG):
    h,w = gray.shape
    margin = cfg.quad_margin
    if np.any(quad[:,0] < margin*w) or np.any(quad[:,0] > (1-margin)*w) \
       or np.any(quad[:,1] < margin*h) or np.any(quad[:,1] > (1-margin)*h):
        return False, -1e9
    area = cv2.contourArea(quad.astype(np.float32))
    if area < cfg.quad_area_min_ratio*h*w or area > cfg.quad_area_max_ratio*h*w:
        return False, -1e9
    wlen = 0.5*(np.linalg.norm(quad[1]-quad[0])+np.linalg.norm(quad[2]-quad[3]))
    hlen = 0.5*(np.linalg.norm(quad[3]-quad[0])+np.linalg.norm(quad[2]-quad[1]))
    ratio = max(wlen,hlen)/max(1.0,min(wlen,hlen))
    if not (cfg.quad_aspect_min <= ratio <= cfg.quad_aspect_max):
        return False, -1e9
    csum = 0.0
    for i in range(4):
        v = edge_contrast(gray, quad[i], quad[(i+1)%4], half=cfg.quad_edge_half, ns=cfg.quad_edge_samples)
        if v < cfg.quad_edge_min_contrast:
            return False, -1e9
        csum += v
    return True, csum + cfg.quad_area_bonus*(area/(h*w))

def detect_by_hough_search(gray, cfg: DetectionConfig = DEFAULT_CONFIG):
    h,w = gray.shape
    g = cv2.GaussianBlur(gray, (0,0), cfg.hough_gaussian_sigma)
    med = np.median(g)
    lo = int(max(cfg.hough_canny_low_min, cfg.hough_canny_low_ratio*med))
    hi = int(max(lo+1, cfg.hough_canny_high_ratio*lo))
    edges = cv2.Canny(g, lo, hi)
    if cfg.hough_dilate_kernel > 0 and cfg.hough_dilate_iterations > 0:
        kernel = np.ones((cfg.hough_dilate_kernel, cfg.hough_dilate_kernel), np.uint8)
        edges = cv2.dilate(edges, kernel, cfg.hough_dilate_iterations)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        int(cfg.hough_votes_ratio*min(h,w)),
        minLineLength=int(cfg.hough_min_line_ratio*min(h,w)),
        maxLineGap=int(cfg.hough_max_gap_ratio*min(h,w)),
    )
    if lines is None or len(lines)<4: return None, edges
    Ls = [seg_to_normal_form(*xy) for xy in lines.reshape(-1,4)]

    feats = np.array([[np.cos(2*L[3]), np.sin(2*L[3])] for L in Ls], np.float32)
    crit  = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, cfg.hough_kmeans_max_iter, cfg.hough_kmeans_eps)
    _, labels, _ = cv2.kmeans(feats, 2, None, crit, cfg.hough_kmeans_attempts, cv2.KMEANS_PP_CENTERS)
    groups = [[],[]]
    for L, lb in zip(Ls, labels.ravel()):
        groups[lb].append(L)
    if len(groups[0])<2 or len(groups[1])<2: return None, edges

    def nms_1d(lines, rho_thr):
        kept=[]
        for L in sorted(lines,key=lambda t:t[4]):
            if all(abs(L[4]-K[4])>rho_thr for K in kept): kept.append(L)
        return kept
    G0 = nms_1d(groups[0], cfg.hough_rho_nms_ratio*max(h,w))
    G1 = nms_1d(groups[1], cfg.hough_rho_nms_ratio*max(h,w))

    best=None; best_score=-1e9
    from itertools import combinations
    for L0a,L0b in combinations(G0,2):
        if deg_diff(line_angle_deg(L0a),line_angle_deg(L0b))>cfg.hough_orientation_tol: continue
        for L1a,L1b in combinations(G1,2):
            if deg_diff(line_angle_deg(L1a),line_angle_deg(L1b))>cfg.hough_orientation_tol: continue
            if abs(deg_diff(line_angle_deg(L0a),line_angle_deg(L1a))-90)>cfg.hough_orthogonality_tol: continue
            pts=[intersect(L0a,L1a),intersect(L0b,L1a),intersect(L0b,L1b),intersect(L0a,L1b)]
            if any(p is None for p in pts): continue
            quad=order_quad(np.stack(pts,0))
            ok, sc=quad_score(gray, quad, cfg)
            if ok and sc>best_score:
                best_score=sc; best=quad
    return best, edges

def detect_by_white_region(gray, cfg: DetectionConfig = DEFAULT_CONFIG):
    g = cv2.GaussianBlur(gray,(0,0),cfg.white_gaussian_sigma)
    _, th = cv2.threshold(g, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(cfg.white_morph_kernel,cfg.white_morph_kernel)),
                          cfg.white_morph_iterations)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if num <= 1:
        return None, th

    H, W = th.shape
    total_area = float(H * W)
    border_margin = max(3, int(0.01 * min(H, W)))
    candidates = []
    for idx in range(1, num):
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if w < 8 or h < 8:
            continue
        fill_ratio = area / max(1.0, float(w * h))
        touch_left = x <= border_margin
        touch_top = y <= border_margin
        touch_right = (x + w) >= (W - border_margin)
        touch_bottom = (y + h) >= (H - border_margin)
        touch_count = int(touch_left) + int(touch_top) + int(touch_right) + int(touch_bottom)
        frame_ratio = area / total_area
        fill_factor = 0.2 + 0.8 * np.clip(fill_ratio, 0.0, 1.0)
        border_factor = 1.0 / (1.0 + 0.6 * touch_count)
        global_penalty = max(0.2, 1.0 - max(0.0, frame_ratio - 0.55) * 0.8)
        score = area * fill_factor * border_factor * global_penalty
        candidates.append((score, area, -touch_count, fill_ratio, idx, touch_count))

    if not candidates:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    else:
        primary = [c for c in candidates if c[-1] <= 2]
        pool = primary if primary else candidates
    pool.sort(reverse=True)
    idx = pool[0][4]

    mask = np.zeros_like(th); mask[lab == idx] = 255
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, mask
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    eps = cfg.white_approx_eps_ratio * peri
    quad = None
    for _ in range(cfg.area_iterations):
        approx = cv2.approxPolyDP(hull, eps, True)
        if len(approx) == 4:
            quad = approx.reshape(-1, 2).astype(np.float32)
            break
        if len(approx) > 4:
            eps *= cfg.white_approx_expand
        else:
            eps *= cfg.white_approx_shrink
    if quad is None:
        rect = cv2.minAreaRect(hull)
        quad = cv2.boxPoints(rect).astype(np.float32)
    return order_quad(quad), mask


def refine_quad_local(gray: np.ndarray, quad: Optional[np.ndarray],
                      cfg: DetectionConfig = DEFAULT_CONFIG) -> Optional[np.ndarray]:
    if quad is None:
        return None
    quad = np.asarray(quad, np.float32)
    if quad.shape != (4, 2):
        return quad
    h, w = gray.shape
    xs = quad[:, 0]
    ys = quad[:, 1]
    pad = max(int(cfg.quad_expand_offset * 1.5), 20)
    x1 = int(max(0, np.floor(xs.min() - pad)))
    y1 = int(max(0, np.floor(ys.min() - pad)))
    x2 = int(min(w, np.ceil(xs.max() + pad)))
    y2 = int(min(h, np.ceil(ys.max() + pad)))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return quad
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return quad
    g = cv2.GaussianBlur(roi, (0, 0), cfg.hough_gaussian_sigma)
    med = np.median(g)
    lo = int(max(cfg.hough_canny_low_min, cfg.hough_canny_low_ratio * med))
    hi = int(max(lo + 1, cfg.hough_canny_high_ratio * lo))
    edges = cv2.Canny(g, lo, hi)
    if cfg.hough_dilate_kernel > 0 and cfg.hough_dilate_iterations >= 0:
        kernel = np.ones((cfg.hough_dilate_kernel, cfg.hough_dilate_kernel), np.uint8)
        edges = cv2.dilate(edges, kernel, cfg.hough_dilate_iterations + 1)
    mask = np.zeros_like(edges)
    local_quad = (quad - np.array([x1, y1], np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, local_quad, 255)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), 1)
    edges = cv2.bitwise_and(edges, mask)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return quad
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50.0:
        return quad
    peri = cv2.arcLength(cnt, True)
    if peri < 1e-3:
        return quad
    eps = 0.01 * peri
    approx = cv2.approxPolyDP(cnt, eps, True)
    tries = 0
    while len(approx) > 4 and tries < 6:
        eps *= 1.5
        approx = cv2.approxPolyDP(cnt, eps, True)
        tries += 1
    if len(approx) < 4:
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(cnt)
        approx = cv2.boxPoints(rect)
    approx = np.asarray(approx, np.float32).reshape(-1, 2)
    refined = approx + np.array([x1, y1], np.float32)
    refined = order_quad(refined)
    ok, _ = quad_score(gray, refined, cfg)
    if not ok:
        return quad
    center_old = quad.mean(axis=0)
    center_new = refined.mean(axis=0)
    if np.linalg.norm(center_new - center_old) > max(pad * 0.8, 40):
        return quad
    return refined

# --------------------- Blob & refine ---------------------
def make_blob_detector(cfg: DetectionConfig = DEFAULT_CONFIG,
                       min_area: Optional[float] = None,
                       max_area: Optional[float] = None,
                       dark_blobs: Optional[bool] = None,
                       min_circ: Optional[float] = None,
                       min_conv: Optional[float] = None):
    if min_area is None:
        min_area = cfg.blob_min_area
    if max_area is None:
        max_area = cfg.blob_max_area
    if dark_blobs is None:
        dark_blobs = cfg.blob_dark
    if min_circ is None:
        min_circ = cfg.blob_min_circularity
    if min_conv is None:
        min_conv = cfg.blob_min_convexity
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = float(cfg.blob_min_threshold)
    p.maxThreshold = float(cfg.blob_max_threshold)
    p.thresholdStep = float(cfg.blob_threshold_step)
    p.filterByArea = True; p.minArea = float(min_area); p.maxArea = float(max_area)
    p.filterByCircularity = True; p.minCircularity = float(min_circ)
    p.filterByInertia = True; p.minInertiaRatio = float(cfg.blob_min_inertia)
    p.filterByConvexity = True; p.minConvexity = float(min_conv)
    p.filterByColor = True; p.blobColor = 0 if dark_blobs else 255
    p.minDistBetweenBlobs = float(cfg.blob_min_dist)
    return cv2.SimpleBlobDetector_create(p)

def segment_component(patch, seed, cfg: DetectionConfig = DEFAULT_CONFIG):
    ksize = max(1, int(cfg.refine_segment_ksize))
    if ksize % 2 == 0:
        ksize += 1
    g = cv2.GaussianBlur(patch,(ksize,ksize),0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th = cv2.medianBlur(th, 3)
    kernel_size = tuple(int(max(1, k)) for k in cfg.refine_open_kernel)
    kernel_size = tuple(k + (k % 2 == 0) for k in kernel_size)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size), 1)
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

def center_by_moments(cnt):
    m = cv2.moments(cnt)
    if abs(m["m00"]) < 1e-6: return None
    return (float(m["m10"]/m["m00"]), float(m["m01"]/m["m00"]))

def center_by_ellipse(cnt):
    if len(cnt) < 5: return None, None
    ell = cv2.fitEllipse(cnt)
    (cx,cy),(ma,Mi),ang = ell
    return (float(cx),float(cy)), ((float(ma),float(Mi)), float(ang), ell)

def center_by_dt_peak(mask, approx=(3,3)):
    if mask.max()==0: return None
    inv = (mask>0).astype(np.uint8)*255
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    y,x = np.unravel_index(np.argmax(dt), dt.shape)
    c0 = np.array([[float(x), float(y)]], np.float32)
    dt8 = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    term=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    try: cv2.cornerSubPix(dt8, c0, (3,3), (-1,-1), term)
    except cv2.error: pass
    return (float(c0[0,0]), float(c0[0,1]))

def average_radius_to_edge(center, cnt):
    cx,cy = center
    pts = cnt.reshape(-1,2).astype(np.float32)
    rr = np.hypot(pts[:,0]-cx, pts[:,1]-cy)
    return float(np.median(rr))

def fuse_centers(patch, mask, cnt, cand, ell_c, ell_info, dt_c):
    scores=[]; cands=[]; tags=[]
    if cand is not None: cands.append(cand); tags.append("seed"); scores.append(0.5)
    if cnt is not None:
        m_c = center_by_moments(cnt)
        if m_c is not None: cands.append(m_c); tags.append("moments"); scores.append(0.9)
    if ell_c is not None and ell_info is not None:
        (ma,Mi), ang, ell = ell_info
        ar = min(ma,Mi)/max(ma,Mi) if max(ma,Mi)>1e-6 else 0
        area_ok = (np.pi*0.25*ma*Mi) > 50
        qual = 0.6 + 0.4*ar if area_ok else 0.3*ar
        cands.append(ell_c); tags.append("ellipse"); scores.append(float(qual))
    if dt_c is not None: cands.append(dt_c); tags.append("dt"); scores.append(0.95)
    Cs = np.array(cands, np.float32); w = np.array(scores, np.float32)
    if len(Cs)==1: return cands[0], tags[0], 1.0
    w /= (w.sum()+1e-9)
    mu = (Cs*w[:,None]).sum(0)
    d = np.linalg.norm(Cs - mu[None,:], axis=1)
    sigma = (np.median(d)+1e-6)
    damp = np.exp(-(d/(3*sigma))**2)
    w2 = w * damp; w2 /= (w2.sum()+1e-9)
    fused = (Cs*w2[:,None]).sum(0)
    conf = float(w2.max()); tag = tags[int(np.argmax(w2))]
    return (float(fused[0]), float(fused[1])), tag, conf

def circle_fit_least_squares(xy):
    x = xy[:,0]; y = xy[:,1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x*x + y*y
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, d = c
    r = np.sqrt(max(1e-6, cx*cx + cy*cy + d))
    return float(cx), float(cy), float(r)

def refine_center_on_patch(rect, c0, r0, cfg: DetectionConfig = DEFAULT_CONFIG):
    h,w = rect.shape
    x0,y0 = c0
    win = int(np.clip(cfg.refine_win_scale*r0, cfg.refine_win_min, cfg.refine_win_max))
    x1 = max(0, int(x0-win)); x2 = min(w, int(x0+win))
    y1 = max(0, int(y0-win)); y2 = min(h, int(y0+win))
    patch = rect[y1:y2, x1:x2]
    if patch.size==0: return (x0,y0), r0, "seed", None, None, None
    seed_local = (x0 - x1, y0 - y1)
    mask, cnt = segment_component(patch, seed_local, cfg)
    if len(cnt)==0:
        edges = cv2.Canny(cv2.GaussianBlur(patch,(0,0),1.0), cfg.fallback_canny_low, cfg.fallback_canny_high)
        ys,xs = np.nonzero(edges)
        if len(xs)<15: return (x0,y0), r0, "seed", None, None, None
        pts = np.stack([xs,ys],1).astype(np.float32)
        cx,cy,r = circle_fit_least_squares(pts)
        return (x1+cx, y1+cy), r, "lsq_circle", pts, None, None
    ell_c, ell_info = center_by_ellipse(cnt)
    mom_c = center_by_moments(cnt)
    dt_c  = center_by_dt_peak(mask)
    fused_c, tag, conf = fuse_centers(patch, mask, cnt, seed_local, ell_c, ell_info, dt_c)
    r_est = average_radius_to_edge(fused_c, cnt) if len(cnt)>0 else r0
    cx,cy = fused_c; cx += x1; cy += y1
    area = float(cv2.contourArea(cnt)) if isinstance(cnt, np.ndarray) and len(cnt)>0 else float(np.pi*r_est*r_est)
    return (cx,cy), r_est, tag, cnt + np.array([x1,y1]), None, dict(area=area)

# --------------------- KMeans helpers ---------------------
def kmeans_1d_sizes(sizes):
    v = np.array(sizes, np.float32)[:,None]
    crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,1e-4)
    _, labels, centers = cv2.kmeans(v, 2, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    if centers[0] > centers[1]:
        labels = 1 - labels; centers = centers[::-1]
    return labels.ravel(), centers.ravel()

# --------------------- 方向估计（由 4 个大圆） ---------------------
def axes_from_big4(points_4):
    P = np.asarray(points_4, np.float32)
    D = np.linalg.norm(P[:,None,:]-P[None,:,:], axis=2); s = D.sum(1)
    i_tl = int(np.argmax(s))
    others = [i for i in range(4) if i!=i_tl]
    def angle_at(i, j, k):
        v1 = P[j]-P[i]; v2 = P[k]-P[i]
        v1/= (np.linalg.norm(v1)+1e-9); v2/= (np.linalg.norm(v2)+1e-9)
        c = np.clip(np.dot(v1,v2), -1, 1); return float(np.degrees(np.arccos(c)))
    angles = []
    for i in others:
        j,k = [x for x in others if x!=i]
        angles.append((abs(angle_at(i,j,k)-90.0), i))
    angles.sort()
    i_br = angles[0][1]
    i_tr, i_bl = [x for x in others if x!=i_br]
    BR = P[i_br]; TR = P[i_tr]; BL = P[i_bl]
    x_vec = TR - BR; y_vec = BL - BR
    x_hat = x_vec / (np.linalg.norm(x_vec)+1e-9)
    y_vec = y_vec - x_hat*np.dot(y_vec, x_hat)
    y_hat = y_vec / (np.linalg.norm(y_vec)+1e-9)
    z = x_hat[0]*y_hat[1] - x_hat[1]*y_hat[0]
    if z < 0:
        x_hat, y_hat = y_hat, x_hat
    return BR, x_hat, y_hat, (i_tl, i_tr, i_br, i_bl)

# --------------------- 小圆编号（按轴） ---------------------
def number_smalls_with_axes(smalls_xy, O, x_hat, y_hat):
    P = np.asarray(smalls_xy, np.float32)
    rel = P - O[None,:]
    u = rel @ x_hat
    v = rel @ y_hat
    K = 7
    v2 = v[:, None].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    def kmeans_pp_1d(v2):
        _, labels, centers = cv2.kmeans(v2, K, None, crit, 10, cv2.KMEANS_PP_CENTERS)
        return labels.ravel(), centers.ravel()
    def lloyd_1d(v):
        centers = np.linspace(float(v.min()), float(v.max()), K, dtype=np.float32)
        for _ in range(25):
            idx = np.argmin((v[:,None] - centers[None,:])**2, axis=1)
            new_c = centers.copy(); moved = 0.0
            for k in range(K):
                if np.any(idx==k):
                    nk = v[idx==k].mean()
                    moved = max(moved, abs(nk - centers[k])); new_c[k] = nk
            centers = new_c
            if moved < 1e-4: break
        return idx.astype(np.int32), centers
    try:
        labels, centers = kmeans_pp_1d(v2)
    except cv2.error:
        labels, centers = lloyd_1d(v)
    if len(np.unique(labels)) < K:
        labels, centers = lloyd_1d(v)
    order_rows = np.argsort(centers)
    rows = [[] for _ in range(K)]
    for i in range(len(P)):
        r_raw = int(labels[i])
        r = int(np.where(order_rows == r_raw)[0][0])
        rows[r].append(i)
    numbered = []; rows_pts=[]
    def k_to_c(r, k): return k if (r!=3 or k<3) else k+1
    for r in range(K):
        idxs = rows[r]
        if not idxs:
            rows_pts.append([]); continue
        idxs.sort(key=lambda i: u[i])
        pts_row=[]; col_k=0
        for i in idxs:
            c = k_to_c(r, col_k)
            if c <= 5:
                numbered.append(dict(seq=0,row=int(r),col=int(c),
                                     x=float(P[i,0]),y=float(P[i,1]),
                                     u=float(u[i]),v=float(v[i])))
                pts_row.append((float(P[i,0]),float(P[i,1])))
            col_k += 1
        rows_pts.append(pts_row)
    numbered.sort(key=lambda d:(d["row"],d["col"]))
    for i,d in enumerate(numbered): d["seq"]=int(i)
    return numbered, rows_pts

# --------------------- 面积筛选（强制 41/4） ---------------------
def pick_by_area_and_count(items, target_n, area_range=None,
                           relax: Optional[float] = None,
                           label: str = "small",
                           cfg: DetectionConfig = DEFAULT_CONFIG):
    if len(items)==0: return []
    if relax is None:
        relax = cfg.area_relax_default
    areas = np.array([it.get("area", np.pi*it["r"]*it["r"]) for it in items], np.float32)
    if area_range is None:
        med = float(np.median(areas))
        mad = float(np.median(np.abs(areas - med)) + 1e-6)
        lo, hi = med - 2.5*mad, med + 2.5*mad
    else:
        lo, hi = float(area_range[0]), float(area_range[1])
    for _ in range(cfg.area_iterations):
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
            lo -= width*relax*0.5; hi += width*relax*0.5
    order = np.argsort(np.abs(areas - float(np.median(areas))))
    return [items[i] for i in order[:target_n]]

# --------------------- Calibrator API ---------------------
class Calibrator:
    def __init__(self, min_short: Optional[int] = None, config: DetectionConfig = DEFAULT_CONFIG):
        self.config = config
        if min_short is None:
            min_short = config.warp_min_short
        self.min_short = int(min_short)

    def process_gray(self, gray: np.ndarray, debug: Optional[Dict[str, Any]] = None) -> Optional[BoardResult]:
        assert gray.ndim == 2 and gray.dtype == np.uint8, "expect uint8 gray"
        cfg = self.config

        if debug is not None:
            debug.clear()
            debug["stage"] = "init"
            debug["fail_reason"] = None

        quad, edges = detect_by_hough_search(gray, cfg)
        if debug is not None:
            debug["stage"] = "hough"
            debug["hough_edges"] = edges.copy() if edges is not None else None
            debug["hough_quad"] = quad.copy() if quad is not None else None
        if quad is None:
            quad, mask = detect_by_white_region(gray, cfg)
            if debug is not None:
                debug["stage"] = "white_region"
                debug["white_mask"] = mask.copy() if mask is not None else None
                debug["white_quad"] = quad.copy() if quad is not None else None
        if quad is None:
            if debug is not None:
                debug["stage"] = "quad_failed"
                debug["fail_reason"] = "quad_not_found"
            return None

        quad_detected = quad.copy()
        quad_refined = refine_quad_local(gray, quad, cfg)
        if quad_refined is not None:
            quad = quad_refined
        if debug is not None:
            debug["quad_detected"] = quad_detected.copy()
            debug["quad_refined"] = quad.copy()

        quad_raw = quad.copy()
        quad = expand_quad(quad, cfg=cfg)

        rect, H = warp_by_quad(gray, quad, min_short=self.min_short, cfg=cfg)
        if debug is not None:
            debug["stage"] = "warp"
            debug["quad_expanded"] = quad.copy()
            debug["homography"] = H.copy()
            debug["rect"] = rect.copy()

        tile_grid = tuple(max(1, int(round(v))) for v in cfg.clahe_tile_grid)
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=tile_grid)
        rect_eq = clahe.apply(rect)
        blur_kx, blur_ky = cfg.rect_blur_kernel
        if blur_kx % 2 == 0: blur_kx += 1
        if blur_ky % 2 == 0: blur_ky += 1
        rect_pre = cv2.GaussianBlur(rect_eq, (int(blur_kx), int(blur_ky)), 0)
        if debug is not None:
            debug["stage"] = "rect_preprocess"
            debug["rect_clahe"] = rect_eq.copy()
            debug["rect_pre"] = rect_pre.copy()

        det = make_blob_detector(cfg)
        kps = det.detect(rect_pre)
        if debug is not None:
            debug["stage"] = "blob_detect"
            debug["keypoints_raw"] = [
                {
                    "x": float(kp.pt[0]),
                    "y": float(kp.pt[1]),
                    "size": float(kp.size),
                }
                for kp in kps
            ]
        if len(kps)==0:
            if debug is not None:
                debug["stage"] = "blob_detect_failed"
                debug["fail_reason"] = "no_blobs"
            return None

        sizes = [kp.size for kp in kps]
        if len(kps) >= 2:
            labels, centers = kmeans_1d_sizes(sizes)
        else:
            labels, centers = np.array([0]), np.array([sizes[0], sizes[0]])
        idx0 = np.where(labels==0)[0]; idx1 = np.where(labels==1)[0]
        small_idx, big_idx = (idx0, idx1) if len(idx0)>=len(idx1) else (idx1, idx0)

        small_refined=[]; big_refined=[]
        for i in small_idx:
            (x,y), r0 = kps[i].pt, kps[i].size * 0.5
            (cx,cy), r_est, tag, cnt, ell, refine_dbg = refine_center_on_patch(rect_pre, (x,y), r0, cfg)
            area = float(refine_dbg["area"]) if (refine_dbg and "area" in refine_dbg) else float(np.pi*r_est*r_est)
            small_refined.append(dict(x=float(cx), y=float(cy), r=float(r_est), tag=tag, area=area))
        for i in big_idx:
            (x,y), r0 = kps[i].pt, kps[i].size * 0.5
            (cx,cy), r_est, tag, cnt, ell, refine_dbg = refine_center_on_patch(rect_pre, (x,y), r0, cfg)
            area = float(refine_dbg["area"]) if (refine_dbg and "area" in refine_dbg) else float(np.pi*r_est*r_est)
            big_refined.append(dict(x=float(cx), y=float(cy), r=float(r_est), tag=tag, area=area))

        if debug is not None:
            debug["stage"] = "refine"
            debug["small_candidates"] = [dict(it) for it in small_refined]
            debug["big_candidates"] = [dict(it) for it in big_refined]

        all_refined = small_refined + big_refined
        if len(all_refined) >= 8 and (len(small_refined) < 30 or len(big_refined) < 2):
            areas = np.array([it["area"] for it in all_refined], np.float32)
            order = np.argsort(-areas)
            big_pick = [all_refined[i] for i in order[:min(6, len(all_refined))]]
            big_refined = pick_by_area_and_count(
                big_pick,
                4,
                area_range=None,
                relax=cfg.area_relax_reassign_big,
                label="big-re",
                cfg=cfg,
            )
            big_ids = set(id(it) for it in big_refined)
            small_refined = [it for it in all_refined if id(it) not in big_ids]

        small_refined = pick_by_area_and_count(
            small_refined, 41, area_range=None, relax=cfg.area_relax_small, label="small", cfg=cfg)
        big_refined   = pick_by_area_and_count(
            big_refined,   4, area_range=None, relax=cfg.area_relax_big, label="big", cfg=cfg)

        if debug is not None:
            debug["stage"] = "post_pick"
            debug["small_selected"] = [dict(it) for it in small_refined]
            debug["big_selected"] = [dict(it) for it in big_refined]
            debug["small_selected_count"] = len(small_refined)
            debug["big_selected_count"] = len(big_refined)
            if len(small_refined) < 41 and debug.get("fail_reason") is None:
                debug["fail_reason"] = "insufficient_small_after_pick"
            if len(big_refined) < 4 and debug.get("fail_reason") is None:
                debug["fail_reason"] = "insufficient_big_after_pick"

        if len(big_refined) == 4:
            big_pts = [(it["x"], it["y"]) for it in big_refined]
            O, x_hat, y_hat, _ = axes_from_big4(big_pts)
        else:
            O = np.array([rect_pre.shape[1]*0.5, rect_pre.shape[0]*0.5], np.float32)
            x_hat = np.array([1.0, 0.0], np.float32)
            y_hat = np.array([0.0, 1.0], np.float32)

        small_xy = [(it["x"], it["y"]) for it in small_refined]
        small_numbered, rows_pts = number_smalls_with_axes(small_xy, O, x_hat, y_hat)

        if debug is not None:
            debug["stage"] = "numbering"
            debug["axis_origin_rect"] = (float(O[0]), float(O[1]))
            debug["axis_x_hat"] = (float(x_hat[0]), float(x_hat[1]))
            debug["axis_y_hat"] = (float(y_hat[0]), float(y_hat[1]))
            debug["small_numbered"] = [dict(d) for d in small_numbered]
            debug["small_rows"] = rows_pts
            if len(small_numbered) < 41 and debug.get("fail_reason") is None:
                debug["fail_reason"] = "insufficient_small_after_numbering"

        if debug is not None:
            debug["stage"] = "done"

        br = BoardResult(
            quad=[tuple(map(float, p)) for p in quad],
            homography=np.asarray(H, np.float32).tolist(),
            big_circles=[Circle(**it) for it in big_refined],
            small_circles_rect=[Circle(**it) for it in small_refined],
            axis_origin_rect=(float(O[0]), float(O[1])),
            axis_x_hat=(float(x_hat[0]), float(x_hat[1])),
            axis_y_hat=(float(y_hat[0]), float(y_hat[1])),
            small_numbered=[SmallNumbered(**d) for d in small_numbered],
            meta={
                "quad_detected": [tuple(map(float, p)) for p in quad_detected],
                "quad_raw": [tuple(map(float, p)) for p in quad_raw],
                "quad_expand": {"scale": cfg.quad_expand_scale, "offset_px": cfg.quad_expand_offset},
            }
        )
        return br

# 便捷函数
def calibrate_from_path(img_path: str) -> Optional[BoardResult]:
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None: return None
    return Calibrator().process_gray(gray)