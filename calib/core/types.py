# calib/core/types.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

Pt = Tuple[float, float]


@dataclass
class Axis:
    origin_rect: Pt
    x_hat: Pt
    y_hat: Pt


@dataclass
class Circle:
    x: float
    y: float
    r: float
    tag: str = ""
    area: float = 0.0
    debug: Optional[Dict[str, Any]] = None


@dataclass
class SmallNumbered:
    seq: int
    row: int
    col: int
    x: float
    y: float
    u: float
    v: float


@dataclass
class BoardResult:
    # ==== 非默认字段（必须在前） ====
    quad: List[Pt]
    homography: List[List[float]]
    big_circles: List[Circle]
    small_circles_rect: List[Circle]
    axis_origin_rect: Pt
    axis_x_hat: Pt
    axis_y_hat: Pt
    small_numbered: List[SmallNumbered]

    # ==== 有默认值的可选字段（必须放在后） ====
    timings_ms: Dict[str, float] = field(default_factory=dict)  # 分阶段耗时（纯计算）
    meta: Dict[str, Any] = field(default_factory=dict)          # 其他附加信息