# -*- coding: utf-8 -*-
"""Physical specification for the calibration board."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class BoardSpec:
    """Calibration board geometry expressed in millimeters."""

    small_diameter_mm: float
    center_spacing_mm: float

    def with_spacing(self, spacing_mm: float) -> "BoardSpec":
        """Return a copy with updated circle spacing."""
        return BoardSpec(small_diameter_mm=self.small_diameter_mm, center_spacing_mm=float(spacing_mm))

    @property
    def small_radius_mm(self) -> float:
        return self.small_diameter_mm * 0.5

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        data["small_radius_mm"] = self.small_radius_mm
        return data


DEFAULT_BOARD_SPEC = BoardSpec(small_diameter_mm=5.0, center_spacing_mm=25.0)

__all__ = ["BoardSpec", "DEFAULT_BOARD_SPEC"]
