"""Dataclasses describing configuration and analysis results for how_many."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict

import json


@dataclass
class AnalysisParams:
    """Runtime configuration for the periodicity analysis pipeline."""

    stripe_width_px: int = 20
    blur_sigma_px: float = 0.0
    auto_analyze: bool = False  # default OFF
    hide_during_capture_ms: int = 40
    suggest_max: int = 5
    min_length_px: int = 48


@dataclass
class UIState:
    """User-interface level preferences for the overlay."""

    manual_count: int = 10
    always_on_top: bool = True
    line_thickness: int = 3
    tick_length_px: int = 10


@dataclass
class Suggestion:
    """Represents an analysis candidate for the number of items."""

    count: int  # items (endpoints included)
    confidence: float  # 0..1
    source: str  # "fft", "acf", "harmonic", etc.


@dataclass
class AppConfig:
    """Persisted configuration for the application."""

    params: AnalysisParams = field(default_factory=AnalysisParams)
    ui: UIState = field(default_factory=UIState)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "AppConfig":
        data: Dict = json.loads(text)
        p = data.get("params", {})
        u = data.get("ui", {})
        return AppConfig(
            params=AnalysisParams(
                stripe_width_px=int(p.get("stripe_width_px", 20)),
                blur_sigma_px=float(p.get("blur_sigma_px", 0.0)),
                auto_analyze=bool(p.get("auto_analyze", False)),
                hide_during_capture_ms=int(p.get("hide_during_capture_ms", 40)),
                suggest_max=int(p.get("suggest_max", 5)),
                min_length_px=int(p.get("min_length_px", 48)),
            ),
            ui=UIState(
                manual_count=int(u.get("manual_count", 10)),
                always_on_top=bool(u.get("always_on_top", True)),
                line_thickness=int(u.get("line_thickness", 3)),
                tick_length_px=int(u.get("tick_length_px", 10)),
            ),
        )


__all__ = [
    "AnalysisParams",
    "UIState",
    "Suggestion",
    "AppConfig",
]
