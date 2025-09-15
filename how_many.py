"""
how_many.py — overlay tool to estimate repeated structures along a line
-------------------------------------------------------------------------------
Summary
  • Always‑on‑top overlay draws a line with two draggable dots.
  • You position it over an image (semiconductor / industrial / biological, etc.).
  • It analyzes a configurable‑width stripe under the line and estimates the
    number of repeated items based on periodicity (FFT + autocorrelation).
  • Presents several candidate counts in a drop‑down; you can also set the count
    manually with up/down arrows. Selecting a candidate updates the manual box.
  • Draws tick markers along the line for visual confirmation.
  • Profile tab shows the 1‑D stripe profile and candidate results.
  • Hold **Ctrl** while dragging an endpoint to snap the line angle to 0°, 45°, 90°, 135°.
Defaults
  • Auto‑analyze is OFF by default (no analysis on move/drop unless enabled).
  • Tick markers always **include endpoints** (not configurable).
-------------------------------------------------------------------------------
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import sys
import json
import math

import numpy as np
import cv2  # opencv-python
from peaks_detrend_standalone import find_peaks, detrend

from PySide6 import QtCore, QtGui, QtWidgets


# ------------------------------ Data Models ----------------------------------


@dataclass
class AnalysisParams:
    stripe_width_px: int = 20
    blur_sigma_px: float = 0.0
    auto_analyze: bool = False  # default OFF
    hide_during_capture_ms: int = 40
    suggest_max: int = 5
    min_length_px: int = 48


@dataclass
class UIState:
    manual_count: int = 10
    always_on_top: bool = True
    line_thickness: int = 3
    tick_length_px: int = 10


@dataclass
class Suggestion:
    count: int  # items (endpoints included)
    confidence: float  # 0..1
    source: str  # "fft", "acf", "harmonic", etc.


@dataclass
class AppConfig:
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


# ----------------------------- Helper Utilities -------------------------------


def qpixmap_to_bgr(pix: QtGui.QPixmap) -> np.ndarray:
    """Convert a QPixmap to an OpenCV BGR numpy array (PySide6-safe)."""
    fmt = getattr(QtGui.QImage, "Format_RGBA8888", None)
    if fmt is None:
        fmt = QtGui.QImage.Format.Format_RGBA8888
    img: QtGui.QImage = pix.toImage().convertToFormat(fmt)
    width = img.width()
    height = img.height()
    bytes_per_line = img.bytesPerLine()
    buf = img.constBits()  # memoryview in PySide6
    arr = np.frombuffer(buf, np.uint8)
    arr = arr.reshape((height, bytes_per_line))  # include stride
    arr = arr[:, : width * 4]  # crop padding
    arr = arr.reshape((height, width, 4))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return bgr


def rotate_point(
    x: float, y: float, cx: float, cy: float, angle_deg: float
) -> Tuple[float, float]:
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x0 = x - cx
    y0 = y - cy
    xr = x0 * cos_t - y0 * sin_t + cx
    yr = x0 * sin_t + y0 * cos_t + cy
    return xr, yr


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ------------------------------ Core Analysis ---------------------------------


def _rank_and_cap(
    suggestions: Dict[int, Suggestion], profile_len: int, max_candidates: int
) -> List[Suggestion]:
    ranked = sorted(
        suggestions.values(),
        key=lambda s: (s.confidence, s.source.startswith("fft")),
        reverse=True,
    )
    filtered: List[Suggestion] = [
        s for s in ranked if 1 <= s.count <= max(1, profile_len)
    ]  # items can be up to length
    return filtered[: max(1, int(max_candidates))]


def estimate_counts_from_profile(
    profile: np.ndarray, max_candidates: int = 5
) -> List[Suggestion]:
    """
    Estimate number of ITEMS along the line (including endpoints) using FFT & autocorrelation.
    Internally, spectral/ACF give an estimate of CYCLES (intervals). We convert via: items = cycles + 1.
    """
    N = int(profile.size)
    if N < 16:
        return []

    # Detrend and normalize
    p = detrend(profile.astype(np.float64))
    m = float(np.max(np.abs(p)))
    if m > 1e-9:
        p /= m

    # Windowing to reduce spectral leakage
    win = np.hanning(N)
    p_win = p * win

    suggestions: Dict[int, Suggestion] = {}

    def add_items(cycles: int, conf: float, source: str) -> None:
        # Convert cycles (intervals) -> items (endpoints included)
        items = int(max(1, cycles + 1))
        prev = suggestions.get(items)
        if (prev is None) or (conf > prev.confidence):
            suggestions[items] = Suggestion(
                count=items, confidence=float(clamp(conf, 0.0, 1.0)), source=source
            )

    # ---------------- FFT ----------------
    fft_mag = np.abs(np.fft.rfft(p_win))
    fft_mag[0] = 0.0  # ignore DC

    spectral = fft_mag.copy()
    spectral[:2] = 0.0  # ignore ultra-low frequencies

    spec_max = float(np.max(spectral)) if np.max(spectral) > 0 else 0.0
    if spec_max > 0:
        prominence = float(spec_max * 0.15)
        peaks, _ = find_peaks(spectral, prominence=prominence)

        if peaks.size > 0:
            for k in peaks:
                cycles = int(k)  # DFT bin index ~ cycles across the sampled window
                conf = float(spectral[k] / spec_max)
                add_items(cycles, conf, "fft-peak")

            # Consider harmonics
            top_idxs = sorted(
                peaks.tolist(), key=lambda i: float(spectral[i]), reverse=True
            )[:3]
            for i in top_idxs:
                base = int(i)
                base_conf = float(spectral[i] / spec_max)
                add_items(base, base_conf, "fft-top")
                if 2 * base < len(spectral):
                    add_items(2 * base, base_conf * 0.6, "harmonic-x2")
                if base % 2 == 0 and base > 0:
                    add_items(base // 2, base_conf * 0.6, "harmonic-x0.5")

    # ---------------- ACF ----------------
    p0 = p - float(np.mean(p))
    acf_full = np.correlate(p0, p0, mode="full")
    acf = acf_full[N - 1 :]  # lags 0..N-1
    if acf[0] != 0:
        acf = acf / float(acf[0])
    acf[:2] = 0.0  # ignore lag 0/1

    if N >= 32 and np.max(acf) > 0:
        acf_peaks, _ = find_peaks(acf, prominence=float(np.max(acf) * 0.1))
        if acf_peaks.size > 0:
            acf_max = float(np.max(acf))
            for lag in acf_peaks[:5]:
                if lag > 0:
                    cycles = int(
                        round((N - 1) / float(lag))
                    )  # use N-1 (interval length) for cycles
                    conf = float(acf[lag] / acf_max)
                    add_items(cycles, conf, "acf-peak")

    return _rank_and_cap(suggestions, N, max_candidates)


# ------------------------------ Overlay Widget --------------------------------


class OverlayWidget(QtWidgets.QWidget):
    lineChanged = QtCore.Signal()
    requestAnalyze = QtCore.Signal()

    def __init__(self, virtual_rect: QtCore.QRect, cfg: AppConfig) -> None:
        super().__init__(
            None, QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowFlag(
            QtCore.Qt.WindowType.WindowStaysOnTopHint, cfg.ui.always_on_top
        )
        self.setGeometry(virtual_rect)
        self.setMouseTracking(True)

        self._p1 = QtCore.QPointF(
            virtual_rect.center().x() - 200, virtual_rect.center().y()
        )
        self._p2 = QtCore.QPointF(
            virtual_rect.center().x() + 200, virtual_rect.center().y()
        )
        self._drag_target: Optional[str] = None  # "p1" | "p2" | "line" | None
        self._drag_offset = QtCore.QPointF(0, 0)

        self._tick_count: int = max(1, cfg.ui.manual_count)
        self._stripe_width: int = max(2, cfg.params.stripe_width_px)
        self._line_thickness: int = max(1, cfg.ui.line_thickness)
        self._tick_length: int = max(4, cfg.ui.tick_length_px)
        self._auto_enabled: bool = bool(cfg.params.auto_analyze)

        # Auto-analyze throttle
        self._auto_timer = QtCore.QTimer(self)
        self._auto_timer.setInterval(200)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.timeout.connect(self.requestAnalyze)

    # ----------------------------- Properties ---------------------------------

    @property
    def p1(self) -> QtCore.QPointF:
        return self._p1

    @property
    def p2(self) -> QtCore.QPointF:
        return self._p2

    def set_tick_count(self, n: int) -> None:
        self._tick_count = max(1, int(n))
        self.update()

    def set_stripe_width(self, w: int, auto_analyze: bool = False) -> None:
        self._stripe_width = max(2, int(w))
        self.update()
        if auto_analyze and self._auto_enabled:
            self._auto_timer.start()

    def set_line_thickness(self, t: int) -> None:
        self._line_thickness = max(1, int(t))
        self.update()

    def set_tick_length(self, t: int) -> None:
        self._tick_length = max(2, int(t))
        self.update()

    def set_auto_analyze(self, enabled: bool) -> None:
        self._auto_enabled = bool(enabled)

    def line_length(self) -> float:
        return float(QtCore.QLineF(self._p1, self._p2).length())

    def stripe_width(self) -> int:
        return int(self._stripe_width)

    def stripe_bbox_virtual(self, margin: int = 4) -> QtCore.QRect:
        """Bounding box in the virtual desktop coordinate system."""
        x1, y1 = self._p1.x(), self._p1.y()
        x2, y2 = self._p2.x(), self._p2.y()
        half = self._stripe_width / 2.0 + margin
        left = math.floor(min(x1, x2) - half)
        right = math.ceil(max(x1, x2) + half)
        top = math.floor(min(y1, y2) - half)
        bottom = math.ceil(max(y1, y2) + half)
        return QtCore.QRect(left, top, right - left, bottom - top)

    # ----------------------------- Interaction --------------------------------

    def _ctrl_down(self) -> bool:
        mods = QtWidgets.QApplication.keyboardModifiers()
        return bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)

    def _snap_endpoint(
        self, target: QtCore.QPointF, anchor: QtCore.QPointF
    ) -> QtCore.QPointF:
        """If Ctrl is held, snap the angle anchor->target to multiples of 45°; preserve length."""
        if not self._ctrl_down():
            return target
        dx = float(target.x() - anchor.x())
        dy = float(target.y() - anchor.y())
        L = math.hypot(dx, dy)
        if L <= 1e-6:
            return target
        angle = math.atan2(dy, dx)
        step = math.pi / 4.0  # 45°
        snapped = round(angle / step) * step
        nx = math.cos(snapped)
        ny = math.sin(snapped)
        newx = float(anchor.x()) + nx * L
        newy = float(anchor.y()) + ny * L
        newx = clamp(
            newx, float(self.rect().left() + 2), float(self.rect().right() - 2)
        )
        newy = clamp(
            newy, float(self.rect().top() + 2), float(self.rect().bottom() - 2)
        )
        return QtCore.QPointF(newx, newy)

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        pos = e.position()
        if self._near(self._p1, pos, 12):
            self._drag_target = "p1"
        elif self._near(self._p2, pos, 12):
            self._drag_target = "p2"
        elif self._near_line(pos, 10):
            self._drag_target = "line"
            self._drag_offset = pos - self._p1
        else:
            self._drag_target = None
        e.accept()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        pos = e.position()
        if self._drag_target == "p1":
            newp = self._clamp_point(pos)
            newp = self._snap_endpoint(newp, self._p2)
            self._p1 = newp
            self.lineChanged.emit()
            self.update()
        elif self._drag_target == "p2":
            newp = self._clamp_point(pos)
            newp = self._snap_endpoint(newp, self._p1)
            self._p2 = newp
            self.lineChanged.emit()
            self.update()
        elif self._drag_target == "line":
            delta = pos - self._drag_offset - self._p1
            self._p1 = self._clamp_point(self._p1 + delta)
            self._p2 = self._clamp_point(self._p2 + delta)
            self.lineChanged.emit()
            self.update()
        else:
            if self._near(self._p1, pos, 12) or self._near(self._p2, pos, 12):
                self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            elif self._near_line(pos, 10):
                self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        self._drag_target = None
        e.accept()
        if self._auto_enabled:
            self._auto_timer.start()

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        key = e.key()
        if key in (QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Equal):
            self.set_tick_count(self._tick_count + 1)
        elif key in (QtCore.Qt.Key.Key_Minus, QtCore.Qt.Key.Key_Underscore):
            self.set_tick_count(max(1, self._tick_count - 1))
        elif key == QtCore.Qt.Key.Key_A:
            self.requestAnalyze.emit()
        elif key == QtCore.Qt.Key.Key_W:
            self.set_stripe_width(self._stripe_width + 1, auto_analyze=True)
        elif key == QtCore.Qt.Key.Key_Q:
            self.set_stripe_width(max(2, self._stripe_width - 1), auto_analyze=True)
        elif key == QtCore.Qt.Key.Key_Escape:
            self.close()

    # ----------------------------- Painting -----------------------------------

    def paintEvent(self, e: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 0))

        line = QtCore.QLineF(self._p1, self._p2)
        if line.length() < 1.0:
            return

        dx = line.dx()
        dy = line.dy()
        L = math.hypot(dx, dy)
        nx = -dy / L
        ny = dx / L
        half_w = self._stripe_width / 2.0

        p1_top = QtCore.QPointF(self._p1.x() + nx * half_w, self._p1.y() + ny * half_w)
        p1_bot = QtCore.QPointF(self._p1.x() - nx * half_w, self._p1.y() - ny * half_w)
        p2_top = QtCore.QPointF(self._p2.x() + nx * half_w, self._p2.y() + ny * half_w)
        p2_bot = QtCore.QPointF(self._p2.x() - nx * half_w, self._p2.y() - ny * half_w)

        poly = QtGui.QPolygonF([p1_top, p2_top, p2_bot, p1_bot])

        stripe_brush = QtGui.QBrush(QtGui.QColor(0, 170, 255, 40))
        stripe_pen = QtGui.QPen(QtGui.QColor(0, 170, 255, 160))
        stripe_pen.setWidth(1)
        painter.setBrush(stripe_brush)
        painter.setPen(stripe_pen)
        painter.drawPolygon(poly)

        line_pen = QtGui.QPen(QtGui.QColor(0, 255, 180, 200))
        line_pen.setWidth(self._line_thickness)
        painter.setPen(line_pen)
        painter.drawLine(line)

        # Tick marks (always include endpoints)
        m = self._tick_count
        if m >= 1:
            t_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220))
            t_pen.setWidth(2)
            painter.setPen(t_pen)
            if m > 1:
                for i in range(m):
                    t = i / float(m - 1)
                    px = self._p1.x() + dx * t
                    py = self._p1.y() + dy * t
                    x1 = px - nx * (self._tick_length / 2.0)
                    y1 = py - ny * (self._tick_length / 2.0)
                    x2 = px + nx * (self._tick_length / 2.0)
                    y2 = py + ny * (self._tick_length / 2.0)
                    painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))
            else:
                # m == 1: draw single tick at the start endpoint
                px = self._p1.x()
                py = self._p1.y()
                x1 = px - nx * (self._tick_length / 2.0)
                y1 = py - ny * (self._tick_length / 2.0)
                x2 = px + nx * (self._tick_length / 2.0)
                y2 = py + ny * (self._tick_length / 2.0)
                painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))

        # End dots
        dot_brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 240))
        dot_pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 200))
        dot_pen.setWidth(1)
        painter.setBrush(dot_brush)
        painter.setPen(dot_pen)
        r = 6
        painter.drawEllipse(self._p1, r, r)
        painter.drawEllipse(self._p2, r, r)

        # HUD: position along the outward normal above the stripe to avoid overlap
        # Anchor at segment midpoint, offset outward by (half stripe + tick length + margin)
        margin = 18.0
        off = half_w + float(self._tick_length) + margin
        midx = (self._p1.x() + self._p2.x()) / 2.0
        midy = (self._p1.y() + self._p2.y()) / 2.0
        hud_cx = midx + nx * off
        hud_cy = midy + ny * off

        hud_w = 340.0
        hud_h = 24.0
        hud_rect = QtCore.QRectF(
            hud_cx - hud_w / 2.0, hud_cy - hud_h / 2.0, hud_w, hud_h
        )

        hud = (
            f"W={int(self._stripe_width)} px  |  L={int(L)} px  |  N={self._tick_count}"
        )
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 180)))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 200)))
        painter.drawRoundedRect(hud_rect, 6, 6)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 255)))
        painter.drawText(hud_rect, QtCore.Qt.AlignmentFlag.AlignCenter, hud)

    # --------------------------- Geometry helpers ------------------------------

    def _clamp_point(self, p: QtCore.QPointF) -> QtCore.QPointF:
        x = clamp(p.x(), float(self.rect().left() + 2), float(self.rect().right() - 2))
        y = clamp(p.y(), float(self.rect().top() + 2), float(self.rect().bottom() - 2))
        return QtCore.QPointF(x, y)

    @staticmethod
    def _near(a: QtCore.QPointF, b: QtCore.QPointF, tol: float) -> bool:
        return QtCore.QLineF(a, b).length() <= tol

    def _near_line(self, p: QtCore.QPointF, tol: float) -> bool:
        # Manual distance to line segment (works across PySide6 versions)
        x1, y1 = float(self._p1.x()), float(self._p1.y())
        x2, y2 = float(self._p2.x()), float(self._p2.y())
        px, py = float(p.x()), float(p.y())
        dx, dy = x2 - x1, y2 - y1
        seg_len2 = dx * dx + dy * dy
        if seg_len2 <= 1e-9:
            d2 = (px - x1) * (px - x1) + (py - y1) * (py - y1)
            return d2 <= float(tol * tol)
        t = ((px - x1) * dx + (py - y1) * dy) / seg_len2
        t = max(0.0, min(1.0, t))
        projx = x1 + t * dx
        projy = y1 + t * dy
        d2 = (px - projx) * (px - projx) + (py - projy) * (py - projy)
        return d2 <= float(tol * tol)


# ------------------------------- Profile Plot ---------------------------------


class ProfilePlot(QtWidgets.QWidget):
    """Simple QWidget that draws the latest 1-D stripe profile and markers (endpoints included)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._profile = np.empty((0,), dtype=np.float64)
        self._markers = 0
        self.setMinimumHeight(180)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def set_profile(self, profile: Optional[np.ndarray]) -> None:
        if profile is None or getattr(profile, "size", 0) == 0:
            self._profile = np.empty((0,), dtype=np.float64)
        else:
            self._profile = np.asarray(profile, dtype=np.float64).copy()
        self.update()

    def set_markers(self, n: int) -> None:
        self._markers = max(0, int(n))
        self.update()

    def paintEvent(self, e: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(8, 8, -8, -8)
        painter.fillRect(self.rect(), QtGui.QColor(250, 250, 250))
        painter.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220)))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        if self._profile.size == 0:
            painter.setPen(QtGui.QPen(QtGui.QColor(120, 120, 120)))
            painter.drawText(
                rect,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                "No profile yet.\nPress ‘Analyze Now’ to capture stripe profile.",
            )
            return

        # Normalize profile to [0,1] for plotting
        p = self._profile.astype(float)
        p_min = float(np.min(p))
        p_max = float(np.max(p))
        rng = p_max - p_min if (p_max - p_min) > 1e-12 else 1.0
        p_norm = (p - p_min) / rng

        N = p_norm.size
        left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
        w = max(1.0, float(right - left))
        h = max(1.0, float(bottom - top))

        # Horizontal grid
        painter.setPen(QtGui.QPen(QtGui.QColor(235, 235, 235)))
        for g in range(1, 4):
            y = top + h * (g / 4.0)
            painter.drawLine(QtCore.QPointF(left, y), QtCore.QPointF(right, y))

        # Draw waveform
        path = QtGui.QPainterPath()

        def x_at(i: int) -> float:
            if N <= 1:
                return left
            return left + (w - 1.0) * (i / float(N - 1))

        def y_at(v: float) -> float:
            return bottom - v * (h - 1.0)

        path.moveTo(x_at(0), y_at(p_norm[0]))
        for i in range(1, N):
            path.lineTo(x_at(i), y_at(p_norm[i]))

        painter.setPen(QtGui.QPen(QtGui.QColor(0, 120, 215), 2))
        painter.drawPath(path)

        # Draw markers: always include endpoints
        m = self._markers
        if m > 0:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 153, 0, 200), 1))
            if m > 1:
                for i in range(m):
                    t = i / float(m - 1)
                    x = left + t * (w - 1.0)
                    painter.drawLine(QtCore.QPointF(x, top), QtCore.QPointF(x, bottom))
            else:
                x = left  # single marker at start
                painter.drawLine(QtCore.QPointF(x, top), QtCore.QPointF(x, bottom))

        # Axes labels (minimal)
        painter.setPen(QtGui.QPen(QtGui.QColor(80, 80, 80)))
        text = f"Length: {N} px   min={p_min:.1f}  max={p_max:.1f}"
        painter.drawText(
            self.rect().adjusted(12, 12, -12, -12),
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft,
            text,
        )


# ---------------------------- Control Dialog UI -------------------------------


class ControlDialog(QtWidgets.QDialog):
    analyzeRequested = QtCore.Signal()
    manualCountChanged = QtCore.Signal(int)
    stripeWidthChanged = QtCore.Signal(int)
    lineThicknessChanged = QtCore.Signal(int)
    tickLengthChanged = QtCore.Signal(int)
    autoAnalyzeToggled = QtCore.Signal(bool)
    alwaysOnTopToggled = QtCore.Signal(bool)
    suggestionChosen = QtCore.Signal(int)

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(None)
        self.setWindowTitle("how_many — Stripe Periodicity Estimator")
        self.setWindowFlag(
            QtCore.Qt.WindowType.WindowStaysOnTopHint, cfg.ui.always_on_top
        )
        self.setMinimumWidth(560)

        # Widgets
        self.suggest_combo = QtWidgets.QComboBox()
        self.suggest_combo.setToolTip("Estimated counts from the signal analysis")
        self.suggest_combo.currentIndexChanged.connect(self._on_suggestion_index)

        self.manual_spin = QtWidgets.QSpinBox()
        self.manual_spin.setRange(1, 10000)
        self.manual_spin.setValue(cfg.ui.manual_count)
        self.manual_spin.valueChanged.connect(self.manualCountChanged)

        self.analyze_btn = QtWidgets.QPushButton("Analyze Now  (A)")
        self.analyze_btn.clicked.connect(self.analyzeRequested)

        self.auto_check = QtWidgets.QCheckBox("Enable")
        self.auto_check.setChecked(cfg.params.auto_analyze)
        self.auto_check.toggled.connect(self.autoAnalyzeToggled)

        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(2, 512)
        self.width_spin.setValue(cfg.params.stripe_width_px)
        self.width_spin.valueChanged.connect(self._on_width_change)

        self.thick_spin = QtWidgets.QSpinBox()
        self.thick_spin.setRange(1, 12)
        self.thick_spin.setValue(cfg.ui.line_thickness)
        self.thick_spin.valueChanged.connect(self.lineThicknessChanged)

        self.ticklen_spin = QtWidgets.QSpinBox()
        self.ticklen_spin.setRange(4, 64)
        self.ticklen_spin.setValue(cfg.ui.tick_length_px)
        self.ticklen_spin.valueChanged.connect(self.tickLengthChanged)

        self.topmost_check = QtWidgets.QCheckBox("Enable")
        self.topmost_check.setChecked(cfg.ui.always_on_top)
        self.topmost_check.toggled.connect(self.alwaysOnTopToggled)

        self.status_label = QtWidgets.QLabel(
            "Move the two dots to span the feature row, then Analyze."
        )
        self.status_label.setWordWrap(True)

        # Tabs
        self.tabs = QtWidgets.QTabWidget(self)

        # --- Controls tab
        controls_page = QtWidgets.QWidget(self)
        controls_form = QtWidgets.QFormLayout(controls_page)
        controls_form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        controls_form.addRow("Estimated numbers:", self.suggest_combo)
        controls_form.addRow("Manual number of items:", self.manual_spin)
        controls_form.addRow(self.analyze_btn)  # spans both columns
        controls_form.addRow("Auto‑analyze:", self.auto_check)  # aligned checkbox row
        controls_form.addRow("Stripe width (px):", self.width_spin)
        controls_form.addRow("Line thickness:", self.thick_spin)
        controls_form.addRow("Tick length (px):", self.ticklen_spin)
        controls_form.addRow(
            "Always on top:", self.topmost_check
        )  # aligned checkbox row
        controls_v = QtWidgets.QVBoxLayout()
        controls_v.addLayout(controls_form)
        controls_v.addWidget(self.status_label)
        controls_page.setLayout(controls_v)
        self.tabs.addTab(controls_page, "Controls")

        # --- Profile tab
        profile_page = QtWidgets.QWidget(self)
        self.profile_plot = ProfilePlot(profile_page)
        self.results_label = QtWidgets.QTextEdit(profile_page)
        self.results_label.setReadOnly(True)
        self.results_label.setMinimumHeight(80)
        self.results_label.setPlaceholderText(
            "Results will appear here after analysis."
        )
        profile_v = QtWidgets.QVBoxLayout(profile_page)
        profile_v.addWidget(self.profile_plot, stretch=1)
        profile_v.addWidget(self.results_label, stretch=0)
        profile_page.setLayout(profile_v)
        self.tabs.addTab(profile_page, "Profile")

        # --- Help tab
        help_page = QtWidgets.QScrollArea(self)
        help_page.setWidgetResizable(True)
        help_body = QtWidgets.QWidget()
        help_layout = QtWidgets.QVBoxLayout(help_body)
        help_text = QtWidgets.QLabel(self._help_markdown(), help_body)
        help_text.setTextFormat(QtCore.Qt.TextFormat.RichText)
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        help_layout.addStretch(1)
        help_page.setWidget(help_body)
        self.tabs.addTab(help_page, "Help")

        # Dialog layout
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(self.tabs)
        self.tabs.setCurrentIndex(0)  # Controls by default

    # --- helpers ---
    def _help_markdown(self) -> str:
        return (
            "<h3>how_many — quick reference</h3>"
            "<ul>"
            "<li><b>Position</b> the two dots across the repeating row.</li>"
            "<li>Adjust <b>Stripe width</b> so the blue rectangle covers the features.</li>"
            "<li>Click <b>Analyze Now</b> (or press <b>A</b>) to compute candidates.</li>"
            "<li>Use <b>Estimated numbers</b> or the <b>Manual</b> spinner to pick the count.</li>"
            "<li>Markers always <b>include endpoints</b>.</li>"
            "<li>Hold <b>Ctrl</b> while dragging an endpoint to snap to 0°/45°/90°/135°.</li>"
            "</ul>"
            "<h4>Shortcuts</h4>"
            "<ul>"
            "<li><b>A</b> — Analyze now</li>"
            "<li><b>+</b>/<b>-</b> — Increase/decrease item count</li>"
            "<li><b>W</b>/<b>Q</b> — Increase/decrease stripe width</li>"
            "<li><b>Esc</b> — Close overlay</li>"
            "</ul>"
        )

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_suggestions(self, suggestions: List[Suggestion]) -> None:
        self.suggest_combo.blockSignals(True)
        self.suggest_combo.clear()
        for s in suggestions:
            self.suggest_combo.addItem(
                f"{s.count} items  ∘  confidence {s.confidence:.2f}  [{s.source}]",
                userData=s.count,
            )
        self.suggest_combo.blockSignals(False)
        if self.suggest_combo.count() > 0:
            self.suggest_combo.setCurrentIndex(0)
            self.suggestionChosen.emit(int(self.suggest_combo.currentData()))

    def _on_suggestion_index(self, idx: int) -> None:
        if idx >= 0:
            val = int(self.suggest_combo.itemData(idx))
            self.suggestionChosen.emit(val)

    def _on_width_change(self, val: int) -> None:
        self.stripeWidthChanged.emit(val)

    # --- profile helpers ---
    def set_profile(self, profile: Optional[np.ndarray]) -> None:
        if hasattr(self, "profile_plot"):
            self.profile_plot.set_profile(profile)

    def set_results_text(self, text: str) -> None:
        if hasattr(self, "results_label"):
            self.results_label.setPlainText(text or "")

    def set_markers(self, n: int) -> None:
        if hasattr(self, "profile_plot"):
            self.profile_plot.set_markers(n)


# ---------------------------- Main Controller ---------------------------------


class MainController(QtCore.QObject):
    def __init__(self, app: QtWidgets.QApplication) -> None:
        super().__init__(None)
        self.app = app
        self.cfg = self._load_config()

        # Compute virtual desktop rect (union of all screens)
        virt_rect = self._virtual_desktop_rect()

        # Windows
        self.overlay = OverlayWidget(virt_rect, self.cfg)
        self.ctrl = ControlDialog(self.cfg)

        # Wire signals
        self.ctrl.analyzeRequested.connect(self.analyze)
        self.ctrl.manualCountChanged.connect(self.overlay.set_tick_count)
        self.ctrl.manualCountChanged.connect(lambda n: self.ctrl.set_markers(int(n)))
        self.ctrl.stripeWidthChanged.connect(
            lambda v: self.overlay.set_stripe_width(
                v, auto_analyze=self.cfg.params.auto_analyze
            )
        )
        self.ctrl.lineThicknessChanged.connect(self.overlay.set_line_thickness)
        self.ctrl.tickLengthChanged.connect(self.overlay.set_tick_length)
        self.ctrl.autoAnalyzeToggled.connect(self._on_auto_toggle)
        self.ctrl.alwaysOnTopToggled.connect(self._on_topmost_toggle)
        self.ctrl.suggestionChosen.connect(self._on_suggestion_chosen)

        self.overlay.requestAnalyze.connect(self.analyze)
        self.overlay.lineChanged.connect(self._on_line_changed)

        # Apply initial values
        self.overlay.set_tick_count(self.cfg.ui.manual_count)
        self.overlay.set_stripe_width(self.cfg.params.stripe_width_px)
        self.overlay.set_line_thickness(self.cfg.ui.line_thickness)
        self.overlay.set_tick_length(self.cfg.ui.tick_length_px)
        self.overlay.set_auto_analyze(self.cfg.params.auto_analyze)
        self.ctrl.set_markers(self.cfg.ui.manual_count)

        # Show windows
        self.overlay.show()
        self.ctrl.show()

        # Place control away from line
        self.ctrl.move(int(virt_rect.left() + 80), int(virt_rect.top() + 80))

    # ---------------------------- Config I/O ----------------------------------

    def _config_path(self) -> Path:
        home = Path.home()
        return home / ".how_many_config.json"

    def _load_config(self) -> AppConfig:
        p = self._config_path()
        if p.exists():
            try:
                return AppConfig.from_json(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return AppConfig()

    def _save_config(self) -> None:
        p = self._config_path()
        try:
            p.write_text(self.cfg.to_json(), encoding="utf-8")
        except Exception:
            pass

    # ---------------------------- Event Handlers ------------------------------

    def _on_line_changed(self) -> None:
        # No-op by default. Auto-analysis happens only via overlay's timer when enabled.
        pass

    def _on_auto_toggle(self, enabled: bool) -> None:
        self.cfg.params.auto_analyze = bool(enabled)
        self.overlay.set_auto_analyze(self.cfg.params.auto_analyze)
        self._save_config()

    def _on_topmost_toggle(self, enabled: bool) -> None:
        self.cfg.ui.always_on_top = bool(enabled)
        self.overlay.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, enabled)
        self.ctrl.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, enabled)
        self.overlay.show()
        self.ctrl.show()
        self._save_config()

    def _on_suggestion_chosen(self, count: int) -> None:
        self.ctrl.manual_spin.blockSignals(True)
        self.ctrl.manual_spin.setValue(int(count))
        self.ctrl.manual_spin.blockSignals(False)
        self.overlay.set_tick_count(int(count))
        self.ctrl.set_markers(int(count))

    # ----------------------------- Core Actions --------------------------------

    def analyze(self) -> None:
        """Capture stripe under the overlay line, estimate counts, update UI."""
        L = self.overlay.line_length()
        if L < self.cfg.params.min_length_px:
            self.ctrl.set_status(
                f"Line too short for analysis (< {self.cfg.params.min_length_px} px)."
            )
            return

        bbox_virtual = self.overlay.stripe_bbox_virtual(margin=8)

        # Hide overlay & control briefly to avoid capturing them
        self.overlay.setVisible(False)
        self.ctrl.setVisible(False)
        self.app.processEvents()
        QtCore.QThread.msleep(int(max(0, self.cfg.params.hide_during_capture_ms)))

        # Grab desktop
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            self.ctrl.set_status("No screen available for capture.")
            self.overlay.setVisible(True)
            self.ctrl.setVisible(True)
            return

        full_pix = screen.grabWindow(0)
        full_bgr = qpixmap_to_bgr(full_pix)

        # Show UI back quickly
        self.overlay.setVisible(True)
        self.ctrl.setVisible(True)
        self.app.processEvents()

        virt_rect = self._virtual_desktop_rect()
        x0 = max(0, bbox_virtual.left() - virt_rect.left())
        y0 = max(0, bbox_virtual.top() - virt_rect.top())
        x1 = min(virt_rect.width(), bbox_virtual.right() - virt_rect.left())
        y1 = min(virt_rect.height(), bbox_virtual.bottom() - virt_rect.top())
        w = max(1, int(x1 - x0))
        h = max(1, int(y1 - y0))

        if w < 4 or h < 4:
            self.ctrl.set_status("Stripe region is out of bounds.")
            return

        roi = full_bgr[int(y0) : int(y0 + h), int(x0) : int(x0 + w), :]

        # Map p1/p2 to ROI-local coords
        p1_v = self.overlay.p1
        p2_v = self.overlay.p2
        p1_local = (p1_v.x() - virt_rect.left() - x0, p1_v.y() - virt_rect.top() - y0)
        p2_local = (p2_v.x() - virt_rect.left() - x0, p2_v.y() - virt_rect.top() - y0)

        stripe = self._extract_aligned_stripe(
            roi, p1_local, p2_local, self.cfg.params.stripe_width_px
        )

        if stripe is None:
            self.ctrl.set_status("Failed to extract stripe (out of bounds).")
            return

        gray = cv2.cvtColor(stripe, cv2.COLOR_BGR2GRAY)
        if self.cfg.params.blur_sigma_px > 0.1:
            ksize = max(3, int(2 * round(3 * self.cfg.params.blur_sigma_px) + 1))
            gray = cv2.GaussianBlur(gray, (ksize, ksize), self.cfg.params.blur_sigma_px)

        # Average across width to get 1D profile
        profile = np.mean(gray, axis=0)

        suggestions = estimate_counts_from_profile(
            profile, max_candidates=self.cfg.params.suggest_max
        )
        if not suggestions:
            self.ctrl.set_status(
                "No strong periodicity found. Try adjusting stripe width or line position."
            )
            # Still show profile so user can see signal
            self.ctrl.set_profile(profile)
            self.ctrl.set_results_text("No strong periodicity found.")
            return

        # Update UI with results & profile
        self.ctrl.set_profile(profile)
        lines = ["Candidates (items, confidence, source):"]
        for s in suggestions:
            lines.append(f"  • {s.count:>4}    {s.confidence:0.3f}    {s.source}")
        self.ctrl.set_results_text("\n".join(lines))

        self.ctrl.set_suggestions(suggestions)
        best = suggestions[0]
        self.overlay.set_tick_count(int(best.count))
        self.ctrl.set_markers(int(best.count))
        self.ctrl.set_status(
            f"Best estimate: {best.count} items (confidence {best.confidence:.2f})."
        )

    def _extract_aligned_stripe(
        self,
        roi_bgr: np.ndarray,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        stripe_w: int,
    ) -> Optional[np.ndarray]:
        """Return rotated stripe image (H=stripe_w, W=line_length_px)."""
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx = x2 - x1
        dy = y2 - y1
        L = math.hypot(dx, dy)
        if L < 4.0:
            return None

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        angle_deg = -math.degrees(math.atan2(dy, dx))

        h, w = roi_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rotated = cv2.warpAffine(
            roi_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

        x1r, y1r = rotate_point(x1, y1, cx, cy, angle_deg)
        x2r, y2r = rotate_point(x2, y2, cx, cy, angle_deg)

        x_start = int(max(0, math.floor(min(x1r, x2r))))
        x_end = int(min(w, math.ceil(max(x1r, x2r))))
        y_center = int(round((y1r + y2r) / 2.0))
        half = int(max(1, round(stripe_w / 2)))
        y_top = int(max(0, y_center - half))
        y_bot = int(min(h, y_center + half))

        if x_end - x_start < 4 or y_bot - y_top < 2:
            return None

        stripe = rotated[y_top:y_bot, x_start:x_end].copy()
        return stripe

    # ----------------------------- System utils --------------------------------

    def _virtual_desktop_rect(self) -> QtCore.QRect:
        rect = QtCore.QRect(0, 0, 0, 0)
        for s in QtGui.QGuiApplication.screens():
            rect = rect.united(s.geometry())
        return rect


# ---------------------------------- Main --------------------------------------


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("how_many")

    ctrl = MainController(app)
    ret = app.exec()

    try:
        ctrl._save_config()
    except Exception:
        pass

    sys.exit(ret)


if __name__ == "__main__":
    main()
