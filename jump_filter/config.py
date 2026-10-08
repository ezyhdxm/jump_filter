"""Explicit, unit-preserving controls for bond spread screening."""

# SETUP LOGIC: standard-library imports have no data or display side effects.
from dataclasses import dataclass, asdict
import math
import pandas as pd

# CONFIGURATION LOGIC: these are implementations, not statistical probabilities.
METHODS = ("hampel", "rolling_iqr", "local_linear", "jump_reversion", "multiscale",
           "local_piecewise", "robust_trend", "consensus", "causal_ewma")


@dataclass(frozen=True)
class FilterConfig:
    """Window counts distinct timestamps; algorithm controls use input spread units.

    The optional absolute cap uses bp through explicit ``spread_units_per_bp``.
    """

    # CONFIGURATION LOGIC: retrospective defaults require future observations.
    method: str = "consensus"
    window: int = 31
    horizon: str = "3D"
    max_gap: str = "1D"
    min_neighbors: int = 6
    threshold: float = 4.5
    abs_floor: float = 1.0
    reversion_tolerance: float = 2.0
    alpha: float = 0.2
    persistence: int = 3
    iqr_multiplier: float = 3.0
    multiscale_votes: int = 2
    trend_penalty: float = 8.0
    huber_delta: float = 2.5
    max_iter: int = 2000
    tolerance: float = 1e-4
    time_basis: str = "wall"
    session_timezone: str = "America/New_York"
    session_open: str = "08:00"
    session_close: str = "18:30"
    holidays: tuple[str, ...] = ()
    quantity_rule: bool = False
    quantity_threshold: float = 1_000_000.0
    quantity_multiplier: float = 1.0
    max_deviation_rule: bool = False
    require_cap_support: bool = False
    max_deviation_bps: float = 10.0
    spread_units_per_bp: float = 1.0

    def __post_init__(self):
        # CONFIGURATION LOGIC: fail before processing on incompatible controls.
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        for name in ("quantity_rule", "max_deviation_rule", "require_cap_support"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("quantity_threshold", "quantity_multiplier", "max_deviation_bps", "spread_units_per_bp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("window", "min_neighbors", "persistence", "multiscale_votes", "max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.window < 3 or not 2 <= self.min_neighbors <= self.window:
            raise ValueError("window >= 3 and 2 <= min_neighbors <= window required")
        if self.persistence < 2:
            raise ValueError("persistence must be at least 2")
        if not 1 <= self.multiscale_votes <= 3 or self.max_iter < 1:
            raise ValueError("multiscale_votes must be 1..3 and max_iter >= 1")
        for name in ("threshold", "abs_floor", "reversion_tolerance", "alpha",
                     "iqr_multiplier", "trend_penalty", "huber_delta", "tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.alpha > 1:
            raise ValueError("alpha must be <= 1")
        if self.time_basis not in ("wall", "trading"):
            raise ValueError("time_basis must be 'wall' or 'trading'")
        from zoneinfo import ZoneInfo
        from datetime import date, time
        ZoneInfo(self.session_timezone)
        opening, closing = time.fromisoformat(self.session_open), time.fromisoformat(self.session_close)
        if opening.tzinfo or closing.tzinfo or opening >= closing:
            raise ValueError("regular session requires naive open < close within one local day")
        if not isinstance(self.holidays, tuple):
            object.__setattr__(self, "holidays", tuple(self.holidays))
        for holiday in self.holidays:
            if not isinstance(holiday, str) or date.fromisoformat(holiday).isoformat() != holiday:
                raise ValueError("holidays require complete YYYY-MM-DD local dates")
        for name in ("horizon", "max_gap"):
            duration = pd.Timedelta(getattr(self, name))
            if pd.isna(duration) or duration <= pd.Timedelta(0):
                raise ValueError(f"{name} must be a positive elapsed duration")

    def to_dict(self):
        # CONFIGURATION LOGIC: serialize an immutable configuration for exports.
        return asdict(self)
