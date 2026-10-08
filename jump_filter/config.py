"""Explicit, unit-preserving controls for bond spread screening."""

# SETUP LOGIC: standard-library imports have no data or display side effects.
from dataclasses import dataclass, asdict
import math
import pandas as pd

# CONFIGURATION LOGIC: these are implementations, not statistical probabilities.
METHODS = ("hampel", "local_linear", "jump_reversion", "causal_ewma", "consensus")


@dataclass(frozen=True)
class FilterConfig:
    """Window counts distinct timestamps; all spread controls use input units."""

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

    def __post_init__(self):
        # CONFIGURATION LOGIC: fail before processing on incompatible controls.
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        for name in ("window", "min_neighbors", "persistence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.window < 3 or not 2 <= self.min_neighbors <= self.window:
            raise ValueError("window >= 3 and 2 <= min_neighbors <= window required")
        if self.persistence < 2:
            raise ValueError("persistence must be at least 2")
        for name in ("threshold", "abs_floor", "reversion_tolerance", "alpha"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.alpha > 1:
            raise ValueError("alpha must be <= 1")
        for name in ("horizon", "max_gap"):
            duration = pd.Timedelta(getattr(self, name))
            if pd.isna(duration) or duration <= pd.Timedelta(0):
                raise ValueError(f"{name} must be a positive elapsed duration")

    def to_dict(self):
        # CONFIGURATION LOGIC: serialize an immutable configuration for exports.
        return asdict(self)
