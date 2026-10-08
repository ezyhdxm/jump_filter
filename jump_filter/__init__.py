"""Generic bond spread filtering; import does not read data or start a UI."""

# SETUP LOGIC: the lightweight API remains usable without dashboard extras.
from .config import FilterConfig, METHODS
from .engine import filter_trades, summarize
from .demo import make_demo

__version__ = "0.1.0"
__all__ = ["FilterConfig", "METHODS", "filter_trades", "summarize", "make_demo", "show_filter"]


def show_filter(frame, **kwargs):
    # UI LOGIC: optional notebook dependencies load only on explicit display.
    from .dashboard import show_filter as display
    return display(frame, **kwargs)
