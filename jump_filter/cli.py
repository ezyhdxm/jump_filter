"""Explicit CSV/Parquet adapter; no data is loaded at import."""

# SETUP LOGIC: command-line and package imports.
import argparse
import json
from pathlib import Path
import pandas as pd
from .config import FilterConfig, METHODS
from .engine import filter_trades, summarize


def main():
    # CONFIGURATION LOGIC: make column mappings and units-related controls explicit.
    parser = argparse.ArgumentParser(description="Annotate bond spread trades without deleting rows")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cusip-col", default="CUSIP")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--spread-col", default="spread")
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--method", choices=METHODS, default="consensus")
    parser.add_argument("--backend", choices=("auto", "python", "numba"), default="auto",
                        help="auto uses optional native acceleration for large supported workloads")
    parser.add_argument("--config", type=Path, help="JSON FilterConfig overrides")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--session-schedule", type=Path, help="CSV with aware open/close session boundaries")
    arguments = parser.parse_args()
    options = json.loads(arguments.config.read_text()) if arguments.config else {}
    options.setdefault("method", arguments.method)

    # FILE IO LOGIC: preserve leading-zero CSV identifiers; parquet retains native schema.
    if arguments.input.suffix.lower() == ".parquet":
        frame = pd.read_parquet(arguments.input)
    else:
        frame = pd.read_csv(arguments.input, dtype={arguments.cusip_col: "string"})
    schedule = pd.read_csv(arguments.session_schedule) if arguments.session_schedule else None

    # Input: frame CUSIP=['A','A','A'],time=['2026-09-14T13:00Z','2026-09-14T14:00Z',
    # '2026-09-14T15:00Z'],spread=[100,130,100],options={method:'hampel',window:3,min_neighbors:2} ->
    # Output: original3rows plus jf_is_outlier=[False,True,False],jf_baseline=[115,100,115],
    # jf_weight=[1,1/30,1] (baselines/weights up to floating-point precision).
    # Trick: filtering annotates the original rows; file serialization happens only afterward.
    # CORE LOGIC: STEP 1
    annotated = filter_trades(frame, FilterConfig(**options), cusip_col=arguments.cusip_col,
                              time_col=arguments.time_col, spread_col=arguments.spread_col,
                              timezone=arguments.timezone, session_schedule=schedule, backend=arguments.backend)

    # FILE IO LOGIC: serialize original-order annotations and optional per-bond counts.
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.output.suffix.lower() == ".parquet":
        annotated.to_parquet(arguments.output, index=False)
    else:
        annotated.to_csv(arguments.output, index=False)
    if arguments.summary:
        arguments.summary.parent.mkdir(parents=True, exist_ok=True)
        summarize(annotated, cusip_col=arguments.cusip_col).to_csv(arguments.summary, index=False)


# SETUP LOGIC: module execution is explicit.
if __name__ == "__main__":
    main()
