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
    parser.add_argument("--config", type=Path, help="JSON FilterConfig overrides")
    parser.add_argument("--summary", type=Path)
    arguments = parser.parse_args()
    options = json.loads(arguments.config.read_text()) if arguments.config else {}
    options.setdefault("method", arguments.method)

    # FILE IO LOGIC: preserve leading-zero CSV identifiers; parquet retains native schema.
    if arguments.input.suffix.lower() == ".parquet":
        frame = pd.read_parquet(arguments.input)
    else:
        frame = pd.read_csv(arguments.input, dtype={arguments.cusip_col: "string"})
    annotated = filter_trades(frame, FilterConfig(**options), cusip_col=arguments.cusip_col,
                              time_col=arguments.time_col, spread_col=arguments.spread_col,
                              timezone=arguments.timezone)
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
