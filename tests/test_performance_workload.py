"""Keep the scale benchmark workload exact, reproducible, and session aware."""

# SETUP LOGIC: Public workload construction and deterministic numerical assertions.
import numpy as np
import pandas as pd
import pytest

from benchmarks.performance import allocate_counts, make_workload


# TEST LOGIC: The requested capacity trial has precisely77 trades for each of13,000 CUSIPs.
def test_requested_million_trade_allocation_is_exact():
    counts = allocate_counts(1_001_000, 13_000)
    assert len(counts) == 13_000
    assert int(counts.sum()) == 1_001_000
    assert (counts == 77).all()


# TEST LOGIC: Dense histories remain a separate workload instead of disappearing in an average.
def test_skew_allocates_half_extra_rows_to_top_one_percent():
    counts = allocate_counts(1_001_000, 13_000, "skewed")
    assert int(counts.sum()) == 1_001_000
    assert (counts > 0).all()
    assert (counts[:130] == 3801).all()
    assert int(counts[:130].sum()) == 494130
    assert float(np.median(counts)) == 39.0
    assert int(counts.max()) > 90 * int(np.median(counts))


# TEST LOGIC: Determinism includes source order and deliberately repeated index labels.
def test_generated_workload_preserves_exact_records_and_duplicate_cohorts():
    frame = make_workload(1001, 13, seed=17)
    pd.testing.assert_frame_equal(frame, make_workload(1001, 13, seed=17))
    assert len(frame) == 1001 and frame["CUSIP"].nunique() == 13
    assert frame.index.has_duplicates
    assert int(frame.duplicated(["CUSIP", "time"]).sum()) == 52
    assert frame["spread"].map(np.isfinite).all()
    assert frame["CUSIP"].str.len().eq(9).all()
    assert not frame["time"].is_monotonic_increasing


# TEST LOGIC: Weekday hours and irregular event gaps are explicit; no weekend trades are generated.
def test_workload_contains_only_irregular_declared_open_session_events():
    frame = make_workload(1001, 13)
    local = frame["time"].dt.tz_convert("America/New_York")
    minutes = local.dt.hour * 60 + local.dt.minute
    assert local.dt.dayofweek.lt(5).all()
    assert minutes.ge(8 * 60).all() and minutes.lt(18 * 60 + 30).all()
    bond = frame.loc[frame["CUSIP"].eq("BEN000001")].sort_values("time")
    gaps = bond["time"].diff().dt.total_seconds()
    assert gaps.eq(0).any() and gaps.nunique() > 20
    assert gaps.gt(12 * 3600).any()


# TEST LOGIC: The generator cannot truthfully claim more distinct CUSIPs than source rows.
@pytest.mark.parametrize("rows,bonds,scenario", [(9, 10, "balanced"), (10, 0, "balanced"), (10, 2, "missing")])
def test_invalid_workload_fails_before_benchmark(rows, bonds, scenario):
    with pytest.raises(ValueError):
        allocate_counts(rows, bonds, scenario)
