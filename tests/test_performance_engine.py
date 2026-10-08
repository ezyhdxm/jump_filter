"""Differential contracts for bulk preparation and optional compiled screening."""

# SETUP LOGIC: tests use the retained numerical reference, not duplicate implementations.
import numpy as np
import pandas as pd
import pytest
from jump_filter import FilterConfig, METHODS, filter_trades, make_demo
from jump_filter import engine


def _legacy_groups(result, data, utc_times, config, accelerator):
    # TEST LOGIC: retain the previous pandas grouping path as an independent preparation oracle.
    for _, group in data.groupby("id", sort=False):
        engine._group_gaps(result, group, utc_times)
        segments = group["stamp"].diff().gt(pd.Timedelta(config.max_gap).value).cumsum()
        for _, segment in group.groupby(segments, sort=False):
            engine._segment(result, segment, config)


def _assert_audit_parity(actual, expected):
    # TEST LOGIC: statuses, flags, counts, weights and all clock/row identities are compared separately.
    assert list(actual.columns) == list(expected.columns)
    for name in actual.columns:
        if name.startswith("jf_") and pd.api.types.is_float_dtype(actual[name]):
            np.testing.assert_allclose(actual[name], expected[name], rtol=2e-9, atol=2e-8, equal_nan=True, err_msg=name)
        else:
            pd.testing.assert_series_equal(actual[name], expected[name])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("time_basis", ("wall", "trading"))
def test_bulk_array_preparation_matches_previous_pandas_path(monkeypatch, method, time_basis):
    # TEST LOGIC: demo includes reversals, volatility/liquidity changes, gaps, sparse bonds and duplicate index labels.
    source = make_demo(seed=103).sample(frac=1, random_state=73)
    source.index = np.arange(len(source)) // 2
    config = FilterConfig(method=method, time_basis=time_basis)
    actual = filter_trades(source, config, backend="python")
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_bulk_groups", _legacy_groups)
        expected = filter_trades(source, config, backend="python")
    _assert_audit_parity(actual, expected)


@pytest.mark.parametrize("method", ("hampel", "local_linear", "jump_reversion", "local_piecewise", "consensus"))
@pytest.mark.parametrize("time_basis", ("wall", "trading"))
def test_optional_compiled_backend_matches_reference_demo(method, time_basis):
    # TEST LOGIC: import skipping keeps the ordinary installation independent of the optional compiler.
    pytest.importorskip("numba")
    source = make_demo(seed=712).sample(frac=1, random_state=12)
    source.index = np.arange(len(source)) // 3
    config = FilterConfig(method=method, time_basis=time_basis)
    actual = filter_trades(source, config, backend="numba")
    expected = filter_trades(source, config, backend="python")
    _assert_audit_parity(actual, expected)
    assert actual.attrs["jump_filter"]["backend"] == "numba"


@pytest.mark.parametrize("unit", ("s", "ms", "us", "ns"))
@pytest.mark.parametrize("zone", ("UTC", "America/New_York"))
def test_bulk_timestamp_conversion_matches_scalar_dst_contract(unit, zone):
    # TEST LOGIC: nonexistent and ambiguous local dates remain NaT instead of being guessed.
    values = pd.Series(pd.to_datetime(["2025-03-09 02:30", "2025-11-02 01:30", "2025-03-10 08:00", None])).dt.as_unit(unit)
    expected = pd.Series([engine._parse_time(value, zone) for value in values], dtype="datetime64[ns, UTC]")
    pd.testing.assert_series_equal(engine._parse_times(values, zone), expected)
    aware = values.dt.tz_localize("UTC")
    expected_aware = pd.Series([engine._parse_time(value, zone) for value in aware], dtype="datetime64[ns, UTC]")
    pd.testing.assert_series_equal(engine._parse_times(aware, zone), expected_aware)


def test_explicit_backend_validation_and_unsupported_solver_metadata():
    # TEST LOGIC: invalid backends fail clearly; joint solvers disclose their Python execution path.
    source = make_demo().iloc[:25]
    with pytest.raises(ValueError, match="backend"):
        filter_trades(source, backend="invalid")
    result = filter_trades(source, FilterConfig(method="robust_trend"), backend="numba")
    assert result.attrs["jump_filter"]["backend"] == "python"


def test_small_interactive_review_avoids_optional_import(monkeypatch):
    # TEST LOGIC: selected-bond auto requests avoid importing/JIT-compiling an accelerator entirely.
    import builtins
    original_import = builtins.__import__

    def guard_import(name, *args, **kwargs):
        # TEST LOGIC: compiler imports are forbidden only during this small review.
        if name.startswith("numba") or name.endswith("accelerator"):
            raise AssertionError("small interactive review imported its compiler")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard_import)
    result = filter_trades(make_demo().iloc[:77])
    assert result.attrs["jump_filter"]["backend"] == "python"


def test_optional_compiler_failure_falls_back_but_explicit_numba_raises(monkeypatch):
    # TEST LOGIC: simulated compiler failures must disclose Python fallback without hiding explicit failures.
    pytest.importorskip("numba")
    from jump_filter import accelerator
    source = make_demo().iloc[:77]
    expected = filter_trades(source, backend="python")

    def broken_kernel(*args):
        # TEST LOGIC: fail before any numeric publication, as a compiler typing failure would.
        raise accelerator.COMPILATION_ERROR("simulated optional compiler failure")

    monkeypatch.setattr(accelerator, "local_kernel", broken_kernel)
    monkeypatch.setattr(engine, "_resolve_backend", lambda method, backend, size: accelerator)
    actual = filter_trades(source, backend="auto")
    _assert_audit_parity(actual, expected)
    assert actual.attrs["jump_filter"]["backend"] == "python"
    with pytest.raises(accelerator.COMPILATION_ERROR):
        filter_trades(source, backend="numba")


@pytest.mark.parametrize("method", ("hampel", "local_linear", "jump_reversion", "local_piecewise", "consensus"))
def test_compiled_custom_calendar_and_duplicate_cohort_parity(method):
    # TEST LOGIC: authoritative split sessions, closed endpoints and invalid rows keep all original audit states.
    pytest.importorskip("numba")
    times = pd.date_range("2025-01-03T13:00Z", periods=31, freq="min")
    source = pd.DataFrame(dict(CUSIP="A", time=times, spread=100 + np.sin(np.arange(31))))
    duplicate = source.iloc[[7]].copy()
    duplicate["spread"] = 135
    source = pd.concat([source, duplicate], ignore_index=True).sample(frac=1, random_state=8)
    schedule = pd.DataFrame(dict(open=[times[0], times[20]], close=[times[11], times[30]]))
    config = FilterConfig(method=method, time_basis="trading", min_neighbors=4, window=12, max_gap="10min")
    actual = filter_trades(source, config, session_schedule=schedule, backend="numba")
    expected = filter_trades(source, config, session_schedule=schedule, backend="python")
    _assert_audit_parity(actual, expected)


@pytest.mark.parametrize("method", ("local_linear", "local_piecewise", "consensus"))
def test_compiled_irregular_sparse_cluster_and_cutoff_decisions(method):
    # TEST LOGIC: extreme irregular timing tests extrapolation conditioning and strict cutoff boundary behavior.
    pytest.importorskip("numba")
    offsets = np.r_[np.arange(12), 10**12 + np.arange(13)]
    source = pd.DataFrame(dict(CUSIP="A", time=pd.to_datetime(offsets, unit="ns", utc=True), spread=100.0))
    source.loc[[5, 18], "spread"] = [101.0, np.nextafter(101.0, np.inf)]
    config = FilterConfig(method=method, horizon="1D", max_gap="1D", abs_floor=1, threshold=4.5)
    actual = filter_trades(source, config, backend="numba")
    expected = filter_trades(source, config, backend="python")
    _assert_audit_parity(actual, expected)
