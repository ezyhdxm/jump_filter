"""Review-population and bounded-cache contracts for large dashboard sources."""
# TEST LOGIC: Small deterministic inputs exercise population boundaries and exact source identity.
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from jump_filter import FilterConfig
from jump_filter.dashboard import ReviewWorkspace


def _frame():
    # TEST LOGIC: Interleaved instruments and repeated index labels expose accidental index-label joins.
    return pd.DataFrame({"CUSIP": ["001", "002", "003"] * 8,
                         "time": pd.date_range("2026-10-01T14:00:00Z", periods=24, freq="min"),
                         "spread": [100., 200., 300.] * 8}, index=[7, 7, 2] * 8)


def _workspace(frame, **kwargs):
    # TEST LOGIC: Mapping is explicit and remains fixed for a workspace's immutable source snapshot.
    return ReviewWorkspace(frame, dict(cusip_col="CUSIP", time_col="time", spread_col="spread", timezone="UTC"), **kwargs)


def test_selected_review_cache_preserves_global_positions_and_settings_calendar_keys(monkeypatch):
    # TEST LOGIC: Observe exactly which populations reach the engine; acceleration itself is covered by engine tests.
    import jump_filter
    actual, calls = jump_filter.filter_trades, []
    def counted(frame, config, **kwargs):
        # TEST LOGIC: Force the reference engine so this test remains independent of optional compiler availability.
        calls.append((len(frame), config.method))
        return actual(frame, config, backend="python", **kwargs)
    monkeypatch.setattr(jump_filter, "filter_trades", counted)
    source = _frame()
    workspace = _workspace(source)
    config = FilterConfig(window=6, min_neighbors=2)
    first = workspace.review(config, bond="001")
    assert calls == [(8, "consensus")]
    assert first["result"]["jf_row_id"].tolist() == list(range(0, 24, 3))
    assert first["result"].index.equals(source.iloc[::3].index)
    assert first["summary"]["CUSIP"].tolist() == ["001"]
    assert first["fitting"]["supplied"].tolist() == [8]
    assert workspace.review(config, bond="001") is first
    workspace.review(config, bond="002")
    workspace.review(replace(config, threshold=5), bond="001")
    assert len(calls) == 3
    schedule = pd.DataFrame({"open": ["2026-10-01T08:00:00-04:00"], "close": ["2026-10-01T18:30:00-04:00"]})
    scheduled = workspace.review(config, bond="001", session_schedule=schedule)
    assert workspace.review(config, bond="001", session_schedule=schedule.copy()) is scheduled
    assert len(calls) == 4
    batch = workspace.review(config, scope="all", bond="001")
    assert len(calls) == 5 and calls[-1][0] == 24
    assert batch["result"]["jf_row_id"].tolist() == list(range(24))
    assert batch["result"].index.equals(source.index)
    assert batch["summary"]["CUSIP"].tolist() == ["001", "002", "003"]


def test_review_cache_evicts_by_count_and_skips_oversized_frames():
    # TEST LOGIC: Revisiting the most recent entries must not let a dashboard retain unbounded annotated portfolios.
    workspace = _workspace(_frame(), max_entries=2)
    config = FilterConfig(method="rolling_iqr", window=6, min_neighbors=2)
    first = workspace.review(config, bond="001")
    workspace.review(config, bond="002")
    assert workspace.review(config, bond="001") is first
    workspace.review(config, bond="003")
    assert len(workspace.cache) == 2
    assert first["key"] in workspace.cache
    assert not any(key[1] == "002" for key in workspace.cache)
    tiny = _workspace(_frame(), max_bytes=1)
    assert len(tiny.review(config, scope="all")["result"]) == 24
    assert not tiny.cache and tiny.cache_bytes == 0


def test_large_notebook_cusip_search_is_bounded_and_does_not_filter_on_search(monkeypatch):
    # TEST LOGIC: A 13,000-instrument selector sends bounded choices while preserving exact CUSIP values.
    pytest.importorskip("ipywidgets")
    from jump_filter.dashboard import FilterDashboard
    monkeypatch.setattr(FilterDashboard, "_render_explanation", lambda panel: None)
    monkeypatch.setattr(FilterDashboard, "_empty_outputs", lambda panel: None)
    monkeypatch.setattr(FilterDashboard, "_render", lambda panel, record: ())
    monkeypatch.setattr(FilterDashboard, "_publish", lambda *args: None)
    source = pd.DataFrame({"CUSIP": [f"B{i:05d}" for i in range(13000)],
                           "time": ["2026-10-01T14:00:00Z"] * 13000, "spread": np.full(13000, 100.)})
    panel = FilterDashboard(source, config=FilterConfig(method="rolling_iqr"))
    assert len(panel.cusip.options) == 100
    panel.run()
    original = panel.result
    assert len(original) == 1 and panel.applied_scope == "selected"
    panel.cusip_search.value = "B12999"
    assert panel.result is original
    assert len(panel.cusip.options) == 2
    assert ("B12999", "B12999") in panel.cusip.options
    panel.method.value = "hampel"
    panel.cusip.value = "B12999"
    assert panel.applied_config.method == "rolling_iqr"
    assert panel.result["CUSIP"].tolist() == ["B12999"]
    assert panel.result["jf_row_id"].tolist() == [12999]
    assert "Pending settings" in panel.pending.value
    panel.cusip.value = "B00000"
    assert panel.result is original
