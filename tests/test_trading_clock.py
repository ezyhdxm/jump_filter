"""Calendar boundaries, elapsed trading time, and authoritative schedule checks."""

# SETUP LOGIC: public immutable controls and deterministic timestamp fixtures.
import numpy as np
import pandas as pd
import pytest
from jump_filter import FilterConfig
from jump_filter.clock import INACTIVE_CLOCK, trading_clock


# TEST LOGIC: parse explicit offsets without interpreting naive local clock strings.
def stamps(values):
    return pd.to_datetime(pd.Series(values, dtype=object), utc=True, format="mixed")


# TEST LOGIC: wall mode preserves exact UTC nanoseconds and marks only invalid times inactive.
def test_wall_clock_is_legacy_elapsed_time():
    parsed = stamps(["2025-01-03T18:29-05:00", "2025-01-06T08:00-05:00", None])
    clock, active, metadata = trading_clock(parsed, FilterConfig(time_basis="wall"))
    np.testing.assert_array_equal(clock, pd.DatetimeIndex(parsed).as_unit("ns").asi8)
    np.testing.assert_array_equal(active, [True, True, False])
    assert clock[2] == INACTIVE_CLOCK
    assert clock[1] - clock[0] == pd.Timedelta("2D13h31min").value
    assert metadata["calendar_kind"] == "wall"
    assert metadata["session_count"] == 0


# TEST LOGIC: scheduled overnight and weekend closures consume no trading time.
def test_overnight_and_weekend_are_compressed():
    parsed = stamps(["2025-01-02T18:29-05:00", "2025-01-03T08:00-05:00",
                     "2025-01-03T18:29-05:00", "2025-01-06T08:00-05:00"])
    clock, active, metadata = trading_clock(parsed, FilterConfig(time_basis="trading"))
    assert active.all()
    assert clock[1] - clock[0] == pd.Timedelta("1min").value
    assert clock[3] - clock[2] == pd.Timedelta("1min").value
    assert metadata["session_count"] == 3
    assert metadata["calendar_kind"] == "configured_weekdays"


# TEST LOGIC: a no-trade interval within an open session remains a real clock gap.
def test_within_session_liquidity_gaps_remain_elapsed():
    parsed = stamps(["2025-01-06T09:00-05:00", "2025-01-06T16:00-05:00"])
    clock, active, _ = trading_clock(parsed, FilterConfig(time_basis="trading"))
    assert active.all()
    assert clock[1] - clock[0] == pd.Timedelta("7h").value


# TEST LOGIC: exact close endpoints are inactive and cannot collide with the following opening cohort.
def test_open_inclusive_close_exclusive_and_closed_dates():
    parsed = stamps(["2025-01-03T07:59:59-05:00", "2025-01-03T08:00-05:00",
                     "2025-01-03T18:29:59-05:00", "2025-01-03T18:30-05:00",
                     "2025-01-04T12:00-05:00", "2025-01-06T08:00-05:00", None])
    clock, active, _ = trading_clock(parsed, FilterConfig(time_basis="trading"))
    np.testing.assert_array_equal(active, [False, True, True, False, False, True, False])
    assert clock[1] == 0
    assert clock[5] - clock[2] == pd.Timedelta("1s").value
    assert (clock[~active] == INACTIVE_CLOCK).all()


# TEST LOGIC: explicit holiday exclusions are compressed and records on the holiday stay audit-visible.
def test_explicit_holiday_is_compressed():
    parsed = stamps(["2025-01-17T18:29-05:00", "2025-01-20T12:00-05:00", "2025-01-21T08:00-05:00"])
    config = FilterConfig(time_basis="trading", holidays=("2025-01-20",))
    clock, active, metadata = trading_clock(parsed, config)
    np.testing.assert_array_equal(active, [True, False, True])
    assert clock[2] - clock[0] == pd.Timedelta("1min").value
    assert metadata["session_count"] == 2
    assert metadata["holidays"] == ["2025-01-20"]


# TEST LOGIC: spring and fall DST weekends preserve local opening hours and compression.
@pytest.mark.parametrize("friday,monday", [
    ("2025-03-07T18:29-05:00", "2025-03-10T08:00-04:00"),
    ("2025-10-31T18:29-04:00", "2025-11-03T08:00-05:00"),
])
def test_dst_changes_utc_boundaries_without_changing_trading_duration(friday, monday):
    clock, active, metadata = trading_clock(stamps([friday, monday]), FilterConfig(time_basis="trading"))
    assert active.all()
    assert clock[1] - clock[0] == pd.Timedelta("1min").value
    assert metadata["timezone"] == "America/New_York"


# TEST LOGIC: a custom early-close schedule overrides weekday hours, holidays, and generic timezone.
def test_custom_early_close_and_weekend_override_regular_calendar():
    schedule = pd.DataFrame({
        "open": ["2025-01-04T09:00-05:00", "2025-01-03T08:00-05:00"],
        "close": ["2025-01-04T11:00-05:00", "2025-01-03T13:00-05:00"],
    })
    snapshot = schedule.copy(deep=True)
    parsed = stamps(["2025-01-03T12:59-05:00", "2025-01-03T13:00-05:00",
                     "2025-01-03T14:00-05:00", "2025-01-04T09:00-05:00"])
    config = FilterConfig(time_basis="trading", holidays=("2025-01-03",), session_timezone="Asia/Tokyo")
    clock, active, metadata = trading_clock(parsed, config, schedule)
    np.testing.assert_array_equal(active, [True, False, False, True])
    assert clock[3] - clock[0] == pd.Timedelta("1min").value
    assert metadata["calendar_kind"] == "custom_schedule"
    assert metadata["timezone"] == "UTC"
    pd.testing.assert_frame_equal(schedule, snapshot)


# TEST LOGIC: touching custom sessions admit the next open once, with increasing clock coordinates.
def test_custom_prefixed_schema_and_touching_sessions():
    schedule = pd.DataFrame({"session_open": ["2025-01-06T08:00Z", "2025-01-06T10:00Z"],
                             "session_close": ["2025-01-06T10:00Z", "2025-01-06T12:00Z"]})
    clock, active, _ = trading_clock(stamps(["2025-01-06T09:59:59Z", "2025-01-06T10:00Z"]),
                                       FilterConfig(time_basis="trading"), schedule)
    assert active.all()
    assert clock[1] - clock[0] == pd.Timedelta("1s").value


# TEST LOGIC: authoritative intervals can express overnight sessions that regular same-day hours cannot.
def test_custom_overnight_session():
    schedule = pd.DataFrame({"open": ["2025-01-03T22:00Z"], "close": ["2025-01-04T02:00Z"]})
    clock, active, _ = trading_clock(stamps(["2025-01-03T23:00Z", "2025-01-04T01:00Z"]),
                                       FilterConfig(time_basis="trading"), schedule)
    assert active.all()
    assert clock[1] - clock[0] == pd.Timedelta("2h").value


# TEST LOGIC: equivalent custom row orders have identical fingerprints; a boundary change does not.
def test_schedule_fingerprint_is_deterministic_and_sensitive():
    schedule = pd.DataFrame({"open": ["2025-01-03T13:00Z", "2025-01-06T13:00Z"],
                             "close": ["2025-01-03T23:30Z", "2025-01-06T23:30Z"]})
    parsed = stamps(["2025-01-03T15:00Z"])
    config = FilterConfig(time_basis="trading")
    first = trading_clock(parsed, config, schedule)[2]
    reordered = trading_clock(parsed, config, schedule.iloc[::-1])[2]
    assert first["schedule_fingerprint"] == reordered["schedule_fingerprint"]
    schedule.loc[0, "close"] = "2025-01-03T20:00Z"
    altered = trading_clock(parsed, config, schedule)[2]
    assert first["schedule_fingerprint"] != altered["schedule_fingerprint"]
    assert len(first["schedule_fingerprint"]) == 64


# TEST LOGIC: invalid custom intervals fail rather than silently changing the declared calendar.
@pytest.mark.parametrize("schedule", [
    pd.DataFrame({"open": ["2025-01-06T08:00"], "close": ["2025-01-06T10:00Z"]}),
    pd.DataFrame({"open": ["2025-01-06T10:00Z"], "close": ["2025-01-06T08:00Z"]}),
    pd.DataFrame({"open": ["2025-01-06T08:00Z"], "close": ["2025-01-06T08:00Z"]}),
    pd.DataFrame({"open": ["2025-01-06T08:00Z", "2025-01-06T09:00Z"],
                  "close": ["2025-01-06T10:00Z", "2025-01-06T11:00Z"]}),
    pd.DataFrame({"open": [None], "close": ["2025-01-06T10:00Z"]}),
    pd.DataFrame({"open": [1700000000], "close": [1700000001]}),
    pd.DataFrame({"other": []}),
])
def test_invalid_custom_schedules_raise(schedule):
    with pytest.raises((ValueError, TypeError)):
        trading_clock(stamps(["2025-01-06T09:00Z"]), FilterConfig(time_basis="trading"), schedule)


# TEST LOGIC: all-invalid inputs and explicitly empty schedules do not fabricate activity.
@pytest.mark.parametrize("values", [[None, None], []])
def test_all_nat_and_empty_inputs(values):
    clock, active, metadata = trading_clock(stamps(values), FilterConfig(time_basis="trading"))
    assert len(clock) == len(values)
    assert not active.any()
    assert (clock == INACTIVE_CLOCK).all()
    assert metadata["session_count"] == 0
    assert metadata["observed_start"] is None


# TEST LOGIC: an empty authoritative table suppresses the generic weekday fallback.
def test_empty_custom_schedule_is_authoritative():
    schedule = pd.DataFrame({"open": [], "close": []})
    clock, active, metadata = trading_clock(stamps(["2025-01-06T09:00-05:00"]),
                                               FilterConfig(time_basis="trading"), schedule)
    assert not active.any()
    assert clock[0] == INACTIVE_CLOCK
    assert metadata["calendar_kind"] == "custom_schedule"
