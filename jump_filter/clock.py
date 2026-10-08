"""Explicit wall or active-session clocks; no inferred bond holiday calendar.

The trading clock is integral(1_session_open dt), in integer nanoseconds.
Regular sessions are configured weekdays and explicit exclusions. A supplied
aware interval table is authoritative and can express holidays, early closes,
split sessions, and overnight sessions without calendar assumptions.
"""

# SETUP LOGIC: standard-library audit hashing and vectorized time handling.
from datetime import date, time
from hashlib import sha256
import json
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

# CONFIGURATION LOGIC: one explicit sentinel distinguishes inactive records from clock zero.
INACTIVE_CLOCK = np.iinfo(np.int64).min


def _session_offset(value):
    # INPUT VALIDATION LOGIC: regular hours must be local clock times without an offset.
    parsed = time.fromisoformat(value)
    if parsed.tzinfo is not None:
        raise ValueError("regular session hours must not include a timezone")
    # Input: value='08:30:00.000001' -> Output: Timedelta(30600000001000ns).
    # Trick: preserve microseconds without interpreting local hours as elapsed UTC times.
    # CORE LOGIC: STEP 1
    return pd.Timedelta(hours=parsed.hour, minutes=parsed.minute,
                        seconds=parsed.second, microseconds=parsed.microsecond)


def _regular_sessions(parsed, config):
    """Generate configured local weekday sessions for the observed date range."""
    # INPUT VALIDATION LOGIC: dates and timezone must be explicit, without guessed holidays.
    ZoneInfo(config.session_timezone)
    opening, closing = _session_offset(config.session_open), _session_offset(config.session_close)
    if opening >= closing:
        raise ValueError("regular session open must be before close within one local day")
    holidays = tuple(date.fromisoformat(item).isoformat() for item in config.holidays)
    if not parsed.notna().any():
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Input: valid times=['2025-01-03T15:00Z','2025-01-06T15:00Z'],timezone='America/New_York',
    # holidays=() -> Output: session dates=['2025-01-03','2025-01-06'] (Friday and Monday).
    # Trick: local dates determine weekdays; UTC dates can differ from their trading dates.
    # CORE LOGIC: STEP 1
    local = parsed[parsed.notna()].tz_convert(config.session_timezone)
    dates = pd.date_range(local.min().date(), local.max().date(), freq="D")
    admitted = (dates.weekday < 5) & ~dates.strftime("%Y-%m-%d").isin(holidays)
    dates = dates[admitted]

    # Input: dates=['2025-03-07','2025-03-10'],hours08:00–18:30,timezone='America/New_York' ->
    # Output: opens=['2025-03-07T13:00Z','2025-03-10T12:00Z'],
    # closes=['2025-03-07T23:30Z','2025-03-10T22:30Z'].
    # Trick: localize local hours separately before UTC conversion; DST shifts UTC session boundaries.
    # CORE LOGIC: STEP 2
    opens = (dates + opening).tz_localize(config.session_timezone, ambiguous="raise", nonexistent="raise")
    closes = (dates + closing).tz_localize(config.session_timezone, ambiguous="raise", nonexistent="raise")
    return opens.tz_convert("UTC").as_unit("ns").asi8, closes.tz_convert("UTC").as_unit("ns").asi8


def _aware_session_value(value):
    # INPUT VALIDATION LOGIC: reject ambiguous numeric epochs and all naive or missing boundaries.
    if isinstance(value, (int, float, np.number)) or pd.isna(value):
        raise ValueError("custom session boundaries require explicit aware timestamps")
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("custom session boundaries must include a timezone")
    return stamp.tz_convert("UTC").value


def _custom_sessions(schedule):
    """Normalize authoritative intervals without changing the source frame."""
    # INPUT VALIDATION LOGIC: support one documented schema and reject ambiguous competing pairs.
    if not isinstance(schedule, pd.DataFrame):
        raise TypeError("session_schedule must be a pandas DataFrame")
    plain = {"open", "close"} <= set(schedule.columns)
    prefixed = {"session_open", "session_close"} <= set(schedule.columns)
    if plain == prefixed:
        raise ValueError("provide exactly one open/close or session_open/session_close column pair")
    opening, closing = ("open", "close") if plain else ("session_open", "session_close")

    # Input: schedule open=['2025-01-06T13:00Z','2025-01-03T13:00Z'],
    # close=['2025-01-06T23:30Z','2025-01-03T23:30Z'] ->
    # Output: UTC intervals sorted chronologically Friday then Monday, with source order unchanged.
    # Trick: parsing loops over schedule intervals only; trade timestamp scoring stays vectorized.
    # CORE LOGIC: STEP 1
    opens = np.array([_aware_session_value(value) for value in schedule[opening]], dtype=np.int64)
    closes = np.array([_aware_session_value(value) for value in schedule[closing]], dtype=np.int64)
    order = np.argsort(opens, kind="stable")
    opens, closes = opens[order], closes[order]

    # INPUT VALIDATION LOGIC: adjacent sessions may touch; they must have positive duration and never overlap.
    if np.any(closes <= opens):
        raise ValueError("every custom session close must be strictly after its open")
    if len(opens) > 1 and np.any(opens[1:] < closes[:-1]):
        raise ValueError("custom sessions must not overlap")
    return opens, closes


def _schedule_prefix(opens, closes):
    # Input: opens=[0,100],closes=[10,120] (ns) -> Output: prefix=[0,10,30].
    # Trick: object arithmetic checks the total before integer casting, preventing nanosecond overflow.
    # CORE LOGIC: STEP 1
    durations = closes.astype(object) - opens.astype(object)
    total = int(np.sum(durations, dtype=object))
    if total > np.iinfo(np.int64).max:
        raise ValueError("total scheduled duration exceeds the int64 nanosecond clock range")
    return np.concatenate(([0], np.cumsum(durations.astype(np.int64), dtype=np.int64)))


def _map_session_clock(stamps, valid, opens, closes):
    """Locate interval memberships and integrate the elapsed active durations."""
    # Input: stamps=[0,5,10,50,100,119,120],sessions=[0,10),[100,120),valid=allTrue ->
    # Output: positions=[0,0,0,0,1,1,1],active=[True,True,False,False,True,True,False].
    # Trick: open is inclusive and close exclusive; closed endpoints cannot merge with a later open.
    # CORE LOGIC: STEP 1
    positions = np.searchsorted(opens, stamps, side="right") - 1
    safe = np.maximum(positions, 0)
    active = valid & (positions >= 0) & (stamps >= opens[safe]) & (stamps < closes[safe])

    # Input: same stamps and sessions as step1,prefix=[0,10,30] ->
    # Output: clock=[0,5,INT64_MIN,INT64_MIN,10,29,INT64_MIN].
    # Trick: compute subtraction only for active records; invalid sentinel arithmetic could overflow.
    # CORE LOGIC: STEP 2
    prefix = _schedule_prefix(opens, closes)
    clock = np.full(len(stamps), INACTIVE_CLOCK, dtype=np.int64)
    selected = positions[active]
    clock[active] = prefix[selected] + (stamps[active] - opens[selected])
    return clock, active


def _clock_metadata(kind, timezone, parsed, opens, closes, config):
    # AUDIT LOGIC: canonical intervals and explicit regular settings make fingerprints reproducible.
    descriptor = dict(calendar_kind=kind, timezone=timezone)
    if kind == "configured_weekdays":
        descriptor.update(session_open=config.session_open, session_close=config.session_close,
                          holidays=sorted(set(config.holidays)))
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = sha256(encoded + opens.astype("<i8").tobytes() + closes.astype("<i8").tobytes()).hexdigest()
    admitted = parsed[parsed.notna()]
    metadata = dict(descriptor, session_count=len(opens), schedule_fingerprint=digest,
                    range_start=pd.Timestamp(int(opens[0]), tz="UTC").isoformat() if len(opens) else None,
                    range_end=pd.Timestamp(int(closes[-1]), tz="UTC").isoformat() if len(closes) else None,
                    observed_start=admitted.min().isoformat() if len(admitted) else None,
                    observed_end=admitted.max().isoformat() if len(admitted) else None,
                    interval_convention="[open,close)", clock_unit="ns")
    return metadata


def trading_clock(parsed_UTC_series, config, session_schedule=None):
    """Return clock int64 array, active bool array, and an audit metadata dict.

    Wall mode preserves UTC nanoseconds. Trading mode compresses only declared
    closures. Within-session inactivity still consumes clock time. An
    authoritative schedule overrides all generic weekdays, hours, and holidays.
    Neither mode changes the original UTC timestamps used in displays.
    """
    # SETUP LOGIC: normalize the already parsed UTC timestamps and allocate explicit inactivity.
    parsed = pd.DatetimeIndex(pd.to_datetime(parsed_UTC_series, utc=True, errors="coerce")).as_unit("ns")
    stamps = parsed.asi8.copy()
    valid = parsed.notna()
    if config.time_basis not in ("wall", "trading"):
        raise ValueError("time_basis must be 'wall' or 'trading'")
    if config.time_basis == "wall":
        empty = np.array([], dtype=np.int64)
        return stamps, valid, _clock_metadata("wall", "UTC", parsed, empty, empty, config)

    # Input: dates Fri2025-01-03 and Mon2025-01-06,default local hours08:00–18:30,
    # no custom table -> Output: two configured weekday intervals each lasting10.5h.
    # Trick: a supplied table is authoritative even when empty; its own intervals alone define activity.
    # CORE LOGIC: STEP 1
    custom = session_schedule is not None
    opens, closes = _custom_sessions(session_schedule) if custom else _regular_sessions(parsed, config)
    kind = "custom_schedule" if custom else "configured_weekdays"
    timezone = "UTC" if custom else config.session_timezone

    # Input: times=['2025-01-03T18:29-05:00','2025-01-06T08:00-05:00'],regular two sessions ->
    # Output: active=[True,True],clock=[37740000000000,37800000000000],difference60s.
    # Trick: no sessions or all-NaT inputs yield inactive sentinels, not a fabricated empty-session match.
    # CORE LOGIC: STEP 2
    if len(opens):
        clock, active = _map_session_clock(stamps, valid, opens, closes)
    else:
        clock = np.full(len(stamps), INACTIVE_CLOCK, dtype=np.int64)
        active = np.zeros(len(stamps), dtype=bool)
    metadata = _clock_metadata(kind, timezone, parsed, opens, closes, config)
    return clock, active, metadata
