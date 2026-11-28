"""Run variants across all primary sessions and print a comparison."""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, time as clock, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from numba import njit


EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

BASE = 80_000
SIZE = 80_000
MAX_TRADES = 5_000
MAX_LOTS = 4

TICK_CENTS = 50
COMMISSION_CENTS = 50
STOP_SLIP_TICKS = 1
MIN_RUNG_GAP_TICKS = 20

RTH_START = clock(9, 30)
RTH_END = clock(16, 0)
ENTRY_START = clock(10, 15)
ENTRY_END = clock(15, 0)

COOLDOWN_100NS = 15 * 60 * 10_000_000
MAX_DAILY_TRADES = 5
MIN_POC_DISTANCE_TICKS = 120

RUNG_TP1, RUNG_POC, RUNG_OPP, RUNG_FIXED = 0, 1, 2, 3

REASONS = {
    0: "tp1", 1: "poc_rung", 2: "opp_rung",
    3: "fixed_runner", 4: "session_end", 5: "session_close",
}

# Prop firm thresholds
PROP_DD_LIMIT_CENTS = 200_000   # $2,000 EOD trailing drawdown limit
PROP_TARGET_CENTS = 300_000     # $3,000 profit target

# Trade-row column layout, shared by simulate() and the reporting code.
LEG_T0 = 3
LEG_P0 = LEG_T0 + MAX_LOTS
LEG_R0 = LEG_P0 + MAX_LOTS
STAT0 = LEG_R0 + MAX_LOTS
NET_COL = STAT0 + 5


@dataclass(frozen=True)
class Variant:
    name: str
    stop_ticks: int
    lots: int
    tp1_ticks: int
    use_poc_rung: bool
    use_opp_rung: bool
    runner_ticks: int
    min_poc_dist: int = MIN_POC_DISTANCE_TICKS
    move_be: bool = False
    trail_ticks: int = 0        # 0 = off; activates after TP1 fills
    entry_hour: int = 10
    entry_minute: int = 15
    atr_mult: float = 0.0       # >0 → TP1 = clamp(ATR15 * mult) in ticks
    tp2_ticks: int = 0          # offset beyond TP1 for lot 2 (3-lot family)
    tp2_at_poc: bool = False    # lot 2 targets developing POC instead
    tp1_lots: int = 1           # how many lots share the TP1 price (rest → runner)
    # Block new entries when price is outside session VWAP ± sigma*stdev
    # (trend/impulse filter). persist_secs=0 → check at signal only;
    # >0 → must have been outside continuously for that long.
    vwap_sigma: float = 0.0
    vwap_persist_secs: int = 0
    family: str = "legacy"
    note: str = ""


def _k60(**kw):
    """K_two_lot_60 base: 2 lots, opp-VA runner, 60pt stop."""
    base = dict(
        stop_ticks=240, lots=2, tp1_ticks=80,
        use_poc_rung=False, use_opp_rung=True, runner_ticks=0,
    )
    base.update(kw)
    return Variant(**base)


LEGACY_VARIANTS = [
    Variant("A_control", stop_ticks=120, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=320,
            family="legacy",
            note="baseline control, no breakeven, 10:15 entry"),
    Variant("O_be_30", stop_ticks=120, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=320,
            move_be=True, family="legacy", note="A + breakeven after TP1"),
    Variant("P_late_30", stop_ticks=120, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=320,
            entry_hour=11, entry_minute=0, family="legacy",
            note="A + 11:00 entry floor"),
    Variant("Q_be_late_30", stop_ticks=120, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=320,
            move_be=True, entry_hour=11, entry_minute=0, family="legacy",
            note="A + breakeven + 11:00 floor"),
    _k60(name="K_two_lot_60", family="legacy",
         note="2 lots: TP1 20pt banked, 1 lot to opp VA edge, 60pt stop"),
    _k60(name="R_be_60", move_be=True, family="legacy",
         note="K + breakeven after TP1"),
    _k60(name="S_be_late_60", move_be=True, entry_hour=11, entry_minute=0,
         family="legacy", note="K + breakeven + 11:00 floor"),
    _k60(name="T_trail20_60", trail_ticks=80, family="legacy",
         note="K + 20pt trail after TP1 (locks BE at TP1)"),
    _k60(name="U_trail30_60", trail_ticks=120, family="legacy",
         note="K + 30pt trail after TP1 (tightens to -10pt at TP1)"),
    _k60(name="V_be_trail30_60", move_be=True, trail_ticks=120, family="legacy",
         note="K + BE at TP1, then 30pt trail once > +30pt"),
    _k60(name="W_be_trail40_60", move_be=True, trail_ticks=160, family="legacy",
         note="K + BE at TP1, then 40pt trail once > +40pt"),
]

# Static TP1 plateau on K base (points → ticks = pts * 4)
STATIC_TP1_VARIANTS = [
    _k60(name="K_two_lot_60", family="static_tp1",
         note="control: TP1 20pt"),
] + [
    _k60(name=f"TP1_{pts}pt", tp1_ticks=pts * 4, family="static_tp1",
         note=f"K base, static TP1 {pts}pt")
    for pts in (15, 20, 25, 30, 35, 40, 45, 50)
]

# ATR(14) on 15-minute RTH bars → TP1 = clamp(k * ATR, 15..50 pt)
ATR_TP1_VARIANTS = [
    _k60(name="K_two_lot_60", family="atr_tp1",
         note="control: fixed TP1 20pt"),
] + [
    _k60(name=f"ATR15_x{str(k).replace('.', 'p')}", atr_mult=float(k),
         family="atr_tp1",
         note=f"TP1 = clamp({k}×ATR(14,15m), 15–50pt); skip entry until ATR warm")
    for k in (0.5, 0.75, 1.0, 1.25, 1.5)
]

# 3 MNQ: TP1=20, TP2 = offset past TP1 or POC, TP3 = opp VA
THREE_LOT_VARIANTS = [
    _k60(name="K_two_lot_60", family="three_lot",
         note="2-lot control for comparison"),
] + [
    _k60(name=f"L3_tp2_{pts}pt", lots=3, tp1_ticks=80, tp2_ticks=pts * 4,
         family="three_lot",
         note=f"3 lots: TP1 20pt, TP2 +{pts}pt past TP1, TP3 opp VA")
    for pts in (10, 15, 20, 25, 30)
] + [
    _k60(name="L3_tp2_poc", lots=3, tp1_ticks=80, tp2_at_poc=True,
         family="three_lot",
         note="3 lots: TP1 20pt, TP2 at POC, TP3 opp VA"),
]

# Scale-out: N lots at TP1, 1 runner to opp VA
SCALE_OUT_VARIANTS = [
    _k60(name="TP1_40pt", tp1_ticks=160, family="scale_out",
         note="2-lot control: 1×40pt + opp VA"),
    _k60(name="L3_2x40_opp", lots=3, tp1_ticks=160, tp1_lots=2,
         family="scale_out",
         note="3 lots: 2 off at 40pt, 1 to opp VA"),
    _k60(name="L3_2x20_opp", lots=3, tp1_ticks=80, tp1_lots=2,
         family="scale_out",
         note="3 lots: 2 off at 20pt, 1 to opp VA"),
    _k60(name="L3_2x45_opp", lots=3, tp1_ticks=180, tp1_lots=2,
         family="scale_out",
         note="3 lots: 2 off at 45pt, 1 to opp VA"),
]

# VWAP ± 2σ impulse filter on the 40pt plateau candidate
VWAP_FILTER_VARIANTS = [
    _k60(name="TP1_40pt", tp1_ticks=160, family="vwap_filter",
         note="control: TP1 40pt, no VWAP filter"),
    _k60(name="TP1_40_vwap2", tp1_ticks=160, vwap_sigma=2.0,
         family="vwap_filter",
         note="TP1 40pt; block entry if price outside VWAP±2σ at signal"),
    _k60(name="TP1_40_vwap2_5m", tp1_ticks=160, vwap_sigma=2.0,
         vwap_persist_secs=300, family="vwap_filter",
         note="TP1 40pt; block while outside VWAP±2σ for ≥5 minutes"),
    _k60(name="TP1_40_vwap2_15m", tp1_ticks=160, vwap_sigma=2.0,
         vwap_persist_secs=900, family="vwap_filter",
         note="TP1 40pt; block while outside VWAP±2σ for ≥15 minutes"),
]

VARIANTS = LEGACY_VARIANTS
TP1_MIN_TICKS = 60    # 15 points
TP1_MAX_TICKS = 200   # 50 points
ATR_LEN = 14
ATR_BAR_MINUTES = 15

FILL_MODELS = ((0, "touch"), (1, "trade_through"))


# ---------------------------------------------------------------- utilities
def timestamp_100ns(instant):
    e = instant.astimezone(UTC) - EPOCH
    return (e.days * 86_400 * 10_000_000
            + e.seconds * 10_000_000
            + e.microseconds * 10)


def from_100ns(value):
    s, f = divmod(int(value), 10_000_000)
    return datetime.fromtimestamp(s, UTC).replace(microsecond=f // 10)


def price_text(ticks):
    return f"{ticks / 4:.2f}"


def eastern_day_id(stamp):
    return from_100ns(stamp).astimezone(EASTERN).date().toordinal()


@njit(cache=True)
def _vwap_entry_ok_core(prices, sizes, stamps, rth_mask, sigma, persist_ns):
    """Numba core for session VWAP ± sigma filter. rth_mask: 1 during RTH."""
    n = len(prices)
    ok = np.ones(n, np.int8)
    sum_v = 0.0
    sum_pv = 0.0
    sum_p2v = 0.0
    outside_since = np.int64(-1)
    for i in range(n):
        if rth_mask[i] == 0:
            ok[i] = 1
            outside_since = np.int64(-1)
            continue
        px = float(prices[i])
        vol = float(sizes[i]) if sizes[i] > 0 else 1.0
        sum_v += vol
        sum_pv += px * vol
        sum_p2v += px * px * vol
        if sum_v <= 0.0:
            ok[i] = 1
            continue
        vwap = sum_pv / sum_v
        var = sum_p2v / sum_v - vwap * vwap
        if var < 0.0:
            var = 0.0
        std = var ** 0.5
        if std <= 0.0:
            ok[i] = 1
            outside_since = np.int64(-1)
            continue
        outside = abs(px - vwap) > (sigma * std)
        if not outside:
            outside_since = np.int64(-1)
            ok[i] = 1
            continue
        stamp = stamps[i]
        if outside_since < 0:
            outside_since = stamp
        fired = 1 if persist_ns <= 0 else (
            1 if (stamp - outside_since) >= persist_ns else 0
        )
        ok[i] = 0 if fired == 1 else 1
    return ok


def session_vwap_entry_ok(prices, sizes, stamps, sigma=2.0, persist_secs=0):
    """Allow-entry mask from developing RTH session VWAP ± sigma*stdev.

    Returns int8 array: 1 = may enter, 0 = block (price treated as impulsive /
    trending relative to session VWAP). Variance is volume-weighted.
    persist_secs > 0 requires a continuous outside-band streak at least that long
    (using tick timestamps); 0 means check the signal tick only.
    """
    n = len(prices)
    if n == 0 or sigma <= 0:
        return np.ones(n, dtype=np.int8)

    # Build RTH mask once (Eastern wall clock).
    rth_mask = np.zeros(n, dtype=np.int8)
    for i, stamp in enumerate(stamps):
        t = from_100ns(int(stamp)).astimezone(EASTERN).time()
        if RTH_START <= t < RTH_END:
            rth_mask[i] = 1

    persist_ns = int(persist_secs) * 10_000_000
    return _vwap_entry_ok_core(
        np.ascontiguousarray(prices, dtype=np.int32),
        np.ascontiguousarray(sizes, dtype=np.int64),
        np.ascontiguousarray(stamps, dtype=np.int64),
        rth_mask,
        float(sigma),
        np.int64(persist_ns),
    )


def atr_15m_ticks(prices, stamps, length=ATR_LEN, bar_minutes=ATR_BAR_MINUTES):
    """Wilder ATR(length) on completed Eastern bar_minutes bars, forward-filled.

    Returns int32 ticks (quarter-points). 0 until `length` bars have closed.
    Only RTH ticks (09:30–16:00 ET) contribute to bars.
    """
    n = len(prices)
    out = np.zeros(n, dtype=np.int32)
    if n == 0:
        return out

    # bucket_id -> [high, low, close] in ticks; build in time order
    buckets = {}
    order = []
    bar_ns = bar_minutes * 60 * 10_000_000
    for i in range(n):
        et = from_100ns(int(stamps[i])).astimezone(EASTERN)
        t = et.time()
        if t < RTH_START or t >= RTH_END:
            continue
        # Floor to bar_minutes within the Eastern local day.
        minute_of_day = et.hour * 60 + et.minute
        floored = (minute_of_day // bar_minutes) * bar_minutes
        # Unique key: ordinal day + floored minute
        key = et.toordinal() * 10_000 + floored
        px = int(prices[i])
        if key not in buckets:
            buckets[key] = [px, px, px]
            order.append(key)
        else:
            h, l, _ = buckets[key]
            if px > h:
                h = px
            if px < l:
                l = px
            buckets[key] = [h, l, px]

    if not order:
        return out

    # Wilder ATR on completed bars
    atr_by_key = {}
    prev_close = None
    atr = None
    trs = []
    for key in order:
        h, l, c = buckets[key]
        if prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        if len(trs) < length:
            atr_by_key[key] = 0
        elif len(trs) == length:
            atr = sum(trs) / float(length)
            atr_by_key[key] = int(round(atr))
        else:
            atr = (atr * (length - 1) + tr) / float(length)
            atr_by_key[key] = int(round(atr))
        prev_close = c

    # Map each tick to ATR of the last *completed* bar (not the forming one).
    bar_ends = []
    for key in order:
        day_ord, floored = divmod(key, 10_000)
        d = datetime.fromordinal(day_ord).date()
        hh, mm = divmod(floored, 60)
        start = timestamp_100ns(
            datetime.combine(d, clock(hh, mm), tzinfo=EASTERN)
        )
        bar_ends.append((start + bar_ns, atr_by_key[key]))

    last_atr = 0
    j = 0
    for i in range(n):
        stamp = int(stamps[i])
        while j < len(bar_ends) and stamp >= bar_ends[j][0]:
            last_atr = bar_ends[j][1]
            j += 1
        out[i] = last_atr
    return out


# ---------------------------------------------------------- profile building
def rth_profile_for_session(prices, sizes, stamps):
    """Build a developing profile from 09:30-16:00 Eastern ticks."""
    eastern_times = [
        from_100ns(stamp).astimezone(EASTERN).time()
        for stamp in stamps
    ]
    rth_mask = np.fromiter(
        (RTH_START <= current < RTH_END for current in eastern_times),
        dtype=np.bool_,
        count=len(eastern_times),
    )
    rth_indices = np.flatnonzero(rth_mask)

    if len(rth_indices) == 0:
        raise ValueError("No RTH ticks found in session")

    rth_prices = np.ascontiguousarray(prices[rth_indices], dtype=np.int32)
    rth_sizes = np.ascontiguousarray(sizes[rth_indices], dtype=np.int64)

    rth_poc, rth_vah, rth_val = build_profile(rth_prices, rth_sizes)

    full_poc = np.empty(len(prices), dtype=np.int32)
    full_vah = np.empty(len(prices), dtype=np.int32)
    full_val = np.empty(len(prices), dtype=np.int32)

    first_poc = rth_poc[0]
    first_vah = rth_vah[0]
    first_val = rth_val[0]

    full_poc[:] = first_poc
    full_vah[:] = first_vah
    full_val[:] = first_val

    full_poc[rth_indices] = rth_poc
    full_vah[rth_indices] = rth_vah
    full_val[rth_indices] = rth_val

    rth_lookup = np.full(len(prices), -1, dtype=np.int64)
    rth_lookup[rth_indices] = np.arange(len(rth_indices), dtype=np.int64)
    last = -1
    for i in range(len(prices)):
        if rth_lookup[i] >= 0:
            last = rth_lookup[i]
        elif last >= 0:
            full_poc[i] = rth_poc[last]
            full_vah[i] = rth_vah[last]
            full_val[i] = rth_val[last]

    return (
        np.ascontiguousarray(full_poc),
        np.ascontiguousarray(full_vah),
        np.ascontiguousarray(full_val),
    )


@njit(cache=True)
def build_value_area(volume, low, high, poc, total):
    included = volume[poc]
    alo, ahi = poc, poc
    while included * 100 < total * 70:
        above = volume[ahi + 1] if ahi < high else -1
        below = volume[alo - 1] if alo > low else -1
        if above < 0 and below < 0:
            break
        if above >= below:
            ahi += 1
            included += above
        else:
            alo -= 1
            included += below
    return alo, ahi


@njit(cache=True)
def self_test():
    v = np.zeros(11, np.int64)
    for i, x in enumerate((20, 40, 90, 150, 180, 260, 200, 120, 80, 40, 20)):
        v[i] = x
    return build_value_area(v, 0, 10, 5, 1200)


@njit(cache=True)
def build_profile(prices, sizes):
    n = len(prices)
    poc_o = np.empty(n, np.int32)
    vah_o = np.empty(n, np.int32)
    val_o = np.empty(n, np.int32)
    vol = np.zeros(SIZE, np.int64)
    total, low, high, poc = 0, SIZE, -1, -1
    for i in range(n):
        b = prices[i] - BASE
        vol[b] += sizes[i]
        total += sizes[i]
        if b < low:
            low = b
        if b > high:
            high = b
        if poc < 0 or vol[b] > vol[poc]:
            poc = b
        elif vol[b] == vol[poc] and b > poc:
            poc = b
        alo, ahi = build_value_area(vol, low, high, poc, total)
        poc_o[i] = poc + BASE
        val_o[i] = alo + BASE
        vah_o[i] = ahi + BASE
    return poc_o, vah_o, val_o


# ----------------------------------------------------------------- simulate
@njit(cache=True)
def simulate(timestamps, prices, poc, vah, val_area, day_ids,
             entry_start, entry_end, flush_instant, execution_model,
             stop_ticks, lots, tp1_ticks,
             use_poc_rung, use_opp_rung, runner_ticks,
             min_poc_dist, move_be, trail_ticks,
             atr_ticks, atr_mult, tp1_min_ticks, tp1_max_ticks,
             tp2_ticks, tp2_at_poc, tp1_lots, entry_ok):
    n = len(prices)

    trades = np.full((MAX_TRADES, STAT0 + 6), -1, np.int64)
    summary = np.zeros(8, np.int64)

    tc, cash, peak, dd, min_eq = 0, 0, 0, 0, 0
    position, entry, entry_time = 0, 0, 0
    tp1_hit, best_price = 0, 0

    leg_open = np.zeros(MAX_LOTS, np.int64)
    leg_target = np.zeros(MAX_LOTS, np.int64)
    leg_stop = np.zeros(MAX_LOTS, np.int64)
    leg_kind = np.full(MAX_LOTS, -1, np.int64)
    leg_xt = np.full(MAX_LOTS, -1, np.int64)
    leg_xp = np.full(MAX_LOTS, -1, np.int64)
    leg_xr = np.full(MAX_LOTS, -1, np.int64)
    legs = 0

    gross, commission = 0, 0
    bpoc, bvah, bval = 0, 0, 0

    ws, wl, s_arm, l_arm, origin, prev = -1, -1, 0, 0, 0, -1
    cur_day, nday, cooldown = -1, 0, 0

    for i in range(n):
        price = prices[i]
        stamp = timestamps[i]
        day = day_ids[i]
        entered = 0

        if day != cur_day:
            cur_day, nday, cooldown = day, 0, 0

        # ---- session close flush ----
        if position != 0 and stamp >= flush_instant:
            for k in range(legs):
                if leg_open[k] == 1:
                    leg_open[k] = 0
                    leg_xt[k], leg_xp[k], leg_xr[k] = stamp, price, 5
                    ch = (price - entry) * position
                    gross += ch * TICK_CENTS
                    commission += COMMISSION_CENTS
                    cash += ch * TICK_CENTS - COMMISSION_CENTS
            if tc < MAX_TRADES:
                row = trades[tc]
                row[0], row[1], row[2] = entry_time, entry, position
                for k in range(MAX_LOTS):
                    row[LEG_T0 + k] = leg_xt[k] if k < legs else -1
                    row[LEG_P0 + k] = leg_xp[k] if k < legs else -1
                    row[LEG_R0 + k] = leg_xr[k] if k < legs else -1
                row[STAT0], row[STAT0 + 1], row[STAT0 + 2] = bpoc, bvah, bval
                row[STAT0 + 3] = gross
                row[STAT0 + 4] = commission
                row[STAT0 + 5] = gross - commission
                tc += 1
            else:
                summary[1] += 1
            position, legs = 0, 0
            tp1_hit, best_price = 0, 0

        # ---- per-leg exits: stop first, then target ----
        if position != 0:
            for k in range(legs):
                if leg_open[k] != 1:
                    continue
                lstop = leg_stop[k]
                s_hit, s_fill = 0, 0
                if position == 1 and price <= lstop:
                    s_hit, s_fill = 1, min(lstop, price) - STOP_SLIP_TICKS
                elif position == -1 and price >= lstop:
                    s_hit, s_fill = 1, max(lstop, price) + STOP_SLIP_TICKS

                if s_hit == 1:
                    leg_open[k] = 0
                    leg_xt[k], leg_xp[k], leg_xr[k] = stamp, s_fill, 2
                    ch = (s_fill - entry) * position
                    gross += ch * TICK_CENTS
                    commission += COMMISSION_CENTS
                    cash += ch * TICK_CENTS - COMMISSION_CENTS
                    continue

                tgt = leg_target[k]
                hit = 0
                if position == 1:
                    if (execution_model == 0 and price >= tgt) or (
                            execution_model == 1 and price > tgt):
                        hit = 1
                else:
                    if (execution_model == 0 and price <= tgt) or (
                            execution_model == 1 and price < tgt):
                        hit = 1
                if hit == 1:
                    leg_open[k] = 0
                    leg_xt[k], leg_xp[k], leg_xr[k] = stamp, tgt, leg_kind[k]
                    ch = (tgt - entry) * position
                    gross += ch * TICK_CENTS
                    commission += COMMISSION_CENTS
                    cash += ch * TICK_CENTS - COMMISSION_CENTS
                    # breakeven: when TP1 fills, every surviving leg's stop
                    # moves to the entry price.
                    if leg_kind[k] == RUNG_TP1:
                        tp1_hit = 1
                        best_price = tgt
                        if move_be:
                            for j in range(legs):
                                if leg_open[j] == 1:
                                    if ((position == 1 and entry > leg_stop[j])
                                            or (position == -1 and entry < leg_stop[j])):
                                        leg_stop[j] = entry
                        if trail_ticks > 0:
                            tstop = best_price - position * trail_ticks
                            for j in range(legs):
                                if leg_open[j] == 1:
                                    if ((position == 1 and tstop > leg_stop[j])
                                            or (position == -1 and tstop < leg_stop[j])):
                                        leg_stop[j] = tstop

            still = 0
            for k in range(legs):
                if leg_open[k] == 1:
                    still = 1
                    break

            if still == 1 and tp1_hit == 1 and trail_ticks > 0:
                if (position == 1 and price > best_price) or (
                        position == -1 and price < best_price):
                    best_price = price
                    tstop = best_price - position * trail_ticks
                    for j in range(legs):
                        if leg_open[j] == 1:
                            if ((position == 1 and tstop > leg_stop[j])
                                    or (position == -1 and tstop < leg_stop[j])):
                                leg_stop[j] = tstop

            if still == 0:
                if tc < MAX_TRADES:
                    row = trades[tc]
                    row[0], row[1], row[2] = entry_time, entry, position
                    for k in range(MAX_LOTS):
                        row[LEG_T0 + k] = leg_xt[k] if k < legs else -1
                        row[LEG_P0 + k] = leg_xp[k] if k < legs else -1
                        row[LEG_R0 + k] = leg_xr[k] if k < legs else -1
                    row[STAT0], row[STAT0 + 1], row[STAT0 + 2] = bpoc, bvah, bval
                    row[STAT0 + 3] = gross
                    row[STAT0 + 4] = commission
                    row[STAT0 + 5] = gross - commission
                    tc += 1
                else:
                    summary[1] += 1
                position, legs = 0, 0
                tp1_hit, best_price = 0, 0

        # ---- entry ----
        if (position == 0 and i > 0
                and stamp >= entry_start and stamp < entry_end
                and stamp >= cooldown and nday < MAX_DAILY_TRADES):
            sh = 0
            lo = 0
            if (ws >= 0 and prev < ws
                    and ((execution_model == 0 and price >= ws) or (
                         execution_model == 1 and price > ws))):
                sh = 1
            if (wl >= 0 and prev > wl
                    and ((execution_model == 0 and price <= wl) or (
                         execution_model == 1 and price < wl))):
                lo = 1

            if sh == 1 and lo == 1:
                summary[2] += 1
            elif sh == 1 or lo == 1:
                if tc >= MAX_TRADES:
                    summary[1] += 1
                else:
                    is_short = sh == 1
                    cand = ws if is_short else wl
                    cpoc = poc[i - 1]
                    cvah = vah[i - 1]
                    cval = val_area[i - 1]
                    trig = cvah if is_short else cval

                    if abs(cpoc - trig) >= min_poc_dist:
                        # Impulse / trend filter (session VWAP ± σ bands).
                        if entry_ok[i] == 0:
                            continue
                        # Resolve TP1: static ticks, or ATR multiple when warm.
                        eff_tp1 = tp1_ticks
                        if atr_mult > 0.0:
                            a = int(atr_ticks[i])
                            if a <= 0:
                                # ATR not warm yet — skip this cross; stay armed.
                                continue
                            raw = int(round(a * atr_mult))
                            if raw < tp1_min_ticks:
                                raw = tp1_min_ticks
                            if raw > tp1_max_ticks:
                                raw = tp1_max_ticks
                            eff_tp1 = raw

                        entered = 1
                        position = -1 if is_short else 1
                        entry = cand
                        entry_time = stamp
                        bpoc, bvah, bval = cpoc, cvah, cval
                        stop = entry - position * stop_ticks

                        legs = 0
                        leg_open[:] = 0
                        leg_xt[:] = -1
                        leg_xp[:] = -1
                        leg_xr[:] = -1
                        leg_kind[:] = -1
                        leg_target[:] = 0
                        leg_stop[:] = 0
                        tp1_hit, best_price = 0, 0

                        ptarget = entry
                        n_tp1 = tp1_lots if tp1_lots > 0 else 1
                        if n_tp1 > lots:
                            n_tp1 = lots
                        for k in range(lots):
                            rp, kind = 0, -1
                            if k < n_tp1 and eff_tp1 > 0:
                                # One or more lots share the same TP1 price.
                                rp = entry + position * eff_tp1
                                kind = RUNG_TP1
                            elif k == n_tp1 and tp2_at_poc == 1:
                                rp, kind = cpoc, RUNG_POC
                            elif k == n_tp1 and tp2_ticks > 0:
                                rp = entry + position * (eff_tp1 + tp2_ticks)
                                kind = RUNG_FIXED
                            elif k == n_tp1 and use_poc_rung:
                                rp, kind = cpoc, RUNG_POC
                            elif use_opp_rung and (
                                    k >= n_tp1
                                    or (k >= 1 and use_poc_rung)):
                                rp = cval if position == -1 else cvah
                                kind = RUNG_OPP
                            elif runner_ticks > 0:
                                rp = entry + position * runner_ticks
                                kind = RUNG_FIXED
                            else:
                                break

                            gap = (rp - ptarget) * position
                            # Allow multiple lots at the exact same TP1.
                            same_tp1 = (
                                kind == RUNG_TP1 and gap == 0 and legs > 0
                                and leg_kind[legs - 1] == RUNG_TP1
                            )
                            if gap < MIN_RUNG_GAP_TICKS and not same_tp1:
                                if runner_ticks > 0 and kind == RUNG_OPP:
                                    rp = entry + position * runner_ticks
                                    kind = RUNG_FIXED
                                    gap = (rp - ptarget) * position
                                if gap < MIN_RUNG_GAP_TICKS:
                                    continue

                            leg_target[k] = rp
                            leg_kind[k] = kind
                            leg_open[k] = 1
                            leg_stop[k] = stop
                            legs += 1
                            ptarget = rp

                        if legs == 0:
                            position = 0
                            entered = 0
                        else:
                            gross = 0
                            commission = legs * COMMISSION_CENTS
                            cash -= commission
                            nday += 1
                            cooldown = stamp + COOLDOWN_100NS

        # ---- crossing-state update ----
        if i > 0 and entered == 0:
            ppoc = poc[i - 1]
            cu = origin == -1 and price > ppoc
            cd = origin == 1 and price < ppoc
            if price > ppoc:
                origin = 1
            elif price < ppoc:
                origin = -1
            if cu:
                s_arm, l_arm = 1, 0
            elif cd:
                l_arm, s_arm = 1, 0
            if poc[i] != ppoc:
                if price > poc[i]:
                    origin = 1
                elif price < poc[i]:
                    origin = -1
                else:
                    origin = 0
        elif i == 0:
            origin = 1 if price > poc[i] else (-1 if price < poc[i] else 0)

        if entered == 1:
            s_arm, l_arm = 0, 0

        prev = price
        ws = -1
        wl = -1
        if position == 0 and s_arm == 1 and price < vah[i]:
            ws = vah[i]
        if position == 0 and l_arm == 1 and price > val_area[i]:
            wl = val_area[i]

        openval = 0
        for k in range(legs):
            if leg_open[k] == 1:
                openval += (price - entry) * position * TICK_CENTS
                openval -= COMMISSION_CENTS
        eq = cash + openval
        if eq > peak:
            peak = eq
        if peak - eq > dd:
            dd = peak - eq
        if eq < min_eq:
            min_eq = eq

    # ---- end-of-file force close ----
    if position != 0:
        fp = prices[n - 1]
        ft = timestamps[n - 1]
        fill = fp - STOP_SLIP_TICKS if position == 1 else fp + STOP_SLIP_TICKS
        for k in range(legs):
            if leg_open[k] == 1:
                leg_open[k] = 0
                leg_xt[k], leg_xp[k], leg_xr[k] = ft, fill, 4
                ch = (fill - entry) * position
                gross += ch * TICK_CENTS
                commission += COMMISSION_CENTS
                cash += ch * TICK_CENTS - COMMISSION_CENTS
        if tc < MAX_TRADES:
            row = trades[tc]
            row[0], row[1], row[2] = entry_time, entry, position
            for k in range(MAX_LOTS):
                row[LEG_T0 + k] = leg_xt[k] if k < legs else -1
                row[LEG_P0 + k] = leg_xp[k] if k < legs else -1
                row[LEG_R0 + k] = leg_xr[k] if k < legs else -1
            row[STAT0], row[STAT0 + 1], row[STAT0 + 2] = bpoc, bvah, bval
            row[STAT0 + 3] = gross
            row[STAT0 + 4] = commission
            row[STAT0 + 5] = gross - commission
            tc += 1
        else:
            summary[1] += 1

    summary[0], summary[3], summary[4], summary[5] = tc, cash, dd, min_eq
    return trades[:tc], summary


# --------------------------------------------------------------------- main
def load_sessions(path, limit):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sel = [r for r in manifest["sessions"] if r["include_in_primary"]]
    if limit is not None:
        sel = sel[:limit]
    return sel


def empty_bucket():
    return {"net": 0, "trades": 0, "wins": 0, "losses": 0,
            "win_cents": 0, "loss_cents": 0, "dd": 0, "days": []}


def analyze_prop_firm(
    days,
    dd_limit=PROP_DD_LIMIT_CENTS,
    target=PROP_TARGET_CENTS,
    cap_floor_at_zero=True,
):
    """Evaluate EOD trailing drawdown and prop-firm pass rate.

    days: list of (day_str, day_net_cents, intraday_dd_cents, min_eq_cents)
    cap_floor_at_zero: True if the EOD trailing floor stops rising at $0 P&L
                       once best EOD profit reaches dd_limit.
    """
    cum_pnl = 0
    best_eod = 0
    max_eod_dd = 0
    max_daily_dd = 0
    worst_day = 0
    curve = []

    for day, dnet, intraday_dd, min_eq in days:
        intraday_low = cum_pnl + min_eq
        max_eod_dd = max(max_eod_dd, best_eod - intraday_low)
        max_daily_dd = max(max_daily_dd, intraday_dd)
        worst_day = min(worst_day, dnet)

        cum_pnl += dnet
        best_eod = max(best_eod, cum_pnl)
        max_eod_dd = max(max_eod_dd, best_eod - cum_pnl)
        curve.append((day, dnet, cum_pnl, best_eod, best_eod - intraday_low))

    def run_eval(start_idx):
        bal, hwm = 0, 0
        for step, (_, dnet, _, min_eq) in enumerate(days[start_idx:], 1):
            floor = min(hwm - dd_limit, 0) if cap_floor_at_zero else (hwm - dd_limit)
            if bal + min_eq <= floor:
                return "FAIL", step
            bal += dnet
            if bal <= floor:
                return "FAIL", step
            if bal >= target:
                return "PASS", step
            hwm = max(hwm, bal)
        return "OPEN", len(days) - start_idx

    d1_status, d1_days = run_eval(0) if days else ("OPEN", 0)
    passes = fails = 0
    for idx in range(len(days)):
        st, _ = run_eval(idx)
        if st == "PASS":
            passes += 1
        elif st == "FAIL":
            fails += 1
    resolved = passes + fails
    pass_rate = round(passes / resolved, 3) if resolved > 0 else None

    return {
        "max_daily_dd": round(max_daily_dd / 100, 2),
        "worst_day": round(worst_day / 100, 2),
        "max_eod_dd": round(max_eod_dd / 100, 2),
        "eval_d1": f"{d1_status}({d1_days}d)",
        "pass_rate": pass_rate,
        "curve": curve,
    }


def select_variants(family: str | None, only: list[str] | None):
    catalog = {
        "legacy": LEGACY_VARIANTS,
        "static_tp1": STATIC_TP1_VARIANTS,
        "atr_tp1": ATR_TP1_VARIANTS,
        "three_lot": THREE_LOT_VARIANTS,
        "scale_out": SCALE_OUT_VARIANTS,
        "vwap_filter": VWAP_FILTER_VARIANTS,
        "tp_study": (
            STATIC_TP1_VARIANTS + ATR_TP1_VARIANTS[1:] + THREE_LOT_VARIANTS[1:]
            + SCALE_OUT_VARIANTS[1:]
        ),
        "all": (
            LEGACY_VARIANTS
            + STATIC_TP1_VARIANTS[1:]
            + ATR_TP1_VARIANTS[1:]
            + THREE_LOT_VARIANTS[1:]
            + SCALE_OUT_VARIANTS[1:]
            + VWAP_FILTER_VARIANTS[1:]
        ),
    }
    # De-dupe by name, first wins (keeps control once in tp_study).
    if family:
        if family not in catalog:
            raise SystemExit(
                f"unknown family {family!r}; choose from {sorted(catalog)}"
            )
        raw = catalog[family]
    else:
        raw = LEGACY_VARIANTS
    seen = set()
    variants = []
    for v in raw:
        if v.name in seen:
            continue
        seen.add(v.name)
        variants.append(v)
    if only:
        wanted = set(only)
        variants = [v for v in variants if v.name in wanted]
        if not variants:
            names = ", ".join(sorted(seen) or [v.name for v in LEGACY_VARIANTS])
            raise SystemExit(f"no variants matched {only}\navailable: {names}")
    return variants


def neighbor_median_report(rows, family_prefix_pts):
    """Plateau check for static TP1_*pt variants (touch fill)."""
    by_pts = {}
    for r in rows:
        if r["fill"] != "touch":
            continue
        name = r["variant"]
        if name.startswith("TP1_") and name.endswith("pt"):
            try:
                pts = int(name[4:-2])
            except ValueError:
                continue
            by_pts[pts] = r["net_dollars"]
        elif name == "K_two_lot_60":
            by_pts[20] = r["net_dollars"]  # control lands on 20pt cell
    if len(by_pts) < 3:
        return
    pts_sorted = sorted(by_pts)
    print("\nstatic TP1 plateau (touch net $, neighbor median):")
    print(f"{'pts':>5}{'net':>12}{'nbr_med':>12}")
    for i, pts in enumerate(pts_sorted):
        nets = [by_pts[pts]]
        if i > 0:
            nets.append(by_pts[pts_sorted[i - 1]])
        if i + 1 < len(pts_sorted):
            nets.append(by_pts[pts_sorted[i + 1]])
        med = float(np.median(nets))
        print(f"{pts:>5}{by_pts[pts]:>12,.2f}{med:>12,.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=Path, default=Path("output/sessions"))
    ap.add_argument("--out", type=Path, default=Path("output/sweep"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="*",
                    help="variant names to run, e.g. O_be_30")
    ap.add_argument(
        "--family",
        choices=(
            "legacy", "static_tp1", "atr_tp1", "three_lot", "scale_out",
            "vwap_filter", "tp_study", "all",
        ),
        default="legacy",
        help="variant family (default: legacy)",
    )
    ap.add_argument(
        "--fill",
        choices=("both", "touch", "trade_through"),
        default="both",
        help="which fill models to run",
    )
    args = ap.parse_args()

    sel = load_sessions(args.sessions / "manifest.json", args.limit)
    if not sel:
        raise SystemExit("No primary sessions found.")
    args.out.mkdir(parents=True, exist_ok=True)

    print("value-area self-test...", flush=True)
    if tuple(self_test()) != (3, 7):
        raise SystemExit(f"self-test failed: {self_test()}")
    print("self-test passed\n", flush=True)

    variants = select_variants(args.family, args.only)
    fill_models = FILL_MODELS
    if args.fill == "touch":
        fill_models = ((0, "touch"),)
    elif args.fill == "trade_through":
        fill_models = ((1, "trade_through"),)

    need_atr = any(v.atr_mult > 0 for v in variants)
    need_vwap = any(v.vwap_sigma > 0 for v in variants)
    print(
        f"family={args.family} variants={len(variants)} "
        f"sessions={len(sel)} atr={need_atr} vwap={need_vwap}",
        flush=True,
    )

    buckets = {}
    for v in variants:
        for _, fill in fill_models:
            buckets[f"{v.name}|{fill}"] = empty_bucket()

    started = time.perf_counter()
    for si, s in enumerate(sel, 1):
        day = s["date_eastern"]
        data = np.load(args.sessions / s["file"])
        prices = np.ascontiguousarray(data["last"], dtype=np.int32)
        if int(prices.min()) < BASE or int(prices.max()) >= BASE + SIZE:
            raise SystemExit(f"{day}: price out of range")
        stamps = np.ascontiguousarray(data["ts_100ns"], dtype=np.int64)
        sizes = np.ascontiguousarray(data["volume"], dtype=np.int64)
        day_ids = np.fromiter(
            (eastern_day_id(x) for x in stamps),
            dtype=np.int64, count=len(stamps))

        date = datetime.fromisoformat(day).date()
        ee = timestamp_100ns(datetime.combine(date, ENTRY_END, tzinfo=EASTERN))
        fl = timestamp_100ns(datetime.combine(date, RTH_END, tzinfo=EASTERN))

        poc, vah, val = rth_profile_for_session(prices, sizes, stamps)
        atr = (
            atr_15m_ticks(prices, stamps)
            if need_atr
            else np.zeros(len(prices), dtype=np.int32)
        )
        # Cache VWAP masks by (sigma, persist) so variants share work.
        vwap_cache = {}

        for v in variants:
            es = timestamp_100ns(datetime.combine(
                date, clock(v.entry_hour, v.entry_minute), tzinfo=EASTERN))
            if v.vwap_sigma > 0:
                key = (float(v.vwap_sigma), int(v.vwap_persist_secs))
                if key not in vwap_cache:
                    vwap_cache[key] = session_vwap_entry_ok(
                        prices, sizes, stamps,
                        sigma=key[0], persist_secs=key[1],
                    )
                entry_ok = vwap_cache[key]
            else:
                entry_ok = np.ones(len(prices), dtype=np.int8)
            for model, fill in fill_models:
                trades, summary = simulate(
                    stamps, prices, poc, vah, val, day_ids,
                    es, ee, fl, model,
                    v.stop_ticks, v.lots, v.tp1_ticks,
                    v.use_poc_rung, v.use_opp_rung, v.runner_ticks,
                    v.min_poc_dist, v.move_be, v.trail_ticks,
                    atr, float(v.atr_mult), TP1_MIN_TICKS, TP1_MAX_TICKS,
                    int(v.tp2_ticks), 1 if v.tp2_at_poc else 0,
                    int(getattr(v, "tp1_lots", 1) or 1),
                    entry_ok,
                )
                b = buckets[f"{v.name}|{fill}"]
                dnet = 0
                for t in trades:
                    net = int(t[NET_COL])
                    dnet += net
                    b["trades"] += 1
                    if net > 0:
                        b["wins"] += 1
                        b["win_cents"] += net
                    else:
                        b["losses"] += 1
                        b["loss_cents"] += -net
                b["net"] += dnet
                b["dd"] = max(b["dd"], int(summary[4]))
                b["days"].append((day, dnet, int(summary[4]), int(summary[5])))
        print(f"[{si}/{len(sel)}] {day}", flush=True)

    equity_rows = []
    rows = []
    for key, b in buckets.items():
        n = b["trades"]
        name, fill = key.split("|")
        pf = analyze_prop_firm(b["days"])
        for day, dnet, cum_pnl, best_eod, eod_dd in pf["curve"]:
            equity_rows.append(dict(
                variant=name, fill=fill, date=day,
                day_pnl=round(dnet / 100, 2),
                cum_pnl=round(cum_pnl / 100, 2),
                best_eod=round(best_eod / 100, 2),
                eod_dd=round(eod_dd / 100, 2),
            ))
        if n == 0:
            rows.append(dict(
                variant=name, fill=fill, trades=0, net_dollars=0,
                win_rate=None, avg_win=None, avg_loss=None,
                expectancy=None, profit_factor=None,
                max_daily_dd=0, worst_day=0, max_eod_dd=0,
                eval_d1=pf["eval_d1"], pass_rate=None,
            ))
            continue
        w = max(b["wins"], 1)
        l = max(b["losses"], 1)
        aw = b["win_cents"] / w / 100
        al = b["loss_cents"] / l / 100
        rows.append(dict(
            variant=name, fill=fill, trades=n,
            net_dollars=round(b["net"] / 100, 2),
            win_rate=round(b["wins"] / n, 3),
            avg_win=round(aw, 2),
            avg_loss=round(al, 2),
            expectancy=round(b["net"] / n / 100, 2),
            profit_factor=round(b["win_cents"] / max(b["loss_cents"], 1), 2),
            max_daily_dd=pf["max_daily_dd"],
            worst_day=pf["worst_day"],
            max_eod_dd=pf["max_eod_dd"],
            eval_d1=pf["eval_d1"],
            pass_rate=pf["pass_rate"],
        ))

    cols = [
        "variant", "fill", "trades", "net_dollars", "win_rate",
        "avg_win", "avg_loss", "expectancy", "profit_factor",
        "max_daily_dd", "worst_day", "max_eod_dd", "eval_d1", "pass_rate",
    ]
    with (args.out / "sweep_summary.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    eq_cols = ["variant", "fill", "date", "day_pnl", "cum_pnl",
               "best_eod", "eod_dd"]
    with (args.out / "daily_equity.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=eq_cols)
        wr.writeheader()
        wr.writerows(equity_rows)

    print("\n" + "=" * 106)
    has_tt = any(r["fill"] == "trade_through" for r in rows)
    if has_tt:
        print(f"{'variant':<18}{'touch net':>11}{'t/t net':>11}{'trades':>7}"
              f"{'win%':>7}{'PF':>6}{'dayDD':>9}{'eodDD':>9}{'eval_d1':>12}{'pass%':>8}")
    else:
        print(f"{'variant':<18}{'net':>11}{'trades':>7}"
              f"{'win%':>7}{'PF':>6}{'dayDD':>9}{'eodDD':>9}{'eval_d1':>12}{'pass%':>8}")
    print("-" * 106)
    by_var = {}
    for r in rows:
        by_var.setdefault(r["variant"], {})[r["fill"]] = r
    for v in variants:
        fills = by_var[v.name]
        t = fills.get("touch") or next(iter(fills.values()))
        pr = f"{t['pass_rate'] * 100:.1f}%" if t["pass_rate"] is not None else "n/a"
        if has_tt:
            tt = fills.get("trade_through", t)
            print(f"{v.name:<18}{t['net_dollars']:>11,.2f}"
                  f"{tt['net_dollars']:>11,.2f}"
                  f"{t['trades']:>7}"
                  f"{(t['win_rate'] or 0) * 100:>6.1f}%"
                  f"{(t['profit_factor'] or 0):>6.2f}"
                  f"{t['max_daily_dd']:>9,.2f}"
                  f"{t['max_eod_dd']:>9,.2f}"
                  f"{t['eval_d1']:>12}"
                  f"{pr:>8}")
        else:
            print(f"{v.name:<18}{t['net_dollars']:>11,.2f}"
                  f"{t['trades']:>7}"
                  f"{(t['win_rate'] or 0) * 100:>6.1f}%"
                  f"{(t['profit_factor'] or 0):>6.2f}"
                  f"{t['max_daily_dd']:>9,.2f}"
                  f"{t['max_eod_dd']:>9,.2f}"
                  f"{t['eval_d1']:>12}"
                  f"{pr:>8}")
    print("=" * 106)
    if args.family in ("static_tp1", "tp_study", "all"):
        neighbor_median_report(rows, None)
    print(f"\nelapsed {time.perf_counter() - started:.1f}s")
    print(f"written: {args.out / 'sweep_summary.csv'}")
    print(f"written: {args.out / 'daily_equity.csv'}")


if __name__ == "__main__":
    main()