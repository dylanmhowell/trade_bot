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
    note: str = ""


VARIANTS = [
    Variant("A_control", stop_ticks=120, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=320,
            note="baseline control, no breakeven, 10:15 entry"),
    Variant("O_be_30", stop_ticks=120, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=320,
            move_be=True, note="A + breakeven after TP1"),
    Variant("P_late_30", stop_ticks=120, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=320,
            entry_hour=11, entry_minute=0, note="A + 11:00 entry floor"),
    Variant("Q_be_late_30", stop_ticks=120, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=320,
            move_be=True, entry_hour=11, entry_minute=0,
            note="A + breakeven + 11:00 floor"),
    Variant("K_two_lot_60", stop_ticks=240, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=0,
            note="2 lots: TP1 20pt banked, 1 lot to opp VA edge, 60pt stop"),
    Variant("R_be_60", stop_ticks=240, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=0,
            move_be=True, note="K + breakeven after TP1"),
    Variant("S_be_late_60", stop_ticks=240, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=0,
            move_be=True, entry_hour=11, entry_minute=0,
            note="K + breakeven + 11:00 floor"),
    # Trailing stop variants
    Variant("T_trail20_60", stop_ticks=240, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=0,
            trail_ticks=80,
            note="K + 20pt trail after TP1 (locks BE at TP1)"),
    Variant("U_trail30_60", stop_ticks=240, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=0,
            trail_ticks=120,
            note="K + 30pt trail after TP1 (tightens to -10pt at TP1)"),
    Variant("V_be_trail30_60", stop_ticks=240, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=0,
            move_be=True, trail_ticks=120,
            note="K + BE at TP1, then 30pt trail once > +30pt"),
    Variant("W_be_trail40_60", stop_ticks=240, lots=2, tp1_ticks=80,
            use_poc_rung=False, use_opp_rung=True, runner_ticks=0,
            move_be=True, trail_ticks=160,
            note="K + BE at TP1, then 40pt trail once > +40pt"),
]

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
             min_poc_dist, move_be, trail_ticks):
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
                        for k in range(lots):
                            rp, kind = 0, -1
                            if k == 0:
                                if tp1_ticks > 0:
                                    rp = entry + position * tp1_ticks
                                    kind = RUNG_TP1
                                elif use_opp_rung:
                                    rp = cval if position == -1 else cvah
                                    kind = RUNG_OPP
                                else:
                                    break
                            elif k == 1 and use_poc_rung:
                                rp, kind = cpoc, RUNG_POC
                            elif ((k >= 1 and use_poc_rung and use_opp_rung)
                                  or (k == 1 and use_opp_rung)):
                                rp = cval if position == -1 else cvah
                                kind = RUNG_OPP
                            elif runner_ticks > 0:
                                rp = entry + position * runner_ticks
                                kind = RUNG_FIXED
                            else:
                                break

                            gap = (rp - ptarget) * position
                            if gap < MIN_RUNG_GAP_TICKS:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=Path, default=Path("output/sessions"))
    ap.add_argument("--out", type=Path, default=Path("output/sweep"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="*",
                    help="variant names to run, e.g. O_be_30")
    args = ap.parse_args()

    sel = load_sessions(args.sessions / "manifest.json", args.limit)
    if not sel:
        raise SystemExit("No primary sessions found.")
    args.out.mkdir(parents=True, exist_ok=True)

    print("value-area self-test...", flush=True)
    if tuple(self_test()) != (3, 7):
        raise SystemExit(f"self-test failed: {self_test()}")
    print("self-test passed\n", flush=True)

    variants = VARIANTS
    if args.only:
        wanted = set(args.only)
        variants = [v for v in VARIANTS if v.name in wanted]
        if not variants:
            names = ", ".join(v.name for v in VARIANTS)
            raise SystemExit(f"no variants matched {args.only}\navailable: {names}")

    buckets = {}
    for v in variants:
        for _, fill in FILL_MODELS:
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

        for v in variants:
            es = timestamp_100ns(datetime.combine(
                date, clock(v.entry_hour, v.entry_minute), tzinfo=EASTERN))
            for model, fill in FILL_MODELS:
                trades, summary = simulate(
                    stamps, prices, poc, vah, val, day_ids,
                    es, ee, fl, model,
                    v.stop_ticks, v.lots, v.tp1_ticks,
                    v.use_poc_rung, v.use_opp_rung, v.runner_ticks,
                    v.min_poc_dist, v.move_be, v.trail_ticks,
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
    print(f"{'variant':<18}{'touch net':>11}{'t/t net':>11}{'trades':>7}"
          f"{'win%':>7}{'PF':>6}{'dayDD':>9}{'eodDD':>9}{'eval_d1':>12}{'pass%':>8}")
    print("-" * 106)
    by_var = {}
    for r in rows:
        by_var.setdefault(r["variant"], {})[r["fill"]] = r
    for v in variants:
        t = by_var[v.name]["touch"]
        tt = by_var[v.name]["trade_through"]
        pr = f"{t['pass_rate'] * 100:.1f}%" if t["pass_rate"] is not None else "n/a"
        print(f"{v.name:<18}{t['net_dollars']:>11,.2f}"
              f"{tt['net_dollars']:>11,.2f}"
              f"{t['trades']:>7}"
              f"{(t['win_rate'] or 0) * 100:>6.1f}%"
              f"{(t['profit_factor'] or 0):>6.2f}"
              f"{t['max_daily_dd']:>9,.2f}"
              f"{t['max_eod_dd']:>9,.2f}"
              f"{t['eval_d1']:>12}"
              f"{pr:>8}")
    print("=" * 106)
    print(f"\nelapsed {time.perf_counter() - started:.1f}s")
    print(f"written: {args.out / 'sweep_summary.csv'}")
    print(f"written: {args.out / 'daily_equity.csv'}")


if __name__ == "__main__":
    main()