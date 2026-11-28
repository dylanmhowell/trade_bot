"""Diagnostics: A-reproducibility, leg counts, concentration, backouts."""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from sweep import (
    BASE, SIZE, MAX_LOTS, EASTERN,
    ENTRY_START, ENTRY_END, RTH_END,
    timestamp_100ns, eastern_day_id,
    rth_profile_for_session, simulate,
    VARIANTS, load_sessions,
    LEG_T0, NET_COL,
)


def run_variant(selected, variant, *, flush_to_close, gap_ticks, sessions_dir):
    """Return (net_cents, trade_count, leg_distribution, per_trade_nets)."""
    leg_dist = {1: 0, 2: 0, 3: 0, 4: 0}
    per_trade = []
    total_cents = 0

    for s in selected:
        day = s["date_eastern"]
        data = np.load(sessions_dir / s["file"])
        prices = np.ascontiguousarray(data["last"], dtype=np.int32)
        stamps = np.ascontiguousarray(data["ts_100ns"], dtype=np.int64)
        sizes = np.ascontiguousarray(data["volume"], dtype=np.int64)
        day_ids = np.fromiter(
            (eastern_day_id(x) for x in stamps),
            dtype=np.int64, count=len(stamps),
        )

        date = datetime.fromisoformat(day).date()
        es = timestamp_100ns(datetime.combine(date, ENTRY_START, tzinfo=EASTERN))
        ee = timestamp_100ns(datetime.combine(date, ENTRY_END, tzinfo=EASTERN))
        if flush_to_close:
            fl = timestamp_100ns(datetime.combine(date, RTH_END, tzinfo=EASTERN))
        else:
            fl = 2**62

        poc, vah, val = rth_profile_for_session(prices, sizes, stamps)

        # Patch the module constant so gap_ticks reaches simulate().
        # Requires MIN_RUNG_GAP_TICKS to be a literal in the compiled code,
        # so use Option B (set it before the run) rather than this path if
        # numba has already specialized. See note below.
        import sweep
        saved_gap = sweep.MIN_RUNG_GAP_TICKS
        sweep.MIN_RUNG_GAP_TICKS = gap_ticks
        try:
            atr = np.zeros(len(prices), dtype=np.int32)
            trades, _summary = simulate(
                stamps, prices, poc, vah, val, day_ids,
                es, ee, fl, 0,
                variant.stop_ticks, variant.lots, variant.tp1_ticks,
                variant.use_poc_rung, variant.use_opp_rung,
                variant.runner_ticks, variant.min_poc_dist,
                getattr(variant, "move_be", False),
                getattr(variant, "trail_ticks", 0),
                atr, float(getattr(variant, "atr_mult", 0.0)),
                60, 200,
                int(getattr(variant, "tp2_ticks", 0)),
                1 if getattr(variant, "tp2_at_poc", False) else 0,
                int(getattr(variant, "tp1_lots", 1) or 1),
                np.ones(len(prices), dtype=np.int8),
                int(getattr(variant, "opp_pull_ticks", 0) or 0),
                1 if getattr(variant, "be_on_poc", False) else 0,
            )
        finally:
            sweep.MIN_RUNG_GAP_TICKS = saved_gap

        for t in trades:
            net = int(t[NET_COL])
            total_cents += net
            per_trade.append(net)
            legs = sum(1 for k in range(MAX_LOTS) if t[LEG_T0 + k] >= 0)
            leg_dist[legs] = leg_dist.get(legs, 0) + 1

    return total_cents, len(per_trade), leg_dist, per_trade


def concentration(per_trade, label):
    if not per_trade:
        print(f"  {label:<20} no trades")
        return None
    ordered = sorted(per_trade, reverse=True)
    total = sum(ordered)
    top5 = sum(ordered[:5])
    top10 = sum(ordered[:10])
    pct5 = top5 / total * 100 if total else 0.0
    pct10 = top10 / total * 100 if total else 0.0
    rest = total - top10
    print(
        f"  {label:<20} net ${total / 100:>10,.2f} | "
        f"top5 ${top5 / 100:>9,.2f} ({pct5:5.1f}%) | "
        f"top10 ${top10 / 100:>9,.2f} ({pct10:5.1f}%) | "
        f"rest ${rest / 100:>9,.2f}"
    )
    return pct5


def main():
    sessions_dir = Path("output/sessions")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    selected = load_sessions(sessions_dir / "manifest.json", limit)
    print(f"Diagnostics over {len(selected)} sessions\n")

    names = ("A_baseline", "J_opp_only_60", "K_two_lot_60",
             "L_opp_90_stop", "N_two_lot_90")
    budget = {v.name: v for v in VARIANTS if v.name in names}

    print("=" * 78)
    print("1. A-REPRODUCIBILITY")
    print("=" * 78)
    print("  original summary.json: touch -2,348.50 / t/t +45.00 (173/172 trades)")
    print()
    checks = [
        ("flush ON,  gap  20 (current sweep)", True, 20),
        ("flush OFF, gap  20", False, 20),
        ("flush ON,  gap   0", True, 0),
        ("flush OFF, gap   0", False, 0),
    ]
    for label, flush, gap in checks:
        net, n, dist, _ = run_variant(
            selected, budget["A_baseline"], flush_to_close=flush,
            gap_ticks=gap, sessions_dir=sessions_dir,
        )
        print(f"  {label:<38} net ${net / 100:>10,.2f}  trades {n:>4}")
    print()

    print("=" * 78)
    print("2. LEG-COUNT DISTRIBUTION (current sweep settings)")
    print("=" * 78)
    print(f"  {'variant':<20} {'net':>12} {'trades':>7} "
          f"{'1leg':>6} {'2leg':>6} {'3leg':>6}")
    print("  " + "-" * 62)
    cached = {}
    for name in names:
        if name not in budget:
            continue
        net, n, dist, per_trade = run_variant(
            selected, budget[name], flush_to_close=True,
            gap_ticks=20, sessions_dir=sessions_dir,
        )
        cached[name] = (net, n, dist, per_trade)
        print(f"  {name:<20} ${net / 100:>11,.2f} {n:>7} "
              f"{dist.get(1, 0):>6} {dist.get(2, 0):>6} {dist.get(3, 0):>6}")
    print()

    print("=" * 78)
    print("3. CONCENTRATION")
    print("=" * 78)
    flags = []
    for name in names:
        if name not in cached:
            continue
        net, n, dist, per_trade = cached[name]
        pct5 = concentration(per_trade, name)
        if pct5 is not None and pct5 > 60.0:
            flags.append((name, pct5))
    print()
    if flags:
        print("  WARNING - net >60% concentrated in 5 trades:")
        for name, pct in flags:
            print(f"    {name}: {pct:.1f}% from top 5")
    else:
        print("  No variant exceeds the 60% top-5 concentration threshold.")
    print()

    print("=" * 78)
    print("4. STOP-WIDTH / LOTS CROSS-TAB")
    print("=" * 78)
    print("  Does the improvement from adding a second lot and from widening")
    print("  the stop combine, or are they substitutes?")
    print()
    print(f"  {'':<14}{'60pt stop':>16}{'90pt stop':>16}")
    print("  " + "-" * 46)
    grid = {
        (1, 240): "J_opp_only_60",
        (2, 240): "K_two_lot_60",
        (1, 360): "L_opp_90_stop",
        (2, 360): "N_two_lot_90",
    }
    for lots in (1, 2):
        label = f"{lots} lot"
        cells = []
        for stop in (240, 360):
            name = grid.get((lots, stop))
            if name and name in cached:
                net, n, _, _ = cached[name]
                cells.append(f"${net / 100:>8,.2f} ({n})")
            else:
                cells.append(f"{'--':>12}")
        print(f"  {label:<14}{cells[0]:>16}{cells[1]:>16}")
    print()


if __name__ == "__main__":
    main()