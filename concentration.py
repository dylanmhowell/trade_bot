"""Concentration check: how much of each variant's net comes from its best trades."""

import argparse
import json
import sys
from datetime import datetime, time as clock
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from sweep import (
    EASTERN, ENTRY_START, ENTRY_END, RTH_END,
    MAX_LOTS, LEG_T0, NET_COL,
    timestamp_100ns, eastern_day_id, price_text,
    rth_profile_for_session, simulate,
    VARIANTS, load_sessions,
)


def run(selected, variant, sessions_dir):
    """Return list of (day, net_cents, legs) for one variant, touch fills only."""
    rows = []
    for s in selected:
        day = s["date_eastern"]
        data = np.load(sessions_dir / s["file"])
        prices = np.ascontiguousarray(data["last"], dtype=np.int32)
        stamps = np.ascontiguousarray(data["ts_100ns"], dtype=np.int64)
        sizes = np.ascontiguousarray(data["volume"], dtype=np.int64)
        day_ids = np.fromiter(
            (eastern_day_id(x) for x in stamps),
            dtype=np.int64, count=len(stamps))

        date = datetime.fromisoformat(day).date()
        es = timestamp_100ns(datetime.combine(date, clock(variant.entry_hour, variant.entry_minute), tzinfo=EASTERN))
        ee = timestamp_100ns(datetime.combine(date, ENTRY_END, tzinfo=EASTERN))
        fl = timestamp_100ns(datetime.combine(date, RTH_END, tzinfo=EASTERN))
        poc, vah, val = rth_profile_for_session(prices, sizes, stamps)

        trades, _ = simulate(
            stamps, prices, poc, vah, val, day_ids,
            es, ee, fl, 0,
            variant.stop_ticks, variant.lots, variant.tp1_ticks,
            variant.use_poc_rung, variant.use_opp_rung,
            variant.runner_ticks, variant.min_poc_dist, variant.move_be,
        )
        for t in trades:
            legs = sum(1 for k in range(MAX_LOTS) if t[LEG_T0 + k] >= 0)
            rows.append((day, int(t[NET_COL]), legs))
    return rows


def report(name, rows):
    if not rows:
        print(f"{name:<20}  no trades")
        return
    nets = sorted((r[1] for r in rows), reverse=True)
    total = sum(nets)
    n = len(nets)
    top1, top5, top10 = sum(nets[:1]), sum(nets[:5]), sum(nets[:10])
    rest10 = total - top10

    def pct(x):
        return (x / total * 100) if total else float("nan")

    # daily aggregation
    by_day = {}
    for day, net, _ in rows:
        by_day[day] = by_day.get(day, 0) + net
    day_nets = sorted(by_day.values(), reverse=True)
    dtotal = sum(day_nets)
    dtop5 = sum(day_nets[:5])

    # leg distribution
    leg_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for _, _, lg in rows:
        leg_counts[lg] = leg_counts.get(lg, 0) + 1
    avg_legs = sum(c * k for k, c in leg_counts.items()) / n

    print(f"\n{name}")
    print(f"  trades {n}   net ${total / 100:,.2f}   "
          f"avg/legs {avg_legs:.2f}   "
          f"legs {leg_counts[1]}L/{leg_counts[2]}L/{leg_counts[3]}L")
    print(f"  trade-level:  best ${nets[0] / 100:,.2f} | "
          f"top1 {pct(top1):5.1f}% | "
          f"top5 {pct(top5):5.1f}% | "
          f"top10 {pct(top10):5.1f}% | "
          f"rest ${rest10 / 100:,.2f}")
    print(f"  day-level:    best ${day_nets[0] / 100:,.2f} | "
          f"top5 days {pct(dtop5):5.1f}% | "
          f"{len(by_day)} sessions traded")
    flag = "  <<< CONCENTRATED" if pct(top5) > 60 else ""
    print(f"  verdict:      {'PASS' if pct(top5) <= 60 else 'FAIL'} "
          f"(top5 <= 60% of net){flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=Path, default=Path("output/sessions"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="*",
                    help="variant names (default: the profitable shortlist)")
    ap.add_argument("--csv", type=Path,
                    help="also write per-trade rows here")
    args = ap.parse_args()

    selected = load_sessions(args.sessions / "manifest.json", args.limit)
    if not selected:
        raise SystemExit("No primary sessions found.")

    shortlist = ["J_opp_only_60", "K_two_lot_60", "L_opp_90_stop", "M_opp_alone"]
    wanted = args.only or shortlist
    variants = [v for v in VARIANTS if v.name in wanted]
    if not variants:
        names = ", ".join(v.name for v in VARIANTS)
        raise SystemExit(f"No variants matched.\nAvailable: {names}")

    print(f"Concentration over {len(selected)} sessions, "
          f"{len(variants)} variant(s), touch fills\n")

    all_rows = []
    for v in variants:
        rows = run(selected, v, args.sessions)
        report(v.name, rows)
        for day, net, legs in rows:
            all_rows.append((day, v.name, legs, net))

    if args.csv:
        with args.csv.open("w", newline="") as f:
            f.write("date,variant,legs,net_dollars\n")
            for day, name, legs, net in all_rows:
                f.write(f"{day},{name},{legs},{net / 100:.2f}\n")
        print(f"\nwrote {args.csv}")

    print("\nReading: top5 <= 60% of net is the pass bar. "
          "Fail means the edge is a handful of trades, not a strategy.")


if __name__ == "__main__":
    main()