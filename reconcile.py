"""Reconcile the patched sweep engine against the original build script."""

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from sweep import (
    EASTERN,
    ENTRY_START,
    ENTRY_END,
    RTH_END,
    MIN_POC_DISTANCE_TICKS,
    LEG_T0,
    LEG_P0,
    LEG_R0,
    STAT0,
    NET_COL,
    timestamp_100ns,
    eastern_day_id,
    price_text,
    rth_profile_for_session,
)


def load_original(path):
    """Load the original build script under a distinct module name so its
    simulate() can coexist with the patched one."""
    spec = importlib.util.spec_from_file_location("original_engine", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["original_engine"] = mod
    spec.loader.exec_module(mod)
    return mod


def session_args(day):
    date = datetime.fromisoformat(day).date()
    es = timestamp_100ns(datetime.combine(date, ENTRY_START, tzinfo=EASTERN))
    ee = timestamp_100ns(datetime.combine(date, ENTRY_END, tzinfo=EASTERN))
    fl = timestamp_100ns(datetime.combine(date, RTH_END, tzinfo=EASTERN))
    return es, ee, fl


def run_patched(selected, sessions_dir):
    """Run the patched simulate() with A_baseline parameters."""
    from sweep import simulate
    out = []
    for s in selected:
        day = s["date_eastern"]
        data = np.load(sessions_dir / s["file"])
        prices = np.ascontiguousarray(data["last"], dtype=np.int32)
        stamps = np.ascontiguousarray(data["ts_100ns"], dtype=np.int64)
        sizes = np.ascontiguousarray(data["volume"], dtype=np.int64)
        day_ids = np.fromiter(
            (eastern_day_id(x) for x in stamps),
            dtype=np.int64, count=len(stamps))

        es, ee, fl = session_args(day)
        poc, vah, val = rth_profile_for_session(prices, sizes, stamps)

        atr = np.zeros(len(prices), dtype=np.int32)
        trades, _summary = simulate(
            stamps, prices, poc, vah, val, day_ids,
            es, ee, fl, 0,
            120, 2, 80, False, True, 320, MIN_POC_DISTANCE_TICKS,
            False, 0,
            atr, 0.0, 60, 200, 0, 0, 1,
            np.ones(len(prices), dtype=np.int8),
        )
        for t in trades:
            out.append({
                "day": day,
                "entry_time": int(t[0]),
                "entry": int(t[1]),
                "pos": int(t[2]),
                "gross": int(t[STAT0 + 3]),
                "comm": int(t[STAT0 + 4]),
                "net": int(t[NET_COL]),
                "raw": [int(x) for x in t],
            })
    return out


def run_original(mod, selected, sessions_dir):
    """Run the original engine's simulate() with equivalent parameters."""
    out = []
    for s in selected:
        day = s["date_eastern"]
        data = np.load(sessions_dir / s["file"])
        prices = np.ascontiguousarray(data["last"], dtype=np.int32)
        stamps = np.ascontiguousarray(data["ts_100ns"], dtype=np.int64)
        sizes = np.ascontiguousarray(data["volume"], dtype=np.int64)
        day_ids = np.fromiter(
            (eastern_day_id(x) for x in stamps),
            dtype=np.int64, count=len(stamps))

        es, ee, _fl = session_args(day)
        poc, vah, val = mod.rth_profile_for_session(prices, sizes, stamps)

        # original row layout:
        # 0 entry_time  1 tp1_time  2 runner_time  3 pos  4 entry
        # 12 gross  13 commission  14 net
        trades, _summary = mod.simulate(
            stamps, prices, poc, vah, val, day_ids, es, ee, 0,
        )
        for t in trades:
            out.append({
                "day": day,
                "entry_time": int(t[0]),
                "entry": int(t[4]),
                "pos": int(t[3]),
                "gross": int(t[12]),
                "comm": int(t[13]),
                "net": int(t[14]),
                "raw": [int(x) for x in t],
            })
    return out


def match(patch_list, orig_list):
    """Pair trades by (day, direction, entry price), nearest timestamp."""
    def key(r):
        return (r["day"], r["pos"], r["entry"])

    o_by = {}
    for r in orig_list:
        o_by.setdefault(key(r), []).append(r)

    pairs, unmatched_p = [], []
    for p in patch_list:
        bucket = o_by.get(key(p), [])
        if not bucket:
            unmatched_p.append(p)
            continue
        best = min(bucket, key=lambda o: abs(o["entry_time"] - p["entry_time"]))
        bucket.remove(best)
        pairs.append((p, best))

    unmatched_o = [r for v in o_by.values() for r in v]
    return pairs, unmatched_p, unmatched_o


def audit_arithmetic(rows, label):
    """Check gross - commission == net on each row."""
    print(f"\nper-trade arithmetic audit ({label}):")
    bad = 0
    for r in rows:
        if r["gross"] - r["comm"] != r["net"]:
            bad += 1
            if bad <= 5:
                print(f"  MISMATCH {r['day']} entry "
                      f"{price_text(r['entry'])}: "
                      f"gross {r['gross']} - comm {r['comm']} = "
                      f"{r['gross'] - r['comm']} but net {r['net']}")
    if bad == 0:
        print("  all rows reconcile")
    else:
        print(f"  {bad} of {len(rows)} rows fail gross - commission == net")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True,
                    help="path to the original backtest build script (.py)")
    ap.add_argument("--sessions", type=Path, default=Path("output/sessions"))
    ap.add_argument("--limit", type=int, default=5,
                    help="sessions to compare (default 5)")
    args = ap.parse_args()

    sessions_dir = args.sessions
    manifest = json.loads((sessions_dir / "manifest.json").read_text())
    sel = [r for r in manifest["sessions"] if r["include_in_primary"]]
    sel = sel[: args.limit]

    print(f"Comparing engines over {len(sel)} session(s)\n")
    mod = load_original(args.original)
    print("original engine loaded\n")

    pl = run_patched(sel, sessions_dir)
    ol = run_original(mod, sel, sessions_dir)

    print(f"patched:  {len(pl)} trades, "
          f"net ${sum(r['net'] for r in pl) / 100:,.2f}")
    print(f"original: {len(ol)} trades, "
          f"net ${sum(r['net'] for r in ol) / 100:,.2f}\n")

    pairs, up, uo = match(pl, ol)
    print(f"matched pairs: {len(pairs)}")
    print(f"patched-only:  {len(up)}")
    for r in up:
        print(f"   {r['day']} {price_text(r['entry'])} "
              f"net ${r['net'] / 100:,.2f}")
    print(f"original-only: {len(uo)}")
    for r in uo:
        print(f"   {r['day']} {price_text(r['entry'])} "
              f"net ${r['net'] / 100:,.2f}")

    if pairs:
        print("\nper-trade net differences (first 25):")
        print(f"{'date':<12}{'entry':>10}{'patched':>12}"
              f"{'original':>12}{'diff':>12}")
        print("-" * 58)
        for p, o in pairs[:25]:
            d = p["net"] - o["net"]
            print(f"{p['day']:<12}{price_text(p['entry']):>10}"
                  f"{p['net'] / 100:>12,.2f}{o['net'] / 100:>12,.2f}"
                  f"{d / 100:>12,.2f}")

        deltas = [p["net"] - o["net"] for p, o in pairs]
        nonzero = sum(1 for d in deltas if d != 0)
        print(f"\nall {len(pairs)} pairs: "
              f"total diff ${sum(deltas) / 100:,.2f}, "
              f"mean ${sum(deltas) / len(deltas) / 100:,.2f}, "
              f"nonzero {nonzero}")

    audit_arithmetic(pl, "patched")
    audit_arithmetic(ol, "original")


if __name__ == "__main__":
    main()