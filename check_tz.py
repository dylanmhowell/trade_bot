"""Locate the RTH open/close volume spikes to verify timezone handling."""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

EASTERN = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

ZONES = {"ct": CT, "et": EASTERN, "utc": UTC}


def parse_line(line):
    """yyyyMMdd HHmmss ffffff;bid;last;ask;volume"""
    head, _, _ = line.partition(";")
    parts = head.split()
    if len(parts) != 3:
        return None
    try:
        base = datetime.strptime(parts[0] + " " + parts[1], "%Y%m%d %H%M%S")
    except ValueError:
        return None
    frac = parts[2]
    if len(frac) == 7:          # '0240000' -> 0.240000 s
        micro = int(frac) // 10
    elif len(frac) == 6:
        micro = int(frac)
    else:
        micro = 0
    return base.replace(microsecond=micro)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--tz", choices=list(ZONES), default="ct",
                    help="timezone the file's timestamps are written in")
    ap.add_argument("--last-field", type=int, default=3,
                    help="1-based semicolon field holding last-trade price")
    ap.add_argument("--vol-field", type=int, default=5,
                    help="1-based semicolon field holding size")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tz = ZONES[args.tz]
    minute_vol = Counter()
    minute_cnt = Counter()
    dates = set()
    n = 0

    with args.src.open() as f:
        for line in f:
            line = line.strip()
            if not line or line[0].isalpha():
                continue
            parts = line.split(";")
            if len(parts) < max(args.last_field, args.vol_field):
                continue
            dt = parse_line(line)
            if dt is None:
                continue
            try:
                size = int(float(parts[args.vol_field - 1]))
            except ValueError:
                continue
            if size <= 0:
                continue

            local = dt.replace(tzinfo=tz).astimezone(EASTERN)
            minute_vol[local.hour * 60 + local.minute] += size
            minute_cnt[local.hour * 60 + local.minute] += 1
            dates.add(local.date())
            n += 1
            if args.limit and n >= args.limit:
                break

    if not n:
        raise SystemExit("no usable rows")

    print(f"file      {args.src.name}")
    print(f"assumed   source timestamps are {args.tz.upper()}")
    print(f"rows      {n:,}   sessions {len(dates)}   "
          f"{min(dates)} .. {max(dates)}")
    print()

    # top minutes by volume
    ranked = sorted(minute_vol.items(), key=lambda kv: -kv[1])[:15]
    print("top 15 minutes of day by volume (Eastern):")
    print(f"  {'time':>6} {'minute':>7} {'volume':>12} {'ticks':>10}")
    for m, v in sorted(ranked):
        print(f"  {m // 60:02d}:{m % 60:02d} {m:>7} {v:>12,} {minute_cnt[m]:>10,}")
    print()

    # find the biggest volume jump in the morning region and the drop in the
    # afternoon region, then report their spacing
    def biggest_jump(lo, hi):
        best, best_m = -1, None
        for m in range(lo, hi):
            prev = minute_vol.get(m - 1, 0)
            cur = minute_vol.get(m, 0)
            jump = cur - prev
            if jump > best:
                best, best_m = jump, m
        return best_m, best

    def biggest_drop(lo, hi):
        best, best_m = 10**18, None
        for m in range(lo, hi):
            prev = minute_vol.get(m - 1, 0)
            cur = minute_vol.get(m, 0)
            drop = prev - cur
            if drop < best:
                best, best_m = drop, m
        return best_m, best

    open_m, open_jump = biggest_jump(480, 660)      # 08:00-11:00 ET
    close_m, close_drop = biggest_drop(900, 1020)   # 15:00-17:00 ET

    def label(m):
        return f"{m // 60:02d}:{m % 60:02d}" if m is not None else "n/a"

    print("detected structure:")
    print(f"  largest morning volume jump   {label(open_m)} "
          f"({open_jump:+,} contracts/min)")
    print(f"  largest afternoon volume drop {label(close_m)} "
          f"({close_drop:+,} contracts/min)")
    if open_m is not None and close_m is not None:
        span = close_m - open_m
        print(f"  spacing {span} minutes (RTH is 390)")
        if span == 390:
            print("  -> spacing matches a 6.5-hour RTH session")
        else:
            print(f"  -> spacing is off by {span - 390:+d} minutes")
    print()

    expected_open, expected_close = 570, 960
    if open_m is not None:
        off = open_m - expected_open
        if off == 0:
            print("VERDICT: open spike lands at 09:30 ET. Timezone is correct.")
        else:
            print(f"VERDICT: open spike is at {label(open_m)}, "
                  f"expected 09:30. Shift of {off:+d} minutes "
                  f"({off / 60:+.2f} h).")
            print(f"         If your file is really in another zone, retry with "
                  f"--tz {'et' if args.tz == 'ct' else 'ct'}.")


if __name__ == "__main__":
    main()