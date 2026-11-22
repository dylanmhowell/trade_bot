"""Convert a NinjaTrader tick export to one .npz per session.

Format: yyyyMMdd HHmmss ffffff;bid;last;ask;volume
Timestamps are Chicago time. Output is split by Eastern trading date.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

EASTERN = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
ZONES = {"ct": CT, "et": EASTERN, "utc": UTC}

BASE = 80000
SIZE = 80000
TICK_SIZE = 0.25


def to_100ns(dt):
    e = dt.astimezone(UTC) - EPOCH
    return (e.days * 86_400 * 10_000_000
            + e.seconds * 10_000_000
            + e.microseconds * 10)


def parse_line(line, tz, last_field, vol_field):
    parts = line.split(";")
    if len(parts) < max(last_field, vol_field):
        return None
    head = parts[0].split()
    if len(head) != 3:
        return None
    try:
        dt = datetime.strptime(head[0] + " " + head[1], "%Y%m%d %H%M%S")
    except ValueError:
        return None
    frac = head[2]
    micro = int(frac) // 10 if len(frac) == 7 else (
        int(frac) if len(frac) == 6 else 0)
    try:
        px = float(parts[last_field - 1])
        sz = int(float(parts[vol_field - 1]))
    except (ValueError, IndexError):
        return None
    if sz <= 0 or px <= 0:
        return None
    return dt.replace(microsecond=micro, tzinfo=tz), px, sz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, default=Path("output/sessions"))
    ap.add_argument("--tz", choices=list(ZONES), default="ct")
    ap.add_argument("--last-field", type=int, default=2)
    ap.add_argument("--vol-field", type=int, default=4)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tz = ZONES[args.tz]

    by_day = defaultdict(lambda: ([], [], []))
    skipped = 0

    with args.src.open() as f:
        for line in f:
            line = line.strip()
            if not line or line[0].isalpha():
                continue
            rec = parse_line(line, tz, args.last_field, args.vol_field)
            if rec is None:
                skipped += 1
                continue
            dt, px, sz = rec
            et = dt.astimezone(EASTERN)
            key = et.date().isoformat()
            stamps, prices, sizes = by_day[key]
            stamps.append(to_100ns(dt))
            prices.append(int(round(px / TICK_SIZE)))
            sizes.append(sz)

    if not by_day:
        raise SystemExit(f"{args.src}: no usable rows (skipped {skipped})")

    manifest = {"sessions": []}
    for day in sorted(by_day):
        stamps, prices, sizes = by_day[day]
        last = np.asarray(prices, dtype=np.int32)
        lo, hi = int(last.min()), int(last.max())
        if lo < BASE or hi >= BASE + SIZE:
            print(f"{day}: SKIP - ticks {lo}..{hi} outside "
                  f"{BASE}..{BASE + SIZE}")
            continue
        dst = args.out / f"{day}.npz"
        np.savez_compressed(
            dst, last=last,
            volume=np.asarray(sizes, dtype=np.int64),
            ts_100ns=np.asarray(stamps, dtype=np.int64))
        manifest["sessions"].append({
            "date_eastern": day,
            "file": dst.name,
            "include_in_primary": True,
        })
        first = datetime.fromtimestamp(stamps[0] / 10_000_000, UTC).astimezone(EASTERN)
        end = datetime.fromtimestamp(stamps[-1] / 10_000_000, UTC).astimezone(EASTERN)
        print(f"{day}.npz  {len(stamps):>9,} ticks  "
              f"{first:%H:%M}..{end:%H:%M} ET  px {lo/4:.2f}..{hi/4:.2f}")

    (args.out / "manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2))
    print(f"\n{len(manifest['sessions'])} sessions -> "
          f"{args.out / 'manifest.json'}  (skipped {skipped} rows)")


if __name__ == "__main__":
    main()