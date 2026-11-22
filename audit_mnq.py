import argparse
import csv
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
RTH_START = 9 * 3600 + 30 * 60
RTH_END = 16 * 3600
MAX_EXAMPLES = 200


def parse_record(line):
    timestamp, last, bid, ask, volume = line.strip().split(";")
    date, clock, fraction = timestamp.split()

    if (
        len(date) != 8
        or len(clock) != 6
        or len(fraction) != 7
        or not (date + clock + fraction).isdigit()
    ):
        raise ValueError("Unexpected timestamp format")

    # datetime stores microseconds. Keep the seventh fractional digit in
    # the separate ordering key so 100-nanosecond ordering is preserved.
    ordering_key = date + clock + fraction
    instant = datetime(
        int(date[:4]),
        int(date[4:6]),
        int(date[6:8]),
        int(clock[:2]),
        int(clock[2:4]),
        int(clock[4:6]),
        int(fraction[:6]),
        tzinfo=UTC,
    )

    prices = tuple(float(value) for value in (last, bid, ask))
    if not all(math.isfinite(value) and value > 0 for value in prices):
        raise ValueError("Nonfinite or nonpositive price")

    size = int(volume)
    if size <= 0:
        raise ValueError("Nonpositive trade volume")

    return ordering_key, instant, *prices, size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("output/audit"))
    args = parser.parse_args()

    if not args.file.is_file():
        raise SystemExit(f"File not found: {args.file}")

    args.out.mkdir(parents=True, exist_ok=True)
    daily = {}
    issues = Counter()
    examples = []
    rows = 0
    valid = 0
    first = None
    last = None
    previous_key = None
    previous_line = None
    previous_rth = None
    started = time.perf_counter()

    def flag(kind, number, text):
        issues[kind] += 1
        if len(examples) < MAX_EXAMPLES:
            examples.append(
                {
                    "line": number,
                    "issue": kind,
                    "record": text.strip()[:300],
                }
            )

    with args.file.open(
        "r", encoding="utf-8-sig", buffering=8 * 1024 * 1024
    ) as handle:
        for rows, line in enumerate(handle, 1):
            if rows % 1_000_000 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"{rows:,} lines scanned | "
                    f"{valid:,} valid | {elapsed:,.0f}s",
                    flush=True,
                )

            if not line.strip():
                flag("blank_line", rows, line)
                continue

            try:
                key, instant, price, bid, ask, volume = parse_record(line)
            except (ValueError, OverflowError) as exc:
                flag(f"invalid_record: {exc}", rows, line)
                continue

            valid += 1
            first = instant if first is None else min(first, instant)
            last = instant if last is None else max(last, instant)

            if previous_key is not None and key < previous_key:
                flag("out_of_order_timestamp", rows, line)
            if line == previous_line:
                # Informational only: these may be legitimate trades.
                issues["identical_consecutive_valid_rows"] += 1

            previous_key = key
            previous_line = line

            if any(
                abs(value * 4 - round(value * 4)) > 0.000001
                for value in (price, bid, ask)
            ):
                flag("price_not_on_quarter_point_grid", rows, line)

            eastern = instant.astimezone(EASTERN)
            day = eastern.date().isoformat()
            seconds = (
                eastern.hour * 3600
                + eastern.minute * 60
                + eastern.second
            )

            if day not in daily:
                daily[day] = {
                    "date_eastern": day,
                    "trade_records": 0,
                    "volume": 0,
                    "rth_trade_records": 0,
                    "rth_volume": 0,
                    "rth_first_trade": "",
                    "rth_last_trade": "",
                    "rth_low": None,
                    "rth_high": None,
                    "rth_crossed_quote_records": 0,
                    "rth_spread_over_2_points": 0,
                    "rth_max_observed_gap_seconds": 0.0,
                    "_spread_sum": 0.0,
                    "_spread_count": 0,
                    "_max_spread": None,
                }

            stats = daily[day]
            stats["trade_records"] += 1
            stats["volume"] += volume

            if not RTH_START <= seconds < RTH_END:
                continue

            stats["rth_trade_records"] += 1
            stats["rth_volume"] += volume
            stamp = eastern.isoformat()
            if (
                not stats["rth_first_trade"]
                or stamp < stats["rth_first_trade"]
            ):
                stats["rth_first_trade"] = stamp
            if stamp > stats["rth_last_trade"]:
                stats["rth_last_trade"] = stamp

            low = stats["rth_low"]
            high = stats["rth_high"]
            stats["rth_low"] = price if low is None else min(low, price)
            stats["rth_high"] = price if high is None else max(high, price)

            spread = ask - bid
            if spread < 0:
                stats["rth_crossed_quote_records"] += 1
                flag("rth_crossed_quote", rows, line)
            else:
                stats["_spread_sum"] += spread
                stats["_spread_count"] += 1
                maximum = stats["_max_spread"]
                stats["_max_spread"] = (
                    spread if maximum is None else max(maximum, spread)
                )
                if spread > 2:
                    stats["rth_spread_over_2_points"] += 1

            if previous_rth is not None:
                prior_day, prior_time, prior_price = previous_rth
                if prior_day == day:
                    gap = (instant - prior_time).total_seconds()
                    stats["rth_max_observed_gap_seconds"] = max(
                        stats["rth_max_observed_gap_seconds"], gap
                    )
                    if gap > 60:
                        flag("rth_gap_over_60_seconds", rows, line)
                    if gap >= 0 and abs(price - prior_price) >= 20:
                        flag("rth_trade_jump_20_points_or_more", rows, line)

            previous_rth = (day, instant, price)

    output_rows = []
    for day in sorted(daily):
        stats = daily[day]
        count = stats.pop("_spread_count")
        total = stats.pop("_spread_sum")
        maximum = stats.pop("_max_spread")
        stats["rth_mean_spread_points"] = (
            round(total / count, 4) if count else None
        )
        stats["rth_max_spread_points"] = maximum
        output_rows.append(stats)

    if output_rows:
        with (args.out / "daily.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(output_rows[0])
            )
            writer.writeheader()
            writer.writerows(output_rows)

    with (args.out / "flagged_examples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["line", "issue", "record"]
        )
        writer.writeheader()
        writer.writerows(examples)

    summary = {
        "source_file": str(args.file.resolve()),
        "source_size_bytes": args.file.stat().st_size,
        "timestamp_assumption": "UTC; not established by file contents",
        "instrument_assumption": "MNQ; expiry unknown",
        "rth_definition": "09:30 <= time < 16:00 America/New_York",
        "calendar_note": "No holiday or early-close calendar applied yet",
        "lines_scanned": rows,
        "valid_trade_records": valid,
        "first_timestamp_utc": first.isoformat() if first else None,
        "last_timestamp_utc": last.isoformat() if last else None,
        "eastern_dates_with_records": len(daily),
        "issue_counts": dict(issues),
        "saved_flagged_examples": len(examples),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }

    with (args.out / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print("\nAUDIT COMPLETE")
    print(json.dumps(summary, indent=2))
    print(f"\nReports saved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()