import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from zoneinfo import ZoneInfo


EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
RTH_START = 9 * 3600 + 30 * 60
RTH_END = 16 * 3600
DTYPE = np.dtype(
    [
        ("ts_100ns", "<i8"),
        ("last", "<i4"),
        ("bid", "<i4"),
        ("ask", "<i4"),
        ("volume", "<i4"),
    ]
)

# Recorded daytime trading ends near 1 p.m. Eastern on these dates.
HOLIDAY_SHORT = {
    "2026-06-19": "Juneteenth",
    "2026-07-03": "Independence Day observed",
    "2026-09-07": "Labor Day",
}
THIN_EARLY = {"2026-06-11", "2026-06-12"}
PRIMARY_START = "2026-06-15"


def ticks(value):
    scaled = value * 4
    rounded = round(scaled)
    if abs(scaled - rounded) > 1e-6:
        raise ValueError(f"Price is not on a 0.25 grid: {value}")
    return int(rounded)


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
    elapsed = instant - EPOCH
    ts_100ns = (
        elapsed.days * 86_400 * 10_000_000
        + elapsed.seconds * 10_000_000
        + elapsed.microseconds * 10
        + int(fraction[6])
    )
    size = int(volume)
    if size <= 0:
        raise ValueError("Nonpositive volume")
    return (
        ts_100ns,
        ticks(float(last)),
        ticks(float(bid)),
        ticks(float(ask)),
        size,
        instant,
    )


def session_class(day):
    if day in HOLIDAY_SHORT:
        return "holiday_short", False, HOLIDAY_SHORT[day]
    if day in THIN_EARLY:
        return "thin_early", False, "Low-liquidity sessions before June 15"
    if day >= PRIMARY_START:
        return "primary", True, ""
    return "other", False, "Before primary start date"


def flush(day, rows, out_dir, manifest):
    if not rows:
        return
    array = np.asarray(rows, dtype=DTYPE)
    if np.any(array["ts_100ns"][1:] < array["ts_100ns"][:-1]):
        raise RuntimeError(f"{day}: timestamps are not ordered")

    path = out_dir / f"{day}.npy"
    np.save(path, array)
    loaded = np.load(path)
    if len(loaded) != len(array):
        raise RuntimeError(f"{day}: saved file failed verification")

    first = datetime.fromtimestamp(
        int(array["ts_100ns"][0]) / 10_000_000, UTC
    )
    last = datetime.fromtimestamp(
        int(array["ts_100ns"][-1]) / 10_000_000, UTC
    )
    kind, primary, note = session_class(day)
    last_eastern = last.astimezone(EASTERN)
    if last_eastern.hour < 15 and kind == "primary":
        kind = "unexpected_short"
        primary = False
        note = "Recorded regular-hours trading ends before 3 p.m. Eastern"

    manifest.append(
        {
            "date_eastern": day,
            "file": path.name,
            "records": int(len(array)),
            "volume": int(array["volume"].sum()),
            "session_class": kind,
            "include_in_primary": primary,
            "first_utc": first.isoformat(),
            "last_utc": last.isoformat(),
            "last_eastern": last_eastern.isoformat(),
            "low": float(array["last"].min()) / 4,
            "high": float(array["last"].max()) / 4,
            "note": note,
        }
    )
    print(
        f"Saved {day}: {len(array):,} records | {kind}",
        flush=True,
    )


def compare_with_audit(manifest, audit_path):
    if not audit_path.is_file():
        print(f"Audit file not found, skipped comparison: {audit_path}")
        return []

    with audit_path.open(newline="", encoding="utf-8") as handle:
        audit = {
            row["date_eastern"]: row
            for row in csv.DictReader(handle)
            if int(row["rth_trade_records"])
        }

    converted = {row["date_eastern"]: row for row in manifest}
    mismatches = []
    for day in sorted(set(audit) | set(converted)):
        expected = audit.get(day)
        actual = converted.get(day)
        if expected is None or actual is None:
            mismatches.append(f"{day}: present on only one side")
            continue
        if int(expected["rth_trade_records"]) != actual["records"]:
            mismatches.append(f"{day}: record count differs")
        if int(expected["rth_volume"]) != actual["volume"]:
            mismatches.append(f"{day}: volume differs")
    return mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("output/sessions"))
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("output/audit/daily.csv"),
    )
    args = parser.parse_args()
    if not args.file.is_file():
        raise SystemExit(f"File not found: {args.file}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    current_day = None
    rows = []
    scanned = 0

    with args.file.open(
        "r", encoding="utf-8-sig", buffering=8 * 1024 * 1024
    ) as handle:
        for scanned, line in enumerate(handle, 1):
            if scanned % 5_000_000 == 0:
                print(f"{scanned:,} lines scanned", flush=True)
            if not line.strip():
                continue

            ts_100ns, last, bid, ask, volume, instant = parse_record(line)
            eastern = instant.astimezone(EASTERN)
            seconds = (
                eastern.hour * 3600
                + eastern.minute * 60
                + eastern.second
            )
            if not RTH_START <= seconds < RTH_END:
                continue

            day = eastern.date().isoformat()
            if current_day is None:
                current_day = day
            elif day != current_day:
                flush(current_day, rows, args.out, manifest)
                current_day = day
                rows = []
            rows.append((ts_100ns, last, bid, ask, volume))

    flush(current_day, rows, args.out, manifest)
    mismatches = compare_with_audit(manifest, args.audit)
    primary = sum(row["include_in_primary"] for row in manifest)
    summary = {
        "source_file": str(args.file.resolve()),
        "timestamp_assumption": "UTC; not established by file contents",
        "instrument": "MNQ; expiry unknown",
        "rth": "09:30 <= time < 16:00 America/New_York",
        "price_encoding": "integer quarter-point ticks",
        "sessions_written": len(manifest),
        "primary_sessions": primary,
        "records_written": sum(row["records"] for row in manifest),
        "lines_scanned": scanned,
        "audit_mismatches": mismatches,
    }

    with (args.out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "sessions": manifest}, handle, indent=2)
    with (args.out / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    print("\nCONVERSION COMPLETE")
    print(json.dumps(summary, indent=2))
    if mismatches:
        raise SystemExit("Conversion does not match the audit.")


if __name__ == "__main__":
    main()