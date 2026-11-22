import argparse
import json
import struct
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


UTC = ZoneInfo("UTC")
EASTERN = ZoneInfo("America/New_York")

# Fixed-width binary record, 28 bytes:
# Eastern time in 100-nanosecond units since midnight,
# last/bid/ask in quarter-point ticks, trade volume.
RECORD = struct.Struct("<QiiiQ")
UNITS_PER_SECOND = 10_000_000
BUFFER_LIMIT = RECORD.size * 100_000

REVIEW_DATES = {
    "2026-06-11": "Thin session identified in initial audit",
    "2026-06-12": "Thin session identified in initial audit",
    "2026-06-19": "Shortened session; calendar review required",
    "2026-07-03": "Shortened session; calendar review required",
    "2026-09-07": "Shortened session; calendar review required",
}


def price_ticks(text):
    # Parse exactly, without floating-point rounding.
    whole, separator, fraction = text.partition(".")
    fraction = fraction.rstrip("0") if separator else ""
    quarters = {"": 0, "25": 1, "5": 2, "75": 3}
    if fraction not in quarters:
        raise ValueError(f"Price not on quarter-point grid: {text}")
    return int(whole) * 4 + quarters[fraction]


def session_bounds(date_text):
    day = date(
        int(date_text[:4]),
        int(date_text[4:6]),
        int(date_text[6:8]),
    )

    opening = datetime(
        day.year, day.month, day.day, 9, 30, tzinfo=EASTERN
    ).astimezone(UTC)
    closing = datetime(
        day.year, day.month, day.day, 16, 0, tzinfo=EASTERN
    ).astimezone(UTC)

    # New York's daytime session falls on the same UTC calendar date.
    start = opening.hour * 3600 + opening.minute * 60
    end = closing.hour * 3600 + closing.minute * 60
    return day.isoformat(), start, end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("output/sessions")
    )
    args = parser.parse_args()

    if not args.file.is_file():
        raise SystemExit(f"Input file not found: {args.file}")

    destination = args.out.resolve()
    staging = destination.with_name(destination.name + "_building")

    if destination.exists() or staging.exists():
        raise SystemExit(
            "Output or unfinished build folder already exists.\n"
            f"Output: {destination}\n"
            f"Build:  {staging}\n"
            "Use a different --out path, or inspect the existing folder."
        )

    staging.mkdir(parents=True)
    started = time.perf_counter()
    source_size = args.file.stat().st_size
    sessions = []
    output = None
    current = None
    buffer = bytearray()
    cached_date = None
    previous_key = None
    lines = 0
    kept = 0
    bytes_read = 0

    def finish_session():
        nonlocal output, current
        if output is None:
            return
        output.write(buffer)
        buffer.clear()
        output.close()
        output = None
        current["file_bytes"] = (
            staging / current["file"]
        ).stat().st_size
        sessions.append(current)
        print(
            f"Saved {current['date_eastern']}: "
            f"{current['records']:,} records",
            flush=True,
        )
        current = None

    try:
        with args.file.open(
            "rb", buffering=8 * 1024 * 1024
        ) as source:
            for lines, raw in enumerate(source, 1):
                bytes_read += len(raw)
                text = raw.decode("utf-8-sig").strip()
                if not text:
                    raise ValueError(f"Blank record at line {lines}")

                stamp, last, bid, ask, volume = text.split(";")
                day_text, clock, fraction = stamp.split()

                if (
                    len(day_text) != 8
                    or len(clock) != 6
                    or len(fraction) != 7
                    or not (day_text + clock + fraction).isdigit()
                ):
                    raise ValueError(
                        f"Invalid timestamp at line {lines}"
                    )

                key = day_text + clock + fraction
                if previous_key is not None and key < previous_key:
                    raise ValueError(
                        f"Out-of-order timestamp at line {lines}"
                    )
                previous_key = key

                if day_text != cached_date:
                    day, start, end = session_bounds(day_text)
                    cached_date = day_text

                seconds_utc = (
                    int(clock[:2]) * 3600
                    + int(clock[2:4]) * 60
                    + int(clock[4:6])
                )

                if start <= seconds_utc < end:
                    if current is None or current["date_eastern"] != day:
                        finish_session()
                        filename = f"{day}.bin"
                        output = (staging / filename).open("wb")
                        current = {
                            "date_eastern": day,
                            "file": filename,
                            "records": 0,
                            "volume": 0,
                            "first_time_100ns": None,
                            "last_time_100ns": None,
                            "review_note": REVIEW_DATES.get(day, ""),
                        }

                    seconds_eastern = (
                        seconds_utc - start + 9 * 3600 + 30 * 60
                    )
                    event_time = (
                        seconds_eastern * UNITS_PER_SECOND
                        + int(fraction)
                    )
                    size = int(volume)
                    if size <= 0:
                        raise ValueError(
                            f"Invalid volume at line {lines}"
                        )

                    buffer.extend(
                        RECORD.pack(
                            event_time,
                            price_ticks(last),
                            price_ticks(bid),
                            price_ticks(ask),
                            size,
                        )
                    )
                    if len(buffer) >= BUFFER_LIMIT:
                        output.write(buffer)
                        buffer.clear()

                    current["records"] += 1
                    current["volume"] += size
                    if current["first_time_100ns"] is None:
                        current["first_time_100ns"] = event_time
                    current["last_time_100ns"] = event_time
                    kept += 1

                if lines % 1_000_000 == 0:
                    percent = bytes_read / source_size * 100
                    elapsed = time.perf_counter() - started
                    print(
                        f"{percent:5.1f}% | {lines:,} lines | "
                        f"{kept:,} RTH records | {elapsed:,.0f}s",
                        flush=True,
                    )

        finish_session()

        manifest = {
            "format_version": 1,
            "source_file": str(args.file.resolve()),
            "source_size_bytes": source_size,
            "source_timestamp_assumption": "UTC",
            "instrument": "MNQ; expiry unknown",
            "timezone": "America/New_York",
            "session_filter": "09:30 <= time < 16:00 Eastern",
            "calendar_note": (
                "No holiday filtering applied. Review notes do not "
                "exclude records."
            ),
            "duplicate_policy": "All records preserved",
            "binary_struct": "<QiiiQ",
            "record_bytes": RECORD.size,
            "fields": [
                "eastern_time_100ns_since_midnight",
                "last_ticks",
                "bid_ticks",
                "ask_ticks",
                "volume",
            ],
            "price_tick_points": 0.25,
            "source_lines": lines,
            "rth_records": kept,
            "session_count": len(sessions),
            "elapsed_seconds": round(
                time.perf_counter() - started, 1
            ),
            "sessions": sessions,
        }

        with (staging / "manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, indent=2)

        staging.rename(destination)
        print("\nCONVERSION COMPLETE")
        print(f"Source lines: {lines:,}")
        print(f"RTH records:  {kept:,}")
        print(f"Sessions:     {len(sessions)}")
        print(f"Saved to:     {destination}")

    except Exception:
        if output is not None:
            output.close()
        print(
            f"\nConversion failed near line {lines:,}.\n"
            f"Incomplete files remain in: {staging}\n"
            "Do not use them for backtesting."
        )
        raise


if __name__ == "__main__":
    main()