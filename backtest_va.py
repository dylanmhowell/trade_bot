import argparse
import csv
import json
import time
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

# Price is stored in quarter-point ticks:
# 1 tick = 0.25 index points.
TICK_CENTS = 50
COMMISSION_CENTS = 50

STOP_TICKS = 120          # 30 points
TARGET_TICKS = 80         # 20 points
STOP_SLIP_TICKS = 1

RTH_START = clock(9, 30)
RTH_END = clock(16, 0)
ENTRY_START = clock(10, 15)
ENTRY_END = clock(15, 0)

COOLDOWN_100NS = 15 * 60 * 10_000_000
MAX_DAILY_TRADES = 5
MIN_POC_DISTANCE_TICKS = 120  # 30 points


def timestamp_100ns(instant):
    elapsed = instant.astimezone(UTC) - EPOCH
    return (
        elapsed.days * 86_400 * 10_000_000
        + elapsed.seconds * 10_000_000
        + elapsed.microseconds * 10
    )


def from_100ns(value):
    seconds, fraction = divmod(int(value), 10_000_000)
    return datetime.fromtimestamp(seconds, UTC).replace(
        microsecond=fraction // 10
    )


def price_text(ticks):
    return f"{ticks / 4:.2f}"


def eastern_day_id(stamp):
    return from_100ns(stamp).astimezone(EASTERN).date().toordinal()


def rth_profile_for_session(prices, sizes, stamps):
    """Build a developing profile from 09:30–16:00 Eastern ticks.

    Returns full-length arrays. Before the RTH open, profile values are
    initialized from the first RTH profile state; those values are not used
    for entries because entries are restricted to 10:15–15:00.
    """
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

    # Carry forward each developing RTH profile state through the full
    # timestamp array. Values before the first RTH tick are placeholders.
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

    # For any records after an RTH tick, carry the most recent RTH profile.
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
    area_low = poc
    area_high = poc

    while included * 100 < total * 70:
        above = -1
        below = -1

        if area_high < high:
            above = volume[area_high + 1]
        if area_low > low:
            below = volume[area_low - 1]

        if above < 0 and below < 0:
            break

        if above >= below:
            area_high += 1
            included += above
        else:
            area_low -= 1
            included += below

    return area_low, area_high


@njit(cache=True)
def self_test():
    volume = np.zeros(11, np.int64)
    volume[0] = 20
    volume[1] = 40
    volume[2] = 90
    volume[3] = 150
    volume[4] = 180
    volume[5] = 260
    volume[6] = 200
    volume[7] = 120
    volume[8] = 80
    volume[9] = 40
    volume[10] = 20
    return build_value_area(volume, 0, 10, 5, 1200)


@njit(cache=True)
def build_profile(prices, sizes):
    count = len(prices)

    poc_out = np.empty(count, np.int32)
    vah_out = np.empty(count, np.int32)
    val_out = np.empty(count, np.int32)

    volume = np.zeros(SIZE, np.int64)
    total = 0
    low = SIZE
    high = -1
    poc = -1

    for index in range(count):
        bin_index = prices[index] - BASE
        size = sizes[index]

        volume[bin_index] += size
        total += size

        if bin_index < low:
            low = bin_index
        if bin_index > high:
            high = bin_index

        if poc < 0 or volume[bin_index] > volume[poc]:
            poc = bin_index
        elif volume[bin_index] == volume[poc] and bin_index > poc:
            poc = bin_index

        area_low, area_high = build_value_area(
            volume, low, high, poc, total
        )

        poc_out[index] = poc + BASE
        val_out[index] = area_low + BASE
        vah_out[index] = area_high + BASE

    return poc_out, vah_out, val_out


@njit(cache=True)
def simulate(
    timestamps,
    prices,
    poc,
    vah,
    val_area,
    day_ids,
    entry_start,
    entry_end,
    execution_model,
):
    count = len(prices)

    # Each row holds one completed strategy trade.
    trades = np.full((MAX_TRADES, 15), -1, np.int64)
    summary = np.zeros(8, np.int64)

    trade_count = 0
    cash = 0
    peak = 0
    max_drawdown = 0

    position = 0
    entry = 0
    entry_time = 0
    stop = 0
    target = 0
    runner_target = 0

    first_open = 0
    runner_open = 0
    first_done = 0

    gross = 0
    commission = 0

    first_exit_time = 0
    first_exit_price = 0
    first_reason = -1

    runner_exit_time = 0
    runner_exit_price = 0
    runner_reason = -1

    entry_poc = 0
    entry_vah = 0
    entry_val = 0

    working_short = -1
    working_long = -1
    short_armed = 0
    long_armed = 0
    origin = 0
    previous = -1

    current_day = -1
    daily_trade_count = 0
    cooldown_until = 0

    for index in range(count):
        price = prices[index]
        stamp = timestamps[index]
        day_id = day_ids[index]
        entered = 0

        if day_id != current_day:
            current_day = day_id
            daily_trade_count = 0
            cooldown_until = 0

        if position != 0:
            stop_hit = 0
            stop_fill = 0

            if position == 1 and price <= stop:
                stop_hit = 1
                stop_fill = min(stop, price) - STOP_SLIP_TICKS
            elif position == -1 and price >= stop:
                stop_hit = 1
                stop_fill = max(stop, price) + STOP_SLIP_TICKS

            if stop_hit == 1:
                reason = 2
                if first_done == 1:
                    reason = 3

                if first_open == 1:
                    first_open = 0
                    first_done = 1
                    first_exit_time = stamp
                    first_exit_price = stop_fill
                    first_reason = reason

                    change = (stop_fill - entry) * position
                    gross += change * TICK_CENTS
                    commission += COMMISSION_CENTS
                    cash += change * TICK_CENTS - COMMISSION_CENTS

                if runner_open == 1:
                    runner_open = 0
                    runner_exit_time = stamp
                    runner_exit_price = stop_fill
                    runner_reason = reason

                    change = (stop_fill - entry) * position
                    gross += change * TICK_CENTS
                    commission += COMMISSION_CENTS
                    cash += change * TICK_CENTS - COMMISSION_CENTS

            else:
                if first_open == 1:
                    hit = 0

                    if position == 1 and (
                        (execution_model == 0 and price >= target)
                        or (execution_model == 1 and price > target)
                    ):
                        hit = 1

                    if position == -1 and (
                        (execution_model == 0 and price <= target)
                        or (execution_model == 1 and price < target)
                    ):
                        hit = 1

                    if hit == 1:
                        first_open = 0
                        first_done = 1
                        first_exit_time = stamp
                        first_exit_price = target
                        first_reason = 0

                        change = (target - entry) * position
                        gross += change * TICK_CENTS
                        commission += COMMISSION_CENTS
                        cash += change * TICK_CENTS - COMMISSION_CENTS

                        # Deliberately do not move stop to breakeven.

                if runner_open == 1:
                    hit = 0

                    if position == 1 and (
                        (
                            execution_model == 0
                            and price >= runner_target
                        )
                        or (
                            execution_model == 1
                            and price > runner_target
                        )
                    ):
                        hit = 1

                    if position == -1 and (
                        (
                            execution_model == 0
                            and price <= runner_target
                        )
                        or (
                            execution_model == 1
                            and price < runner_target
                        )
                    ):
                        hit = 1

                    if hit == 1:
                        runner_open = 0
                        runner_exit_time = stamp
                        runner_exit_price = runner_target
                        runner_reason = 1

                        change = (runner_target - entry) * position
                        gross += change * TICK_CENTS
                        commission += COMMISSION_CENTS
                        cash += change * TICK_CENTS - COMMISSION_CENTS

            if first_open == 0 and runner_open == 0:
                if trade_count < MAX_TRADES:
                    row = trades[trade_count]
                    row[0] = entry_time
                    row[1] = first_exit_time
                    row[2] = runner_exit_time
                    row[3] = position
                    row[4] = entry
                    row[5] = first_exit_price
                    row[6] = runner_exit_price
                    row[7] = first_reason
                    row[8] = runner_reason
                    row[9] = entry_poc
                    row[10] = entry_vah
                    row[11] = entry_val
                    row[12] = gross
                    row[13] = commission
                    row[14] = gross - commission
                    trade_count += 1
                else:
                    summary[1] += 1

                position = 0

        if (
            position == 0
            and index > 0
            and stamp >= entry_start
            and stamp < entry_end
            and stamp >= cooldown_until
            and daily_trade_count < MAX_DAILY_TRADES
        ):
            short_signal = 0
            long_signal = 0

            if (
                working_short >= 0
                and previous < working_short
                and (
                    (execution_model == 0 and price >= working_short)
                    or (
                        execution_model == 1
                        and price > working_short
                    )
                )
            ):
                short_signal = 1

            if (
                working_long >= 0
                and previous > working_long
                and (
                    (execution_model == 0 and price <= working_long)
                    or (
                        execution_model == 1
                        and price < working_long
                    )
                )
            ):
                long_signal = 1

            if short_signal == 1 and long_signal == 1:
                summary[2] += 1

            elif short_signal == 1 or long_signal == 1:
                if trade_count >= MAX_TRADES:
                    summary[1] += 1
                else:
                    is_short = short_signal == 1
                    candidate_entry = (
                        working_short if is_short else working_long
                    )
                    candidate_poc = poc[index - 1]
                    candidate_vah = vah[index - 1]
                    candidate_val = val_area[index - 1]

                    # For shorts, compare POC to triggering VAH.
                    # For longs, compare POC to triggering VAL.
                    trigger_level = (
                        candidate_vah if is_short else candidate_val
                    )

                    if (
                        abs(candidate_poc - trigger_level)
                        >= MIN_POC_DISTANCE_TICKS
                    ):
                        entered = 1
                        position = -1 if is_short else 1
                        entry = candidate_entry
                        entry_time = stamp

                        entry_poc = candidate_poc
                        entry_vah = candidate_vah
                        entry_val = candidate_val

                        stop = entry - position * STOP_TICKS
                        target = entry + position * TARGET_TICKS

                        # Runner target is frozen at entry:
                        # short -> VAL; long -> VAH.
                        runner_target = (
                            entry_val if position == -1 else entry_vah
                        )

                        first_open = 1
                        runner_open = 1
                        first_done = 0

                        gross = 0
                        commission = 2 * COMMISSION_CENTS
                        cash -= commission

                        first_exit_time = 0
                        first_exit_price = 0
                        first_reason = -1
                        runner_exit_time = 0
                        runner_exit_price = 0
                        runner_reason = -1

                        daily_trade_count += 1
                        cooldown_until = stamp + COOLDOWN_100NS

                        short_armed = 0
                        long_armed = 0

        # Update the profile-crossing state only after processing a possible
        # signal, so newly armed orders apply to subsequent ticks.
        if index > 0 and entered == 0:
            prior_poc = poc[index - 1]
            crossed_up = origin == -1 and price > prior_poc
            crossed_down = origin == 1 and price < prior_poc

            if price > prior_poc:
                origin = 1
            elif price < prior_poc:
                origin = -1

            if crossed_up == 1:
                short_armed = 1
                long_armed = 0
            elif crossed_down == 1:
                long_armed = 1
                short_armed = 0

            if poc[index] != prior_poc:
                if price > poc[index]:
                    origin = 1
                elif price < poc[index]:
                    origin = -1
                else:
                    origin = 0

        elif index == 0:
            if price > poc[index]:
                origin = 1
            elif price < poc[index]:
                origin = -1

        if entered == 1:
            short_armed = 0
            long_armed = 0

        previous = price

        # The current tick's updated profile levels are used to stage orders
        # for subsequent ticks. Signal fills are checked against orders that
        # were already working before this tick.
        working_short = -1
        working_long = -1

        if (
            position == 0
            and short_armed == 1
            and price < vah[index]
        ):
            working_short = vah[index]

        if (
            position == 0
            and long_armed == 1
            and price > val_area[index]
        ):
            working_long = val_area[index]

        open_value = 0

        if first_open == 1:
            open_value += (price - entry) * position * TICK_CENTS
            open_value -= COMMISSION_CENTS

        if runner_open == 1:
            open_value += (price - entry) * position * TICK_CENTS
            open_value -= COMMISSION_CENTS

        equity = cash + open_value
        if equity > peak:
            peak = equity

        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Force-close any remaining position at the last observed price.
    if position != 0:
        final_price = prices[count - 1]
        final_time = timestamps[count - 1]

        if position == 1:
            fill = final_price - STOP_SLIP_TICKS
        else:
            fill = final_price + STOP_SLIP_TICKS

        if first_open == 1:
            first_exit_time = final_time
            first_exit_price = fill
            first_reason = 4

            change = (fill - entry) * position
            gross += change * TICK_CENTS
            commission += COMMISSION_CENTS
            cash += change * TICK_CENTS - COMMISSION_CENTS

        if runner_open == 1:
            runner_exit_time = final_time
            runner_exit_price = fill
            runner_reason = 4

            change = (fill - entry) * position
            gross += change * TICK_CENTS
            commission += COMMISSION_CENTS
            cash += change * TICK_CENTS - COMMISSION_CENTS

        if trade_count < MAX_TRADES:
            row = trades[trade_count]
            row[0] = entry_time
            row[1] = first_exit_time
            row[2] = runner_exit_time
            row[3] = position
            row[4] = entry
            row[5] = first_exit_price
            row[6] = runner_exit_price
            row[7] = first_reason
            row[8] = runner_reason
            row[9] = entry_poc
            row[10] = entry_vah
            row[11] = entry_val
            row[12] = gross
            row[13] = commission
            row[14] = gross - commission
            trade_count += 1
        else:
            summary[1] += 1

    summary[0] = trade_count
    summary[3] = cash
    summary[4] = max_drawdown

    return trades[:trade_count], summary


def load_sessions(path, limit):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    selected = [
        row
        for row in manifest["sessions"]
        if row["include_in_primary"]
    ]
    if limit is not None:
        selected = selected[:limit]
    return selected


def write_sample(day, timestamps, prices, poc, vah, val_area, output):
    targets = (
        clock(10, 15),
        clock(11, 0),
        clock(12, 0),
        clock(14, 0),
        clock(15, 0),
    )

    rows = []
    seen = set()

    for index, stamp in enumerate(timestamps):
        eastern = from_100ns(stamp).astimezone(EASTERN)
        current = eastern.time().replace(second=0, microsecond=0)

        if current not in targets or current in seen:
            continue

        seen.add(current)
        rows.append(
            {
                "date": day,
                "time_eastern": eastern.isoformat(),
                "price": price_text(int(prices[index])),
                "poc": price_text(int(poc[index])),
                "vah": price_text(int(vah[index])),
                "val": price_text(int(val_area[index])),
            }
        )

    if not rows:
        return

    destination = output / "profile_sample.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sessions",
        type=Path,
        default=Path("output/sessions"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/backtest"),
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    selected = load_sessions(args.sessions / "manifest.json", args.limit)
    if not selected:
        raise SystemExit("No primary sessions found.")

    args.out.mkdir(parents=True, exist_ok=True)

    print("Checking value-area calculation...", flush=True)
    low, high = self_test()
    if (low, high) != (3, 7):
        raise SystemExit(f"Value-area self-test failed: {(low, high)}")
    print("Value-area self-test passed.", flush=True)

    all_rows = []
    summaries = []
    started = time.perf_counter()

    for session in selected:
        day = session["date_eastern"]
        data = np.load(args.sessions / session["file"])

        prices = np.ascontiguousarray(data["last"], dtype=np.int32)
        if int(prices.min()) < BASE or int(prices.max()) >= BASE + SIZE:
            raise SystemExit(f"{day}: price outside expected range")

        stamps = np.ascontiguousarray(data["ts_100ns"], dtype=np.int64)
        sizes = np.ascontiguousarray(data["volume"], dtype=np.int64)
        day_ids = np.ascontiguousarray(
            np.fromiter(
                (eastern_day_id(stamp) for stamp in stamps),
                dtype=np.int64,
                count=len(stamps),
            )
        )

        date = datetime.fromisoformat(day).date()
        entry_start = timestamp_100ns(
            datetime.combine(date, ENTRY_START, tzinfo=EASTERN)
        )
        entry_end = timestamp_100ns(
            datetime.combine(date, ENTRY_END, tzinfo=EASTERN)
        )

        print(
            f"Building RTH profile for {day} "
            f"({len(data):,} records)...",
            flush=True,
        )

        build_started = time.perf_counter()
        try:
            poc, vah, val_area = rth_profile_for_session(
                prices, sizes, stamps
            )
        except ValueError as error:
            raise SystemExit(f"{day}: {error}") from error

        print(
            f"Profile built in "
            f"{time.perf_counter() - build_started:.1f}s",
            flush=True,
        )

        if session is selected[0]:
            write_sample(
                day,
                stamps,
                prices,
                poc,
                vah,
                val_area,
                args.out,
            )

        for model in (0, 1):
            fill_name = "touch" if model == 0 else "trade_through"
            name = f"{fill_name}_fixed_stop"

            trades, summary = simulate(
                stamps,
                prices,
                poc,
                vah,
                val_area,
                day_ids,
                entry_start,
                entry_end,
                model,
            )

            net = 0
            for trade in trades:
                net += int(trade[14])
                all_rows.append((day, name, trade))

            summaries.append(
                {
                    "date": day,
                    "variant": name,
                    "trades": int(summary[0]),
                    "overflow": int(summary[1]),
                    "conflicts": int(summary[2]),
                    "net_dollars": round(net / 100, 2),
                    "max_drawdown_dollars": round(
                        int(summary[4]) / 100, 2
                    ),
                }
            )

            print(
                f"  {name}: {int(summary[0])} trades, "
                f"net ${net / 100:,.2f}",
                flush=True,
            )

    reasons = {
        0: "tp1_20_points",
        1: "opposite_value_area",
        2: "stop_30_points",
        3: "stop_after_tp1",
        4: "session_end",
    }

    with (args.out / "trades.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "date",
            "variant",
            "direction",
            "entry_time_eastern",
            "entry_price",
            "tp1_time_eastern",
            "tp1_price",
            "tp1_reason",
            "runner_time_eastern",
            "runner_price",
            "runner_reason",
            "frozen_poc",
            "entry_vah",
            "entry_val",
            "gross_dollars",
            "commission_dollars",
            "net_dollars",
        ]

        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for day, name, trade in all_rows:
            direction = "long" if trade[3] == 1 else "short"

            writer.writerow(
                {
                    "date": day,
                    "variant": name,
                    "direction": direction,
                    "entry_time_eastern": from_100ns(
                        trade[0]
                    ).astimezone(EASTERN).isoformat(),
                    "entry_price": price_text(trade[4]),
                    "tp1_time_eastern": from_100ns(
                        trade[1]
                    ).astimezone(EASTERN).isoformat(),
                    "tp1_price": price_text(trade[5]),
                    "tp1_reason": reasons[int(trade[7])],
                    "runner_time_eastern": from_100ns(
                        trade[2]
                    ).astimezone(EASTERN).isoformat(),
                    "runner_price": price_text(trade[6]),
                    "runner_reason": reasons[int(trade[8])],
                    "frozen_poc": price_text(trade[9]),
                    "entry_vah": price_text(trade[10]),
                    "entry_val": price_text(trade[11]),
                    "gross_dollars": f"{trade[12] / 100:.2f}",
                    "commission_dollars": f"{trade[13] / 100:.2f}",
                    "net_dollars": f"{trade[14] / 100:.2f}",
                }
            )

    result = {
        "sessions": len(selected),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "methodology": {
            "profile": "Developing RTH volume profile, 09:30–16:00 Eastern",
            "value_area": "70%, single-row expansion, upward ties",
            "poc_tie": "highest price",
            "entry_window": "10:15 inclusive to 15:00 exclusive Eastern",
            "cooldown": "15 minutes after entry",
            "daily_trade_limit": MAX_DAILY_TRADES,
            "minimum_poc_distance": "30 points from triggering VA boundary",
            "first_target": "20 points",
            "runner_target": "opposite frozen value-area edge",
            "stop": "30 points; not moved to breakeven",
            "fills": "Last-trade prices only; quotes and queue position not used",
            "stop_slippage_ticks": STOP_SLIP_TICKS,
            "commission": "$0.50 per contract per side",
            "contract_expiry": "unknown",
        },
        "runs": summaries,
    }

    with (args.out / "summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)

    print("\nBACKTEST COMPLETE")
    print(json.dumps(result, indent=2))
    print(f"Reports saved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()