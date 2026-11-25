"""TopstepX live market data (SignalR) and developing value-area helper."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import numpy as np
from signalrcore.hub_connection_builder import HubConnectionBuilder

from backtest_va import BASE, SIZE, build_value_area

logger = logging.getLogger(__name__)
EASTERN = ZoneInfo("America/New_York")
RTC_MARKET_HUB = "https://rtc.topstepx.com/hubs/market"


def price_to_ticks(price: float) -> int:
    scaled = float(price) * 4.0
    return int(round(scaled))


def ticks_to_price(ticks: int) -> float:
    return ticks / 4.0


@dataclass
class LiveTrade:
    price_ticks: int
    volume: int
    timestamp: datetime
    raw_price: float


class DevelopingValueArea:
    """Incremental RTH volume profile (same binning as backtest_va)."""

    def __init__(self, base: int = BASE, size: int = SIZE):
        self.base = base
        self.size = size
        self.volume = np.zeros(size, dtype=np.int64)
        self.total = 0
        self.low = size
        self.high = -1
        self.poc = -1
        self.session_date: Optional[str] = None

    def reset(self):
        self.volume.fill(0)
        self.total = 0
        self.low = self.size
        self.high = -1
        self.poc = -1

    def maybe_roll_day(self, ts: datetime):
        day = ts.astimezone(EASTERN).date().isoformat()
        if self.session_date != day:
            self.session_date = day
            self.reset()

    def add(self, price_ticks: int, volume: int) -> Optional[tuple[int, int, int]]:
        if volume <= 0:
            return None
        bin_index = price_ticks - self.base
        if bin_index < 0 or bin_index >= self.size:
            logger.warning("Price tick %s outside profile range", price_ticks)
            return None

        self.volume[bin_index] += int(volume)
        self.total += int(volume)
        if bin_index < self.low:
            self.low = bin_index
        if bin_index > self.high:
            self.high = bin_index

        if self.poc < 0 or self.volume[bin_index] > self.volume[self.poc]:
            self.poc = bin_index
        elif self.volume[bin_index] == self.volume[self.poc] and bin_index > self.poc:
            self.poc = bin_index

        area_low, area_high = build_value_area(
            self.volume, self.low, self.high, self.poc, self.total
        )
        return (
            self.poc + self.base,
            area_high + self.base,
            area_low + self.base,
        )


def _unwrap_signalr_args(args: tuple[Any, ...]) -> tuple[Any, Any]:
    """signalrcore sometimes packs (contractId, payload) inside a single list arg."""
    if len(args) == 1 and isinstance(args[0], (list, tuple)) and len(args[0]) == 2:
        return args[0][0], args[0][1]
    if len(args) >= 2:
        return args[0], args[1]
    return None, None


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=EASTERN)
    if not value:
        return datetime.now(EASTERN)
    text = str(value)
    # Topstep sometimes sends 7-digit fractional seconds
    if "." in text:
        head, frac_and_tz = text.split(".", 1)
        digits = ""
        rest = ""
        for ch in frac_and_tz:
            if ch.isdigit():
                digits += ch
            else:
                rest = frac_and_tz[len(digits) :]
                break
        digits = (digits + "000000")[:6]
        text = f"{head}.{digits}{rest}"
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(EASTERN)
    return dt if dt.tzinfo else dt.replace(tzinfo=EASTERN)


class TopstepMarketHub:
    """Thin async-friendly wrapper around the ProjectX market SignalR hub."""

    def __init__(
        self,
        token_factory: Callable[[], str],
        contract_id: str,
        event_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        hub_url: str = RTC_MARKET_HUB,
    ):
        self.token_factory = token_factory
        self.contract_id = contract_id
        self.event_queue = event_queue
        self.loop = loop
        self.hub_url = hub_url.rstrip("/")
        self.connection = None
        self._started = False

    def _enqueue(self, item: Any):
        def _put():
            try:
                self.event_queue.put_nowait(item)
            except asyncio.QueueFull:
                pass

        self.loop.call_soon_threadsafe(_put)

    def _on_open(self):
        logger.info("Market hub connected; subscribing %s", self.contract_id)
        assert self.connection is not None
        self.connection.send("SubscribeContractQuotes", [self.contract_id])
        self.connection.send("SubscribeContractTrades", [self.contract_id])
        self._enqueue(("status", f"Subscribed to {self.contract_id}"))

    def _on_quote(self, *args):
        contract_id, payload = _unwrap_signalr_args(args)
        if not isinstance(payload, dict):
            return
        self._enqueue(("quote", contract_id or self.contract_id, payload))

    def _on_trade(self, *args):
        contract_id, payload = _unwrap_signalr_args(args)
        trades = payload if isinstance(payload, list) else [payload]
        for trade in trades:
            if isinstance(trade, dict):
                self._enqueue(("trade", contract_id or self.contract_id, trade))

    def start(self):
        token = self.token_factory()
        url = f"{self.hub_url}?access_token={token}"
        self.connection = (
            HubConnectionBuilder()
            .with_url(
                url,
                options={
                    "access_token_factory": self.token_factory,
                    "skip_negotiation": True,
                },
            )
            .with_automatic_reconnect(
                {
                    "type": "raw",
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                    "max_attempts": 10,
                }
            )
            .build()
        )
        self.connection.on_open(self._on_open)
        self.connection.on_reconnect(self._on_open)
        self.connection.on_close(self._on_close)
        self.connection.on_error(self._on_error)
        self.connection.on("GatewayQuote", self._on_quote)
        self.connection.on("GatewayTrade", self._on_trade)
        self.connection.start()
        self._started = True

    def _on_close(self):
        logger.warning("Market hub closed")
        self._enqueue(("hub_lost", "market hub closed"))

    def _on_error(self, data=None):
        logger.warning("Market hub error: %s", data)
        self._enqueue(("hub_lost", f"market hub error: {data}"))

    def stop(self):
        if self.connection and self._started:
            try:
                self.connection.send(
                    "UnsubscribeContractQuotes", [self.contract_id]
                )
                self.connection.send(
                    "UnsubscribeContractTrades", [self.contract_id]
                )
            except Exception:
                pass
            try:
                self.connection.stop()
            except Exception:
                pass
        self._started = False


def trade_from_payload(payload: dict) -> Optional[LiveTrade]:
    price = payload.get("price")
    if price is None:
        return None
    volume = int(payload.get("volume") or 0)
    if volume <= 0:
        volume = 1
    return LiveTrade(
        price_ticks=price_to_ticks(price),
        volume=volume,
        timestamp=_parse_timestamp(payload.get("timestamp")),
        raw_price=float(price),
    )


def seed_value_area_from_bars(
    value_area: DevelopingValueArea, bars: list[dict]
) -> tuple[Optional[tuple[int, int, int]], Optional[float], int]:
    """Fold OHLC bars into the developing VA.

    Volume is spread across the bar's [low, high] tick range so midday
    restarts approximate a tick volume profile without tick history.
    Returns (profile_tuple_or_None, last_close_price, bars_used).
    """
    profile = None
    last_close = None
    used = 0

    # History API often returns newest-first
    ordered = sorted(bars, key=lambda b: str(b.get("t") or ""))

    for bar in ordered:
        try:
            low = price_to_ticks(float(bar["l"]))
            high = price_to_ticks(float(bar["h"]))
            close = float(bar["c"])
            volume = int(bar.get("v") or 0)
        except (KeyError, TypeError, ValueError):
            continue

        if volume <= 0:
            continue
        if high < low:
            low, high = high, low

        ts = _parse_timestamp(bar.get("t"))
        value_area.maybe_roll_day(ts)

        span = high - low + 1
        if span <= 1:
            profile = value_area.add(low, volume) or profile
        else:
            each = volume // span
            rem = volume % span
            for i, tick in enumerate(range(low, high + 1)):
                part = each + (1 if i < rem else 0)
                if part > 0:
                    profile = value_area.add(tick, part) or profile

        last_close = close
        used += 1

    return profile, last_close, used
