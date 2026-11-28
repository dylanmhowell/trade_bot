"""FastAPI backend for K_two_lot_60 strategy with Topstep API."""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, time as clock, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from zoneinfo import ZoneInfo

from backtest_va import from_100ns, rth_profile_for_session
from config import settings
from live_log import LiveDiagLog
from topstep_live import (
    DevelopingValueArea,
    TopstepMarketHub,
    price_to_ticks,
    seed_value_area_from_bars,
    ticks_to_price,
    trade_from_payload,
)


# ============================================================================
# Configuration
# ============================================================================

TOPSTEP_USERNAME = settings.topstep_username
TOPSTEP_API_KEY = settings.topstep_api_key
TOPSTEP_BASE_URL = settings.topstep_base_url
TOPSTEP_ACCOUNT_ID = settings.topstep_account_id
TOPSTEP_CONTRACT_ID = settings.topstep_contract_id
TOPSTEP_SYMBOL_SEARCH = settings.topstep_symbol_search
TOPSTEP_CONTRACT_LIVE = settings.topstep_contract_live
TOPSTEP_RTC_URL = settings.topstep_rtc_url
TOPSTEP_ENABLE_ORDERS = settings.topstep_enable_orders

EASTERN = ZoneInfo("America/New_York")

# K_two_lot_60 parameters (TP1 = 40 points)
STOP_TICKS = settings.stop_ticks          # 60 points
LOTS = settings.lots
TP1_TICKS = settings.tp1_ticks            # 40 points
USE_POC_RUNG = False
USE_OPP_RUNG = True
RUNNER_TICKS = 0          # Runner goes to opposite VA edge
MIN_POC_DIST = 120        # 30 points
ENTRY_HOUR, ENTRY_MINUTE = 10, 15
ENTRY_END = clock(15, 0)
RTH_START = clock(9, 30)
RTH_END = clock(16, 0)

TICK_CENTS = 50
COMMISSION_CENTS = 50
STOP_SLIP_TICKS = 1

SESSIONS_DIR = Path("output/sessions")
REPLAY_SPEEDS = {"10": 10.0, "50": 50.0, "max": None}
BROKER_SYNC_INTERVAL_SEC = 5.0
SESSION_HEALTH_INTERVAL_SEC = 20.0
SESSION_RECLAIM_COOLDOWN_SEC = 15.0


# ============================================================================
# Data Models
# ============================================================================

class TopstepAuthError(RuntimeError):
    """Raised when TopstepX returns 401/403 or an invalid/expired token."""


class RunnerTPRequest(BaseModel):
    price: float


class State(Enum):
    FLAT = "flat"
    ARMED_SHORT = "armed_short"  # Below VAH, waiting for cross up through VAH
    ARMED_LONG = "armed_long"    # Above VAL, waiting for cross down through VAL
    IN_POSITION = "in_position"


@dataclass
class Profile:
    poc: int = 0
    vah: int = 0
    val: int = 0
    timestamp: Optional[datetime] = None


@dataclass
class Leg:
    open: bool = False
    target: int = 0
    stop: int = 0
    kind: str = ""  # "tp1", "opp"
    filled_price: Optional[int] = None
    filled_time: Optional[datetime] = None


@dataclass
class Position:
    direction: int = 0          # 1 = long, -1 = short
    entry_price: int = 0
    entry_time: Optional[datetime] = None
    legs: list = None
    gross_pnl_cents: int = 0

    def __post_init__(self):
        if self.legs is None:
            self.legs = [Leg(), Leg()]


@dataclass
class TradeSummary:
    date: str = ""
    entry_time: str = ""
    exit_time: str = ""
    direction: str = ""
    entry_price: float = 0.0
    gross_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    exit_reasons: list = None

    def __post_init__(self):
        if self.exit_reasons is None:
            self.exit_reasons = []


# ============================================================================
# Strategy Engine
# ============================================================================

class KTwoLot60Engine:
    """Real-time strategy engine for K_two_lot_60."""

    def __init__(self, diag: Optional[LiveDiagLog] = None):
        self.state = State.FLAT
        self.profile = Profile()
        self.position: Optional[Position] = None
        self.origin = 0           # 1 = above POC, -1 = below POC
        self.prev_price: Optional[int] = None
        self.mark_price: Optional[int] = None
        self.diag = diag

        # Session state
        self.session_date: Optional[str] = None
        self.entry_window_open = False
        self.rth_active = False

        # Statistics
        self.daily_trades: list[TradeSummary] = []
        self.total_commission_cents = 0
        self.daily_realized_cents = 0
        self.equity_peak_cents = 0
        self.max_drawdown_cents = 0
        # Live mode clears this until today's VA has been seeded from history
        self.entries_enabled = True

    def _px(self, ticks: Optional[int]) -> Optional[float]:
        return None if ticks is None else ticks / 4.0

    def update_profile(self, poc: int, vah: int, val: int, ts: datetime):
        """Update value area profile from external source."""
        self.profile = Profile(poc=poc, vah=vah, val=val, timestamp=ts)

    def on_tick(self, price: int, ts: datetime, is_rth: bool) -> list[dict]:
        """Process a single tick. Returns list of actions to execute."""
        actions = []
        self.mark_price = price

        # Update session timing
        self._update_session_state(ts, is_rth)

        if not is_rth:
            # Force close any position at RTH end
            if self.position and self.position.direction != 0:
                actions.extend(self._close_all("session_close", price, ts))
            self.prev_price = price
            self._mark_equity(self._unrealized_cents())
            return actions

        # Update crossing state
        self._update_crossing_state(price, ts)

        # Check for exits first
        if self.position and self.position.direction != 0:
            exit_actions = self._check_exits(price, ts)
            if exit_actions:
                actions.extend(exit_actions)
                if self.state == State.FLAT:
                    self.prev_price = price
                    self._mark_equity(self._unrealized_cents())
                    return actions  # Position closed

        # Check for entry (must already be armed from a POC cross)
        if (
            self.state in (State.ARMED_SHORT, State.ARMED_LONG)
            and not self.position
        ):
            if not self.entries_enabled:
                # Only note when a trigger cross actually happens
                self._check_entry(price, ts, log_only_block="entries_disabled")
            elif not self.entry_window_open:
                self._check_entry(
                    price, ts, log_only_block="outside_entry_window"
                )
            else:
                entry_action = self._check_entry(price, ts)
                if entry_action:
                    actions.append(entry_action)

        self.prev_price = price
        self._mark_equity(self._unrealized_cents())
        return actions

    def _update_session_state(self, ts: datetime, is_rth: bool):
        """Update entry window and RTH state."""
        self.rth_active = is_rth

        ts_et = ts.astimezone(EASTERN)
        t = ts_et.time()
        date = ts_et.date().isoformat()

        if date != self.session_date:
            self.session_date = date
            self.daily_trades = []
            self.daily_realized_cents = 0
            self.equity_peak_cents = 0
            self.max_drawdown_cents = 0
            self.total_commission_cents = 0
            if not self.position:
                self.state = State.FLAT
                self.origin = 0
                self.prev_price = None
                self.mark_price = None

        entry_start = clock(ENTRY_HOUR, ENTRY_MINUTE)
        self.entry_window_open = entry_start <= t < ENTRY_END
        if self.diag:
            self.diag.entry_window(
                self.entry_window_open,
                ts=ts,
                time_et=t.isoformat(timespec="seconds"),
            )

    def _unrealized_cents(self) -> int:
        ref_price = (
            self.mark_price if self.mark_price is not None else self.prev_price
        )
        if not self.position or ref_price is None:
            return 0
        unrealized = 0
        for leg in self.position.legs:
            if leg.open:
                ch = (ref_price - self.position.entry_price) * (
                    self.position.direction
                )
                unrealized += ch * TICK_CENTS
        return unrealized

    def _mark_equity(self, unrealized_cents: int):
        """Track peak equity and max drawdown for the session day."""
        equity = self.daily_realized_cents + unrealized_cents
        if equity > self.equity_peak_cents:
            self.equity_peak_cents = equity
        drawdown = self.equity_peak_cents - equity
        if drawdown > self.max_drawdown_cents:
            self.max_drawdown_cents = drawdown

    def _update_crossing_state(
        self, price: int, ts: Optional[datetime] = None
    ):
        """Update POC crossing armed state."""
        if self.profile.poc == 0:
            return

        prev_state = self.state

        # Never overwrite an open position's state with arming flags
        if self.state != State.IN_POSITION:
            if self.origin == -1 and price > self.profile.poc:
                self.state = State.ARMED_SHORT
            elif self.origin == 1 and price < self.profile.poc:
                self.state = State.ARMED_LONG

        # Preserve prior origin when price prints directly on the POC
        if price > self.profile.poc:
            self.origin = 1
        elif price < self.profile.poc:
            self.origin = -1

        if self.diag and self.state != prev_state:
            self.diag.state_change(
                self.state.value,
                ts=ts,
                price=self._px(price),
                poc=self._px(self.profile.poc),
                vah=self._px(self.profile.vah),
                val=self._px(self.profile.val),
                origin=self.origin,
            )

    def _check_entry(
        self,
        price: int,
        ts: datetime,
        log_only_block: Optional[str] = None,
    ) -> Optional[dict]:
        """Check if entry conditions met."""
        if self.profile.vah == 0 or self.profile.val == 0:
            return None

        if self.state == State.ARMED_SHORT:
            trigger = self.profile.vah
            # Must be below VAH and cross up through it
            if (
                self.prev_price is not None
                and self.prev_price < trigger
                and price >= trigger
            ):
                dist = abs(self.profile.poc - trigger)
                if dist < MIN_POC_DIST:
                    if self.diag:
                        self.diag.near_miss(
                            "poc_distance",
                            ts=ts,
                            side="short",
                            price=self._px(price),
                            trigger=self._px(trigger),
                            poc=self._px(self.profile.poc),
                            dist_pts=dist / 4.0,
                            min_pts=MIN_POC_DIST / 4.0,
                        )
                    return None
                if log_only_block:
                    if self.diag:
                        self.diag.near_miss(
                            log_only_block,
                            ts=ts,
                            side="short",
                            price=self._px(price),
                            trigger=self._px(trigger),
                            poc=self._px(self.profile.poc),
                        )
                    return None
                return self._enter_position(-1, price, ts, trigger)

        elif self.state == State.ARMED_LONG:
            trigger = self.profile.val
            # Must be above VAL and cross down through it
            if (
                self.prev_price is not None
                and self.prev_price > trigger
                and price <= trigger
            ):
                dist = abs(self.profile.poc - trigger)
                if dist < MIN_POC_DIST:
                    if self.diag:
                        self.diag.near_miss(
                            "poc_distance",
                            ts=ts,
                            side="long",
                            price=self._px(price),
                            trigger=self._px(trigger),
                            poc=self._px(self.profile.poc),
                            dist_pts=dist / 4.0,
                            min_pts=MIN_POC_DIST / 4.0,
                        )
                    return None
                if log_only_block:
                    if self.diag:
                        self.diag.near_miss(
                            log_only_block,
                            ts=ts,
                            side="long",
                            price=self._px(price),
                            trigger=self._px(trigger),
                            poc=self._px(self.profile.poc),
                        )
                    return None
                return self._enter_position(1, price, ts, trigger)

        return None

    def _enter_position(
        self,
        direction: int,
        price: int,
        ts: datetime,
        trigger_price: int,
    ) -> dict:
        """Create entry action and initialize position."""
        self.position = Position(
            direction=direction,
            entry_price=price,
            entry_time=ts,
        )
        self.state = State.IN_POSITION

        # Calculate targets and stops
        stop = price - direction * STOP_TICKS

        # Leg 1: TP1 at 40 points
        self.position.legs[0].open = True
        self.position.legs[0].target = price + direction * TP1_TICKS
        self.position.legs[0].stop = stop
        self.position.legs[0].kind = "tp1"

        # Leg 2: Opposite VA edge (or stop out)
        opp_target = self.profile.val if direction == -1 else self.profile.vah
        self.position.legs[1].open = True
        self.position.legs[1].target = opp_target
        self.position.legs[1].stop = stop
        self.position.legs[1].kind = "opp"

        self.total_commission_cents += LOTS * COMMISSION_CENTS

        if self.diag:
            self.diag.info(
                "entry",
                ts=ts,
                direction="short" if direction == -1 else "long",
                price=self._px(price),
                trigger=self._px(trigger_price),
                stop=self._px(stop),
                tp1=self._px(self.position.legs[0].target),
                opp=self._px(opp_target),
                poc=self._px(self.profile.poc),
                vah=self._px(self.profile.vah),
                val=self._px(self.profile.val),
                qty=LOTS,
            )
            self.diag.state_change(
                self.state.value,
                ts=ts,
                price=self._px(price),
            )

        return {
            "action": "enter",
            "direction": "short" if direction == -1 else "long",
            "price": price,
            "quantity": LOTS,
            "legs": [
                {
                    "target": self.position.legs[0].target,
                    "stop": stop,
                    "kind": "tp1",
                },
                {
                    "target": self.position.legs[1].target,
                    "stop": stop,
                    "kind": "opp",
                },
            ],
            "timestamp": ts.isoformat(),
        }

    def sync_entry_fill_price(self, fill_ticks: int) -> Optional[list[dict]]:
        """Re-anchor entry, TP1, and stop to the actual broker fill price."""
        pos = self.position
        if not pos or pos.direction == 0:
            return None
        pos.entry_price = fill_ticks
        stop = fill_ticks - pos.direction * STOP_TICKS
        pos.legs[0].target = fill_ticks + pos.direction * TP1_TICKS
        pos.legs[0].stop = stop
        pos.legs[1].stop = stop
        return [
            {
                "target": pos.legs[0].target,
                "stop": stop,
                "kind": pos.legs[0].kind,
            },
            {
                "target": pos.legs[1].target,
                "stop": stop,
                "kind": pos.legs[1].kind,
            },
        ]

    def abort_unfilled_entry(self, ts: Optional[datetime] = None):
        """Roll back local position state if the broker rejected an entry."""
        if not self.position:
            return
        self.total_commission_cents = max(
            0, self.total_commission_cents - LOTS * COMMISSION_CENTS
        )
        self.position = None
        self.state = State.FLAT
        if self.diag:
            self.diag.info("entry_aborted", ts=ts)
            self.diag.state_change(self.state.value, ts=ts)

    def _check_exits(self, price: int, ts: datetime) -> list[dict]:
        """Check if any legs should exit."""
        actions = []
        pos = self.position
        if not pos or pos.direction == 0:
            return actions

        all_closed = True

        for i, leg in enumerate(pos.legs):
            if not leg.open:
                continue

            # Check stop first
            stop_hit = False
            if pos.direction == 1 and price <= leg.stop:
                stop_hit = True
                fill = leg.stop - STOP_SLIP_TICKS
            elif pos.direction == -1 and price >= leg.stop:
                stop_hit = True
                fill = leg.stop + STOP_SLIP_TICKS

            if stop_hit:
                leg.open = False
                leg.filled_price = fill
                leg.filled_time = ts
                ch = (fill - pos.entry_price) * pos.direction
                pos.gross_pnl_cents += ch * TICK_CENTS
                actions.append({
                    "action": "exit",
                    "leg": i,
                    "kind": leg.kind,
                    "price": fill,
                    "reason": "stop",
                    "timestamp": ts.isoformat(),
                })
                if self.diag:
                    self.diag.info(
                        "exit",
                        ts=ts,
                        reason="stop",
                        leg=i,
                        kind=leg.kind,
                        fill=self._px(fill),
                        entry=self._px(pos.entry_price),
                    )
                continue

            # Check target
            target_hit = False
            if pos.direction == 1 and price >= leg.target:
                target_hit = True
                fill = leg.target
            elif pos.direction == -1 and price <= leg.target:
                target_hit = True
                fill = leg.target

            if target_hit:
                leg.open = False
                leg.filled_price = fill
                leg.filled_time = ts
                ch = (fill - pos.entry_price) * pos.direction
                pos.gross_pnl_cents += ch * TICK_CENTS
                actions.append({
                    "action": "exit",
                    "leg": i,
                    "kind": leg.kind,
                    "price": fill,
                    "reason": "target",
                    "timestamp": ts.isoformat(),
                })
                if self.diag:
                    self.diag.info(
                        "exit",
                        ts=ts,
                        reason="target",
                        leg=i,
                        kind=leg.kind,
                        fill=self._px(fill),
                        entry=self._px(pos.entry_price),
                    )
                continue

            all_closed = False

        if all_closed:
            actions.extend(self._record_trade(ts))
            self.position = None
            self.state = State.FLAT
            if self.diag:
                self.diag.state_change(self.state.value, ts=ts)

        return actions

    def _close_all(self, reason: str, price: int, ts: datetime) -> list[dict]:
        """Force close all legs at market."""
        actions = []
        pos = self.position
        if not pos:
            return actions

        fill = (
            price - STOP_SLIP_TICKS
            if pos.direction == 1
            else price + STOP_SLIP_TICKS
        )

        for i, leg in enumerate(pos.legs):
            if leg.open:
                leg.open = False
                leg.filled_price = fill
                leg.filled_time = ts
                ch = (fill - pos.entry_price) * pos.direction
                pos.gross_pnl_cents += ch * TICK_CENTS
                actions.append({
                    "action": "exit",
                    "leg": i,
                    "kind": leg.kind,
                    "price": fill,
                    "reason": reason,
                    "timestamp": ts.isoformat(),
                })
                if self.diag:
                    self.diag.info(
                        "exit",
                        ts=ts,
                        reason=reason,
                        leg=i,
                        kind=leg.kind,
                        fill=self._px(fill),
                        entry=self._px(pos.entry_price),
                    )

        actions.extend(self._record_trade(ts))
        self.position = None
        self.state = State.FLAT
        if self.diag:
            self.diag.state_change(self.state.value, ts=ts)
        return actions

    def _close_n_legs(
        self, n: int, reason: str, price: int, ts: datetime
    ) -> list[dict]:
        """Close up to n open legs at mark (broker size reconciliation)."""
        actions = []
        pos = self.position
        if not pos or n <= 0:
            return actions

        fill = (
            price - STOP_SLIP_TICKS
            if pos.direction == 1
            else price + STOP_SLIP_TICKS
        )
        closed = 0
        for i, leg in enumerate(pos.legs):
            if closed >= n:
                break
            if not leg.open:
                continue
            leg.open = False
            leg.filled_price = fill
            leg.filled_time = ts
            ch = (fill - pos.entry_price) * pos.direction
            pos.gross_pnl_cents += ch * TICK_CENTS
            actions.append({
                "action": "exit",
                "leg": i,
                "kind": leg.kind,
                "price": fill,
                "reason": reason,
                "timestamp": ts.isoformat(),
            })
            if self.diag:
                self.diag.info(
                    "exit",
                    ts=ts,
                    reason=reason,
                    leg=i,
                    kind=leg.kind,
                    fill=self._px(fill),
                    entry=self._px(pos.entry_price),
                )
            closed += 1

        if not any(leg.open for leg in pos.legs):
            actions.extend(self._record_trade(ts))
            self.position = None
            self.state = State.FLAT
            if self.diag:
                self.diag.state_change(self.state.value, ts=ts)

        return actions

    def open_leg_count(self) -> int:
        if not self.position:
            return 0
        return sum(1 for leg in self.position.legs if leg.open)

    def _record_trade(self, ts: datetime) -> list[dict]:
        """Record completed trade to history."""
        pos = self.position
        if not pos:
            return []

        commission = LOTS * COMMISSION_CENTS
        net = pos.gross_pnl_cents - commission

        summary = TradeSummary(
            date=self.session_date or ts.astimezone(EASTERN).date().isoformat(),
            entry_time=pos.entry_time.isoformat() if pos.entry_time else "",
            exit_time=ts.isoformat(),
            direction="long" if pos.direction == 1 else "short",
            entry_price=pos.entry_price / 4.0,
            gross_pnl=pos.gross_pnl_cents / 100.0,
            commission=commission / 100.0,
            net_pnl=net / 100.0,
            exit_reasons=[
                {
                    "leg": i,
                    "kind": leg.kind,
                    "price": leg.filled_price / 4.0 if leg.filled_price else 0,
                    "time": (
                        leg.filled_time.isoformat() if leg.filled_time else ""
                    ),
                }
                for i, leg in enumerate(pos.legs)
            ],
        )

        self.daily_trades.append(summary)
        self.daily_realized_cents += net

        return [{
            "action": "trade_summary",
            "data": asdict(summary),
        }]

    def get_status(self) -> dict:
        """Get current engine status."""
        unrealized = self._unrealized_cents()
        self._mark_equity(unrealized)
        daily_pnl = self.daily_realized_cents + unrealized

        trigger = None
        trigger_side = None
        poc_dist_ok = None
        if self.state == State.ARMED_SHORT and self.profile.vah:
            trigger = self.profile.vah / 4.0
            trigger_side = "short"
            poc_dist_ok = abs(self.profile.poc - self.profile.vah) >= (
                MIN_POC_DIST
            )
        elif self.state == State.ARMED_LONG and self.profile.val:
            trigger = self.profile.val / 4.0
            trigger_side = "long"
            poc_dist_ok = abs(self.profile.poc - self.profile.val) >= (
                MIN_POC_DIST
            )

        armed = self.state in (State.ARMED_SHORT, State.ARMED_LONG)
        can_enter = bool(
            self.entries_enabled
            and self.entry_window_open
            and armed
            and not self.position
            and poc_dist_ok
        )

        last_ticks = (
            self.mark_price if self.mark_price is not None else self.prev_price
        )

        return {
            "state": self.state.value,
            "in_position": (
                self.position is not None and self.position.direction != 0
            ),
            "direction": (
                "long"
                if self.position and self.position.direction == 1
                else "short"
                if self.position and self.position.direction == -1
                else "flat"
            ),
            "entry_price": (
                self.position.entry_price / 4.0 if self.position else 0
            ),
            "last_price": last_ticks / 4.0 if last_ticks is not None else None,
            "unrealized_pnl": unrealized / 100.0,
            "daily_pnl": daily_pnl / 100.0,
            "realized_pnl": self.daily_realized_cents / 100.0,
            "max_drawdown": self.max_drawdown_cents / 100.0,
            "entries_enabled": self.entries_enabled,
            "entry_window_open": self.entry_window_open,
            "rth_active": self.rth_active,
            "origin": self.origin,
            "armed": armed,
            "trigger_side": trigger_side,
            "trigger_price": trigger,
            "poc_distance_ok": poc_dist_ok,
            "can_enter": can_enter,
            "daily_trades": len(self.daily_trades),
            "total_commission": self.total_commission_cents / 100.0,
            "profile": {
                "poc": self.profile.poc / 4.0,
                "vah": self.profile.vah / 4.0,
                "val": self.profile.val / 4.0,
            }
            if self.profile.poc
            else None,
            "trade_levels": self._trade_levels(),
        }

    def _trade_levels(self) -> Optional[dict]:
        """Active entry / TP / stop levels for the open position."""
        pos = self.position
        if not pos or pos.direction == 0:
            return None
        tp1 = pos.legs[0] if len(pos.legs) > 0 else None
        opp = pos.legs[1] if len(pos.legs) > 1 else None
        stop = None
        if tp1 and tp1.open:
            stop = tp1.stop / 4.0
        elif opp and opp.open:
            stop = opp.stop / 4.0
        return {
            "direction": "long" if pos.direction == 1 else "short",
            "entry": pos.entry_price / 4.0,
            "tp1": tp1.target / 4.0 if tp1 and tp1.open else None,
            "tp1_open": bool(tp1 and tp1.open),
            "runner": opp.target / 4.0 if opp and opp.open else None,
            "runner_open": bool(opp and opp.open),
            "stop": stop,
            "stop_open": stop is not None,
        }

    def set_runner_target(self, price_ticks: int) -> bool:
        """Update the open opposite-VA (runner) take-profit target."""
        pos = self.position
        if not pos or pos.direction == 0:
            return False
        if len(pos.legs) < 2 or not pos.legs[1].open:
            return False
        # Keep target on the profit side of entry
        if pos.direction == 1 and price_ticks <= pos.entry_price:
            return False
        if pos.direction == -1 and price_ticks >= pos.entry_price:
            return False
        pos.legs[1].target = int(price_ticks)
        pos.legs[1].kind = "opp"
        return True


# ============================================================================
# TopstepX / ProjectX Gateway API Client
# Docs: https://gateway.docs.projectx.com/
# ============================================================================

class TopstepClient:
    """Client for the TopstepX ProjectX Gateway REST API."""

    ORDER_TYPE_LIMIT = 1
    ORDER_TYPE_MARKET = 2
    ORDER_TYPE_STOP = 4
    ORDER_TYPE_TRAILING_STOP = 5
    SIDE_BID = 0  # buy
    SIDE_ASK = 1  # sell

    def __init__(
        self,
        user_name: str,
        api_key: str,
        base_url: str = TOPSTEP_BASE_URL,
        account_id: Optional[int] = None,
    ):
        self.user_name = user_name
        self.api_key = api_key
        self.account_id = account_id
        self.token: Optional[str] = None
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=30.0
        )

    def _headers(self) -> dict:
        if not self.token:
            raise TopstepAuthError(
                "Not authenticated — call authenticate() first"
            )
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "accept": "text/plain",
        }

    async def _post(
        self,
        path: str,
        payload: Optional[dict] = None,
        auth: bool = True,
    ) -> dict:
        headers = self._headers() if auth else None
        kwargs: dict = {}
        if headers is not None:
            kwargs["headers"] = headers
        if payload is not None:
            kwargs["json"] = payload

        response = await self.client.post(path, **kwargs)
        if response.status_code in (401, 403):
            raise TopstepAuthError(
                f"HTTP {response.status_code} on {path}"
            )
        if response.status_code >= 400:
            snippet = response.text[:200] if response.text else ""
            raise RuntimeError(
                f"HTTP {response.status_code} on {path}: {snippet}"
            )
        try:
            return response.json()
        except ValueError as e:
            snippet = response.text[:200] if response.text else ""
            raise RuntimeError(
                f"Invalid JSON from {path} (HTTP {response.status_code}): "
                f"{snippet}"
            ) from e

    async def authenticate(self) -> bool:
        """POST /api/Auth/loginKey — returns a JWT valid ~24h."""
        try:
            data = await self._post(
                "/api/Auth/loginKey",
                payload={"userName": self.user_name, "apiKey": self.api_key},
                auth=False,
            )
            if not data.get("success") or not data.get("token"):
                print(
                    f"Auth failed: errorCode={data.get('errorCode')} "
                    f"message={data.get('errorMessage')}"
                )
                return False

            self.token = data["token"]

            if self.account_id is None:
                accounts = await self.search_accounts(only_active=True)
                tradeable = [a for a in accounts if a.get("canTrade", True)]
                chosen = (
                    tradeable[0]
                    if tradeable
                    else (accounts[0] if accounts else None)
                )
                if chosen:
                    self.account_id = chosen["id"]
                    print(
                        f"Using Topstep account {self.account_id} "
                        f"({chosen.get('name')})"
                    )
                else:
                    print("Auth ok but no accounts found")
                    return False
            return True
        except Exception as e:
            print(f"Auth failed: {e}")
            return False

    async def validate_session(self) -> bool:
        """POST /api/Auth/validate — refresh token before 24h expiry."""
        try:
            data = await self._post("/api/Auth/validate")
        except TopstepAuthError:
            return False
        if data.get("success") and data.get("newToken"):
            self.token = data["newToken"]
            return True
        return bool(data.get("success"))

    async def session_healthy(self) -> bool:
        """Cheap auth probe using token validation."""
        try:
            return await self.validate_session()
        except Exception:
            return False

    async def search_accounts(self, only_active: bool = True) -> list[dict]:
        """POST /api/Account/search"""
        data = await self._post(
            "/api/Account/search",
            payload={"onlyActiveAccounts": only_active},
        )
        if not data.get("success"):
            raise RuntimeError(
                data.get("errorMessage") or "Account search failed"
            )
        return data.get("accounts") or []

    async def get_account(self) -> dict:
        accounts = await self.search_accounts(only_active=True)
        for acct in accounts:
            if self.account_id is None or acct.get("id") == self.account_id:
                return acct
        return accounts[0] if accounts else {"error": "No accounts"}

    async def search_open_positions(self) -> list[dict]:
        """POST /api/Position/searchOpen"""
        if self.account_id is None:
            raise RuntimeError("account_id is required")
        data = await self._post(
            "/api/Position/searchOpen",
            payload={"accountId": self.account_id},
        )
        if not data.get("success"):
            raise RuntimeError(
                data.get("errorMessage") or "Position search failed"
            )
        return data.get("positions") or []

    async def search_trades(
        self,
        start_timestamp: datetime,
        end_timestamp: Optional[datetime] = None,
    ) -> list[dict]:
        """POST /api/Trade/search — half-turn fills for the account."""
        if self.account_id is None:
            raise RuntimeError("account_id is required")
        payload = {
            "accountId": self.account_id,
            "startTimestamp": start_timestamp.isoformat(),
        }
        if end_timestamp is not None:
            payload["endTimestamp"] = end_timestamp.isoformat()
        data = await self._post("/api/Trade/search", payload=payload)
        if not data.get("success"):
            raise RuntimeError(
                data.get("errorMessage") or "Trade search failed"
            )
        return data.get("trades") or []

    async def search_open_orders(self) -> list[dict]:
        """POST /api/Order/searchOpen"""
        if self.account_id is None:
            raise RuntimeError("account_id is required")
        data = await self._post(
            "/api/Order/searchOpen",
            payload={"accountId": self.account_id},
        )
        if not data.get("success"):
            raise RuntimeError(
                data.get("errorMessage") or "Open order search failed"
            )
        return data.get("orders") or []

    async def place_order(
        self,
        contract_id: str,
        side: int,
        size: int,
        order_type: int = ORDER_TYPE_MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        custom_tag: Optional[str] = None,
    ) -> dict:
        """POST /api/Order/place"""
        payload = {
            "accountId": self.account_id,
            "contractId": contract_id,
            "type": order_type,
            "side": side,
            "size": size,
            "limitPrice": limit_price,
            "stopPrice": stop_price,
            "trailPrice": None,
            "customTag": custom_tag,
        }
        return await self._post("/api/Order/place", payload=payload)

    async def cancel_order(self, order_id: int) -> dict:
        """POST /api/Order/cancel"""
        return await self._post(
            "/api/Order/cancel",
            payload={"accountId": self.account_id, "orderId": order_id},
        )

    async def modify_order(
        self,
        order_id: int,
        size: Optional[int] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> dict:
        """POST /api/Order/modify"""
        payload = {
            "accountId": self.account_id,
            "orderId": order_id,
            "size": size,
            "limitPrice": limit_price,
            "stopPrice": stop_price,
            "trailPrice": None,
        }
        return await self._post("/api/Order/modify", payload=payload)

    async def cancel_open_orders(
        self, contract_id: Optional[str] = None
    ) -> list[dict]:
        """Cancel all open orders, optionally filtered to one contract."""
        cancelled = []
        for order in await self.search_open_orders():
            if contract_id and order.get("contractId") != contract_id:
                continue
            order_id = order.get("id")
            if order_id is None:
                continue
            result = await self.cancel_order(order_id)
            cancelled.append({"orderId": order_id, **result})
        return cancelled

    async def place_exit_brackets(
        self,
        contract_id: str,
        direction: str,
        legs: list[dict],
    ) -> dict:
        """Place stop (2), TP1 limit (1), and runner limit (1) after entry."""
        if len(legs) < 2:
            raise RuntimeError("Exit brackets require tp1 and runner legs")

        exit_side = self.SIDE_ASK if direction == "long" else self.SIDE_BID
        tp1_px = legs[0]["target"] / 4.0
        opp_px = legs[1]["target"] / 4.0
        stop_px = legs[0]["stop"] / 4.0

        # Place protective stop first
        stop = await self.place_order(
            contract_id=contract_id,
            side=exit_side,
            size=LOTS,
            order_type=self.ORDER_TYPE_STOP,
            stop_price=stop_px,
            custom_tag=f"stop-{uuid.uuid4().hex[:10]}",
        )
        tp1 = await self.place_order(
            contract_id=contract_id,
            side=exit_side,
            size=1,
            order_type=self.ORDER_TYPE_LIMIT,
            limit_price=tp1_px,
            custom_tag=f"tp1-{uuid.uuid4().hex[:10]}",
        )
        opp = await self.place_order(
            contract_id=contract_id,
            side=exit_side,
            size=1,
            order_type=self.ORDER_TYPE_LIMIT,
            limit_price=opp_px,
            custom_tag=f"opp-{uuid.uuid4().hex[:10]}",
        )
        return {
            "tp1": tp1,
            "opp": opp,
            "stop": stop,
            "prices": {"tp1": tp1_px, "runner": opp_px, "stop": stop_px},
        }

    async def close_contract(self, contract_id: str) -> dict:
        """POST /api/Position/closeContract — flatten one contract."""
        return await self._post(
            "/api/Position/closeContract",
            payload={
                "accountId": self.account_id,
                "contractId": contract_id,
            },
        )

    async def partial_close_contract(self, contract_id: str, size: int) -> dict:
        """POST /api/Position/partialCloseContract"""
        return await self._post(
            "/api/Position/partialCloseContract",
            payload={
                "accountId": self.account_id,
                "contractId": contract_id,
                "size": size,
            },
        )

    async def search_contracts(
        self, search_text: str, live: bool = False
    ) -> list[dict]:
        """POST /api/Contract/search"""
        data = await self._post(
            "/api/Contract/search",
            payload={"searchText": search_text, "live": live},
        )
        if not data.get("success"):
            raise RuntimeError(
                data.get("errorMessage") or "Contract search failed"
            )
        return data.get("contracts") or []

    async def retrieve_bars(
        self,
        contract_id: str,
        start_time: datetime,
        end_time: datetime,
        unit: int = 1,
        unit_number: int = 1,
        limit: int = 20000,
        live: Optional[bool] = None,
        include_partial_bar: bool = True,
    ) -> list[dict]:
        """POST /api/History/retrieveBars (max 20,000 bars per request)."""
        if live is None:
            live = TOPSTEP_CONTRACT_LIVE
        data = await self._post(
            "/api/History/retrieveBars",
            payload={
                "contractId": contract_id,
                "live": live,
                "startTime": start_time.isoformat(),
                "endTime": end_time.isoformat(),
                "unit": unit,
                "unitNumber": unit_number,
                "limit": limit,
                "includePartialBar": include_partial_bar,
            },
        )
        if not data.get("success"):
            raise RuntimeError(
                data.get("errorMessage")
                or f"retrieveBars failed (errorCode={data.get('errorCode')})"
            )
        return data.get("bars") or []

    async def resolve_contract(
        self,
        contract_id: Optional[str] = None,
        search_text: str = "MNQ",
        live: bool = False,
    ) -> dict:
        """Resolve an explicit contract id or the active front-month match."""
        if contract_id:
            data = await self._post(
                "/api/Contract/searchById",
                payload={"contractId": contract_id},
            )
            if data.get("success") and data.get("contract"):
                return data["contract"]
            raise RuntimeError(f"Contract not found: {contract_id}")

        contracts = await self.search_contracts(search_text, live=live)
        if not contracts and live:
            # Practice accounts often only resolve with live=false
            contracts = await self.search_contracts(search_text, live=False)

        needle = search_text.upper()
        preferred = [
            c
            for c in contracts
            if c.get("activeContract")
            and (
                needle in str(c.get("name", "")).upper()
                or needle in str(c.get("symbolId", "")).upper()
                or needle in str(c.get("id", "")).upper()
            )
        ]
        if not preferred:
            preferred = [c for c in contracts if c.get("activeContract")]
        if not preferred and contracts:
            preferred = contracts
        if not preferred:
            raise RuntimeError(f"No contracts found for search '{search_text}'")
        return preferred[0]

    async def execute_strategy_action(
        self, action: dict, contract_id: str, enable_orders: bool = True
    ) -> Optional[dict]:
        """Map engine enter/exit actions onto Topstep endpoints."""
        kind = action.get("action")
        if kind not in ("enter", "exit"):
            return None

        if not enable_orders:
            return {
                "skipped": True,
                "reason": "TOPSTEP_ENABLE_ORDERS=false",
                "action": kind,
            }

        if kind == "enter":
            side = (
                self.SIDE_BID
                if action.get("direction") == "long"
                else self.SIDE_ASK
            )
            size = int(action.get("quantity") or LOTS)
            result = await self.place_order(
                contract_id=contract_id,
                side=side,
                size=size,
                order_type=self.ORDER_TYPE_MARKET,
                custom_tag=f"enter-{uuid.uuid4().hex[:12]}",
            )
            return {"action": "enter", "broker": result}

        # Exit one strategy leg (1 contract) via partial close when possible.
        size = 1
        reason = action.get("reason")
        if reason in ("session_close", "manual_close"):
            result = await self.close_contract(contract_id)
            return {"action": "exit", "broker": result, "mode": "closeContract"}

        result = await self.partial_close_contract(contract_id, size=size)
        if not result.get("success"):
            fallback = await self.close_contract(contract_id)
            return {
                "action": "exit",
                "broker": fallback,
                "mode": "closeContract_fallback",
                "partial": result,
            }
        return {"action": "exit", "broker": result, "mode": "partialClose"}

    async def close_all_positions(self) -> dict:
        """Cancel open orders, then close every open contract position."""
        cancelled = []
        closed = []

        for order in await self.search_open_orders():
            order_id = order.get("id")
            if order_id is None:
                continue
            result = await self.cancel_order(order_id)
            cancelled.append({"orderId": order_id, **result})

        for position in await self.search_open_positions():
            contract_id = position.get("contractId")
            if not contract_id:
                continue
            result = await self.close_contract(contract_id)
            closed.append({
                "contractId": contract_id,
                "size": position.get("size"),
                "type": position.get("type"),
                **result,
            })

        return {
            "success": (
                all(r.get("success", False) for r in closed) if closed else True
            ),
            "accountId": self.account_id,
            "cancelledOrders": cancelled,
            "closedPositions": closed,
        }

    async def close(self):
        await self.client.aclose()


# ============================================================================
# Connection Manager
# ============================================================================

def _broker_signed_size(position: Optional[dict]) -> int:
    """Map Topstep position type/size to signed contracts (long +, short -)."""
    if not position:
        return 0
    size = int(position.get("size") or 0)
    if size <= 0:
        return 0
    ptype = int(position.get("type") or 0)
    if ptype == 2:  # Short
        return -size
    if ptype == 1:  # Long
        return size
    return size


def _flatten_broker_log(result: dict) -> dict:
    """Flatten nested broker payloads for one-line log fields."""
    broker = result.get("broker") or {}
    return {
        "action": result.get("action"),
        "mode": result.get("mode"),
        "skipped": result.get("skipped"),
        "reason": result.get("reason"),
        "success": broker.get("success", result.get("success")),
        "orderId": broker.get("orderId"),
        "errorCode": broker.get("errorCode"),
        "errorMessage": broker.get("errorMessage"),
    }


def list_sessions(sessions_dir: Path = SESSIONS_DIR) -> list[dict]:
    """Load session metadata from manifest.json."""
    manifest_path = sessions_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sessions = []
    for row in manifest.get("sessions", []):
        sessions.append({
            "date": row.get("date_eastern"),
            "file": row.get("file"),
            "records": row.get("records"),
            "session_class": row.get("session_class"),
            "include_in_primary": row.get("include_in_primary", False),
            "low": row.get("low"),
            "high": row.get("high"),
            "note": row.get("note") or "",
        })
    sessions.sort(key=lambda s: s["date"] or "", reverse=True)
    return sessions


def resolve_session_file(
    session_date: str, sessions_dir: Path = SESSIONS_DIR
) -> Path:
    """Resolve a session date to its .npy path."""
    for row in list_sessions(sessions_dir):
        if row["date"] == session_date:
            path = sessions_dir / row["file"]
            if not path.exists():
                raise FileNotFoundError(f"Session file missing: {path}")
            return path
    # Fall back to conventional filename
    path = sessions_dir / f"{session_date}.npy"
    if path.exists():
        return path
    raise FileNotFoundError(f"No session found for {session_date}")


class ConnectionManager:
    """Manages WebSocket connections and strategy execution."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.engine: Optional[KTwoLot60Engine] = None
        self.topstep: Optional[TopstepClient] = None
        self.running = False
        self.test_mode = True
        self.mode = "test"
        self.replay_task: Optional[asyncio.Task] = None
        self.contract_id: Optional[str] = None
        self.market_hub: Optional[TopstepMarketHub] = None
        self.value_area = DevelopingValueArea()
        self._live_queue: Optional[asyncio.Queue] = None
        self.diag: Optional[LiveDiagLog] = None
        self.broker_snapshot: dict = {
            "size": 0,
            "signed_size": 0,
            "average_price": None,
            "contract_id": None,
            "synced_at": None,
            "desync": False,
            "note": None,
        }
        self._last_broker_sync = 0.0
        self.topstep_session: dict = {
            "state": "unknown",  # unknown | ok | lost | reclaiming
            "message": None,
            "reason": None,
            "updated_at": None,
        }
        self._reclaiming_session = False
        self._last_reclaim_attempt = 0.0
        self._last_session_health = 0.0
        # Working broker exit orders for the open live trade
        self.working_exits: dict = {
            "tp1": None,
            "opp": None,
            "stop": None,
        }

    def _clear_working_exits(self):
        self.working_exits = {"tp1": None, "opp": None, "stop": None}

    @staticmethod
    def _order_id_from_result(result: Optional[dict]) -> Optional[int]:
        if not result or not result.get("success"):
            return None
        oid = result.get("orderId")
        if oid is None:
            oid = (result.get("order") or {}).get("id")
        return int(oid) if oid is not None else None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)

    async def _replace_topstep_client(self) -> TopstepClient:
        """Close any prior TopstepClient and create a fresh one."""
        if self.topstep is not None:
            try:
                await self.topstep.close()
            except Exception:
                pass
            self.topstep = None

        self.topstep = TopstepClient(
            user_name=TOPSTEP_USERNAME,
            api_key=TOPSTEP_API_KEY,
            base_url=TOPSTEP_BASE_URL,
            account_id=TOPSTEP_ACCOUNT_ID,
        )
        return self.topstep

    async def start_strategy(
        self,
        mode: str = "test",
        session_date: Optional[str] = None,
        speed: Union[str, float] = 50,
        test_mode: Optional[bool] = None,
    ):
        """Start the strategy engine in test, replay, or live mode."""
        # Backward compatible with older clients that only sent test_mode
        if test_mode is not None and mode == "test":
            mode = "test" if test_mode else "live"

        if mode not in ("test", "replay", "live"):
            raise ValueError(f"Unknown mode: {mode}")
        if mode == "replay" and not session_date:
            raise ValueError("session_date is required for replay mode")

        await self.stop_async()

        self.diag = LiveDiagLog() if mode == "live" else None
        log_path = None
        if self.diag:
            log_path = self.diag.open_day()
            self.diag.info("mode", mode=mode)

        self.engine = KTwoLot60Engine(diag=self.diag)
        self.running = True
        self.mode = mode
        self.test_mode = mode != "live"
        if self.diag and log_path:
            self.diag.info("engine_start", log_file=str(log_path))
            await self.broadcast({
                "type": "status",
                "message": f"Live diag log: {log_path}",
            })

        if mode == "live":
            if not TOPSTEP_USERNAME or not TOPSTEP_API_KEY:
                self.running = False
                raise RuntimeError(
                    "Live mode requires TOPSTEP_USERNAME and TOPSTEP_API_KEY "
                    "in .env"
                )
            client = await self._replace_topstep_client()
            ok = await client.authenticate()
            if not ok:
                self.running = False
                raise RuntimeError("TopstepX authentication failed")

        self.replay_task = asyncio.create_task(
            self._market_data_loop(mode, session_date=session_date, speed=speed)
        )

    async def ensure_topstep(self) -> TopstepClient:
        """Authenticate a Topstep client even when not running live."""
        if self.topstep and self.topstep.token:
            return self.topstep
        if not TOPSTEP_USERNAME or not TOPSTEP_API_KEY:
            raise RuntimeError(
                "TOPSTEP_USERNAME and TOPSTEP_API_KEY are required in .env"
            )
        client = await self._replace_topstep_client()
        ok = await client.authenticate()
        if not ok:
            raise RuntimeError("TopstepX authentication failed")
        return client

    def _engine_status_with_broker(self) -> Optional[dict]:
        if not self.engine:
            return None
        status = self.engine.get_status()
        status["broker"] = dict(self.broker_snapshot)
        status["topstep_session"] = dict(self.topstep_session)
        status["working_exits"] = {
            "tp1": self.working_exits.get("tp1"),
            "opp": self.working_exits.get("opp"),
            "stop": self.working_exits.get("stop"),
        }
        return status

    async def cancel_working_and_open_orders(
        self, reason: str = "cleanup"
    ) -> list[dict]:
        """Cancel tracked exit orders and any other open orders on contract."""
        if not self.topstep:
            return []
        cancelled = []
        try:
            cancelled = await self.topstep.cancel_open_orders(self.contract_id)
        except Exception as e:
            if self.diag:
                self.diag.info("cancel_orders_error", error=str(e), reason=reason)
        self._clear_working_exits()
        if self.diag:
            self.diag.info(
                "cancel_orders",
                reason=reason,
                count=len(cancelled),
            )
        return cancelled

    async def place_live_exit_brackets(
        self, enter_action: dict
    ) -> Optional[dict]:
        """After a filled entry, rest stop + TP1 + runner limits."""
        if not self.topstep or not self.contract_id:
            return None
        if not TOPSTEP_ENABLE_ORDERS:
            return {"skipped": True, "reason": "TOPSTEP_ENABLE_ORDERS=false"}

        legs = enter_action.get("legs") or []

        # Re-anchor targets/stop to actual broker fill price if available
        try:
            pos = await self._broker_position_for_contract()
            if pos and pos.get("averagePrice") is not None and self.engine:
                fill_ticks = price_to_ticks(float(pos["averagePrice"]))
                updated_legs = self.engine.sync_entry_fill_price(fill_ticks)
                if updated_legs:
                    legs = updated_legs
        except Exception:
            pass

        brackets = await self.topstep.place_exit_brackets(
            contract_id=self.contract_id,
            direction=enter_action.get("direction") or "long",
            legs=legs,
        )
        self.working_exits = {
            "tp1": self._order_id_from_result(brackets.get("tp1")),
            "opp": self._order_id_from_result(brackets.get("opp")),
            "stop": self._order_id_from_result(brackets.get("stop")),
        }
        if self.diag:
            self.diag.info(
                "exit_brackets",
                tp1=self.working_exits["tp1"],
                opp=self.working_exits["opp"],
                stop=self.working_exits["stop"],
                prices=brackets.get("prices"),
            )
        await self.broadcast({
            "type": "broker",
            "data": {"action": "brackets", **brackets},
        })

        # Never leave a live position open without a working stop-loss
        if self.working_exits["stop"] is None:
            if self.diag:
                self.diag.info(
                    "stop_placement_failed",
                    stop_result=brackets.get("stop"),
                )
            await self.broadcast({
                "type": "status",
                "message": (
                    "CRITICAL: Stop-loss order failed — flattening position"
                ),
            })
            await self.cancel_working_and_open_orders(
                reason="stop_placement_failed"
            )
            await self.topstep.close_contract(self.contract_id)
            if self.engine and self.engine.position:
                mark = (
                    self.engine.mark_price
                    or self.engine.prev_price
                    or self.engine.position.entry_price
                )
                actions = self.engine._close_all(
                    "stop_placement_failed", mark, datetime.now(EASTERN)
                )
                for act in actions:
                    await self.broadcast({"type": "action", "data": act})
            return brackets

        return brackets

    async def on_single_leg_filled_broker(self, leg_key: str = "tp1"):
        """After one leg closes, shrink stop to 1 lot and drop its order id."""
        self.working_exits[leg_key] = None
        stop_id = self.working_exits.get("stop")
        if not stop_id or not self.topstep or not TOPSTEP_ENABLE_ORDERS:
            return
        try:
            result = await self.topstep.modify_order(stop_id, size=1)
            if self.diag:
                self.diag.info(
                    "stop_resize",
                    orderId=stop_id,
                    size=1,
                    success=result.get("success"),
                    errorCode=result.get("errorCode"),
                )
            await self.broadcast({
                "type": "broker",
                "data": {"action": "stop_resize", "broker": result, "size": 1},
            })
        except Exception as e:
            if self.diag:
                self.diag.info("stop_resize_error", error=str(e))

    async def update_runner_tp(self, price: float) -> dict:
        """Move the runner take-profit (engine + resting limit order)."""
        if not self.engine:
            return {"success": False, "error": "No engine"}
        # Snap to MNQ tick (0.25)
        price = round(float(price) * 4.0) / 4.0
        ok = self.engine.set_runner_target(price_to_ticks(price))
        if not ok:
            return {"success": False, "error": "Runner target not adjustable"}

        broker = None
        opp_id = self.working_exits.get("opp")
        if (
            self.mode == "live"
            and self.topstep
            and opp_id
            and TOPSTEP_ENABLE_ORDERS
        ):
            broker = await self.topstep.modify_order(opp_id, limit_price=price)
            if self.diag:
                self.diag.info(
                    "runner_modify",
                    orderId=opp_id,
                    limitPrice=price,
                    success=broker.get("success"),
                    errorCode=broker.get("errorCode"),
                )

        status = self._engine_status_with_broker()
        await self.broadcast({
            "type": "runner_update",
            "price": price,
            "status": status,
            "broker": broker,
        })
        return {
            "success": True,
            "price": price,
            "status": status,
            "broker": broker,
        }

    def _set_topstep_session(
        self, state: str, message: str = None, reason: str = None
    ):
        self.topstep_session = {
            "state": state,
            "message": message,
            "reason": reason,
            "updated_at": datetime.now(EASTERN).isoformat(),
        }

    async def _broadcast_session(
        self, state: str, message: str, reason: str = None
    ):
        self._set_topstep_session(state, message=message, reason=reason)
        await self.broadcast({
            "type": "session",
            "state": state,
            "message": message,
            "reason": reason,
            "status": self._engine_status_with_broker(),
        })
        if self.diag:
            self.diag.info(
                "topstep_session",
                state=state,
                message=message,
                reason=reason,
            )

    def _looks_like_auth_failure(self, err) -> bool:
        if isinstance(err, TopstepAuthError):
            return True
        text = str(err).lower()
        needles = (
            "401",
            "403",
            "unauthor",
            "not authenticated",
            "invalid token",
            "jwt",
            "expired",
            "login",
            "session",
        )
        return any(n in text for n in needles)

    async def _restart_market_hub(self):
        """Stop/start the SignalR market hub with the current token."""
        if not (
            self.running
            and self.topstep
            and self.contract_id
            and self._live_queue is not None
        ):
            return
        loop = asyncio.get_running_loop()
        if self.market_hub:
            try:
                await asyncio.to_thread(self.market_hub.stop)
            except Exception:
                pass
            self.market_hub = None

        self.market_hub = TopstepMarketHub(
            token_factory=lambda: self.topstep.token,
            contract_id=self.contract_id,
            event_queue=self._live_queue,
            loop=loop,
            hub_url=TOPSTEP_RTC_URL,
        )
        await asyncio.to_thread(self.market_hub.start)

    async def reclaim_topstep_session(self, reason: str) -> bool:
        """Re-login via loginKey and restart the market hub."""
        if self.mode != "live" or not self.topstep:
            return False
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._reclaiming_session:
            return False
        if now - self._last_reclaim_attempt < SESSION_RECLAIM_COOLDOWN_SEC:
            return False

        self._reclaiming_session = True
        self._last_reclaim_attempt = now
        try:
            await self._broadcast_session(
                "lost",
                "Topstep session lost — reclaiming…",
                reason=reason,
            )
            await self._broadcast_session(
                "reclaiming",
                "Re-logging into Topstep…",
                reason=reason,
            )

            ok = await self.topstep.authenticate()
            if not ok:
                await self._broadcast_session(
                    "lost",
                    "Topstep re-login failed — check credentials / try again",
                    reason=reason,
                )
                return False

            await self._restart_market_hub()
            self._last_session_health = loop.time()
            await self._broadcast_session(
                "ok",
                "Topstep session reclaimed",
                reason=reason,
            )
            return True
        except Exception as e:
            await self._broadcast_session(
                "lost",
                f"Topstep reclaim failed: {e}",
                reason=reason,
            )
            return False
        finally:
            self._reclaiming_session = False

    async def handle_hub_lost(self, reason: str):
        """Market hub dropped — restart if auth ok, else full reclaim."""
        if self.mode != "live" or not self.topstep or self._reclaiming_session:
            return
        healthy = await self.topstep.session_healthy()
        self._last_session_health = asyncio.get_running_loop().time()
        if healthy:
            try:
                await self._restart_market_hub()
                await self.broadcast({
                    "type": "status",
                    "message": f"Market hub restarted ({reason})",
                })
                self._set_topstep_session("ok", message="Topstep session ok")
            except Exception as e:
                await self.reclaim_topstep_session(
                    reason=f"hub_restart_failed:{e}"
                )
            return
        await self.reclaim_topstep_session(reason=reason)

    async def ensure_topstep_session(self, reason: str = "health") -> bool:
        """Probe auth health; reclaim automatically if session was stolen."""
        if self.mode != "live" or not self.topstep:
            return True
        if self._reclaiming_session:
            return False

        healthy = await self.topstep.session_healthy()
        self._last_session_health = asyncio.get_running_loop().time()
        if healthy:
            if self.topstep_session.get("state") != "ok":
                self._set_topstep_session("ok", message="Topstep session ok")
            return True

        return await self.reclaim_topstep_session(reason=reason)

    async def sync_broker_position(self, reason: str = "manual") -> dict:
        """Reconcile local engine state against Topstep open positions.

        Detects external flattens / partial closes (e.g. Topstep auto TP) and
        books them on the engine without sending more broker orders.
        """
        client = await self.ensure_topstep()
        positions = await client.search_open_positions()
        contract_id = self.contract_id
        match = None
        if contract_id:
            match = next(
                (p for p in positions if p.get("contractId") == contract_id),
                None,
            )
        elif positions:
            match = positions[0]
            contract_id = match.get("contractId")

        signed = _broker_signed_size(match)
        abs_size = abs(signed)
        ts = datetime.now(EASTERN)
        mark = None
        if self.engine:
            mark = (
                self.engine.mark_price
                if self.engine.mark_price is not None
                else self.engine.prev_price
            )
        if mark is None and match and match.get("averagePrice") is not None:
            mark = price_to_ticks(float(match["averagePrice"]))
        if mark is None:
            mark = 0

        note = "ok"
        desync = False
        reconciled: list[dict] = []
        engine_legs = self.engine.open_leg_count() if self.engine else 0
        engine_dir = (
            self.engine.position.direction
            if self.engine and self.engine.position
            else 0
        )
        engine_signed = engine_dir * engine_legs

        if self.engine and engine_legs > 0 and abs_size == 0:
            # Broker already flat — something else closed us out
            reconciled = self.engine._close_all("external_flat", mark, ts)
            note = "engine_flattened_external"
            desync = True
            # Drop any resting TP/stop orders left behind
            try:
                await self.cancel_working_and_open_orders(
                    reason="external_flat"
                )
            except Exception:
                pass
        elif self.engine and engine_legs > 0 and abs_size > 0:
            same_side = (engine_signed > 0 and signed > 0) or (
                engine_signed < 0 and signed < 0
            )
            if not same_side:
                note = "broker_opposite_side"
                desync = True
            elif abs_size < engine_legs:
                trimmed = engine_legs - abs_size
                reconciled = self.engine._close_n_legs(
                    trimmed, "external_partial", mark, ts
                )
                note = "engine_trimmed_to_broker"
                desync = True
                # Likely TP1 fill at broker — resize stop to remaining size
                if abs_size == 1:
                    await self.on_single_leg_filled_broker("tp1")
            elif abs_size > engine_legs:
                note = "broker_larger_than_engine"
                desync = True
        elif self.engine and engine_legs == 0 and abs_size > 0:
            note = "broker_open_engine_flat"
            desync = True

        self.broker_snapshot = {
            "size": abs_size,
            "signed_size": signed,
            "average_price": (
                float(match["averagePrice"])
                if match and match.get("averagePrice") is not None
                else None
            ),
            "contract_id": contract_id,
            "synced_at": ts.isoformat(),
            "desync": desync,
            "note": note,
            "reason": reason,
        }
        self._last_broker_sync = asyncio.get_running_loop().time()

        if self.diag:
            self.diag.info(
                "broker_sync",
                reason=reason,
                broker_size=abs_size,
                broker_signed=signed,
                engine_legs=self.engine.open_leg_count() if self.engine else 0,
                note=note,
                desync=desync,
            )

        # Broadcast local exits / trade summaries without placing new orders
        for action in reconciled:
            await self.broadcast({"type": "action", "data": action})

        status = self._engine_status_with_broker()
        notify = reason != "poll" or desync or bool(reconciled)
        if status and notify:
            await self.broadcast({
                "type": "sync_result",
                "status": status,
                "reconciled": len(reconciled),
                "message": (
                    f"Broker sync: {note.replace('_', ' ')}"
                    f" · broker {signed:+d} · engine legs "
                    f"{self.engine.open_leg_count() if self.engine else 0}"
                ),
            })

        return {
            "success": True,
            "broker": dict(self.broker_snapshot),
            "reconciled_actions": len(reconciled),
            "status": status,
            "positions": positions,
        }

    async def close_all_positions(self, price: Optional[int] = None) -> dict:
        """Flatten broker positions (live) and/or local engine position."""
        result = {
            "engine": None,
            "broker": None,
        }

        if (
            self.engine
            and self.engine.position
            and self.engine.position.direction != 0
        ):
            fill_price = (
                price
                if price is not None
                else (
                    self.engine.mark_price
                    or self.engine.prev_price
                    or self.engine.position.entry_price
                )
            )
            ts = datetime.now(EASTERN)
            actions = self.engine._close_all("manual_close", fill_price, ts)
            for action in actions:
                await self.broadcast({"type": "action", "data": action})
            result["engine"] = {
                "success": True,
                "actions": actions,
                "status": self._engine_status_with_broker(),
            }
            await self.broadcast({
                "type": "tick",
                "price": fill_price / 4.0,
                "timestamp": ts.isoformat(),
                "status": self._engine_status_with_broker(),
            })

        # Broker flatten only in live mode (replay/test stay local-engine only)
        if self.mode == "live" and TOPSTEP_USERNAME and TOPSTEP_API_KEY:
            try:
                client = await self.ensure_topstep()
                result["broker"] = await client.close_all_positions()
                self._clear_working_exits()
            except Exception as e:
                result["broker"] = {"success": False, "error": str(e)}

        if result["engine"] is None and result["broker"] is None:
            return {"success": True, "message": "No open positions"}

        broker_ok = (
            result["broker"] is None
            or result["broker"].get("success", False)
            or (
                not result["broker"].get("closedPositions")
                and not result["broker"].get("error")
            )
        )
        engine_ok = result["engine"] is None or result["engine"].get(
            "success", False
        )
        result["success"] = broker_ok and engine_ok
        if result.get("engine") and self.engine:
            result["engine"]["status"] = self._engine_status_with_broker()
        return result

    async def _market_data_loop(
        self,
        mode: str,
        session_date: Optional[str] = None,
        speed: Union[str, float] = 50,
    ):
        """Main loop for processing market data."""
        try:
            if mode == "test":
                await self._simulated_data_loop()
            elif mode == "replay":
                await self._replay_data_loop(session_date, speed)
            else:
                await self._live_data_loop()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.broadcast({
                "type": "status",
                "message": f"Data loop failed: {e}",
            })
            self.running = False

    async def _simulated_data_loop(self):
        """Simulated market data for testing."""
        import random

        base_price = 85000  # ~21250.00 in MNQ points (ticks * 0.25)
        self.engine.update_profile(
            poc=base_price,
            vah=base_price + 160,  # >= MIN_POC_DIST (120)
            val=base_price - 160,
            ts=datetime.now(EASTERN),
        )

        price = base_price
        while self.running:
            price += random.randint(-2, 2)

            ts = datetime.now(EASTERN)
            is_rth = RTH_START <= ts.time() < RTH_END

            actions = self.engine.on_tick(price, ts, is_rth)

            for action in actions:
                await self.broadcast({
                    "type": "action",
                    "data": action,
                })

            await self.broadcast({
                "type": "tick",
                "price": price / 4.0,
                "timestamp": ts.isoformat(),
                "status": self.engine.get_status(),
            })

            await asyncio.sleep(0.1)

    async def _replay_data_loop(
        self,
        session_date: str,
        speed: Union[str, float] = 50,
    ):
        """Replay a historical RTH session from output/sessions/*.npy."""
        path = resolve_session_file(session_date)
        data = await asyncio.to_thread(np.load, path)

        prices = np.ascontiguousarray(data["last"], dtype=np.int32)
        stamps = np.ascontiguousarray(data["ts_100ns"], dtype=np.int64)
        sizes = np.ascontiguousarray(data["volume"], dtype=np.int64)
        count = len(prices)
        if count == 0:
            raise RuntimeError(f"Session {session_date} is empty")

        await self.broadcast({
            "type": "status",
            "message": (
                f"Building value-area profile for {session_date} "
                f"({count:,} ticks)…"
            ),
        })

        poc_arr, vah_arr, val_arr = await asyncio.to_thread(
            rth_profile_for_session, prices, sizes, stamps
        )

        speed_key = (
            str(speed).lower()
            if not isinstance(speed, (int, float))
            else None
        )
        if speed_key in REPLAY_SPEEDS:
            speed_factor = REPLAY_SPEEDS[speed_key]
        else:
            try:
                speed_factor = float(speed)
            except (TypeError, ValueError):
                speed_factor = 50.0
            if speed_factor <= 0:
                speed_factor = None  # treat as max

        await self.broadcast({
            "type": "status",
            "message": (
                f"Replaying {session_date} at "
                f"{'max' if speed_factor is None else f'{speed_factor:g}x'} "
                f"speed"
            ),
        })

        loop = asyncio.get_running_loop()
        last_progress_pct = -1
        last_ui_broadcast = 0.0
        ui_interval = 0.05  # ~20 chart updates/sec max
        NS_PER_SEC = 10_000_000

        for i in range(count):
            if not self.running:
                break

            price = int(prices[i])
            ts = from_100ns(int(stamps[i])).astimezone(EASTERN)
            t = ts.time()
            is_rth = RTH_START <= t < RTH_END

            self.engine.update_profile(
                poc=int(poc_arr[i]),
                vah=int(vah_arr[i]),
                val=int(val_arr[i]),
                ts=ts,
            )
            actions = self.engine.on_tick(price, ts, is_rth)

            for action in actions:
                await self.broadcast({"type": "action", "data": action})

            now = loop.time()
            should_push_tick = (
                actions
                or i == 0
                or i == count - 1
                or (now - last_ui_broadcast) >= ui_interval
            )
            if should_push_tick:
                await self.broadcast({
                    "type": "tick",
                    "price": price / 4.0,
                    "timestamp": ts.isoformat(),
                    "status": self.engine.get_status(),
                })
                last_ui_broadcast = now

            pct = int((i + 1) * 100 / count)
            if pct > last_progress_pct:
                await self.broadcast({
                    "type": "replay_progress",
                    "session_date": session_date,
                    "index": i + 1,
                    "total": count,
                    "pct": pct,
                    "timestamp": ts.isoformat(),
                })
                last_progress_pct = pct

            # Pace relative to recorded timestamps
            if speed_factor is not None and i + 1 < count:
                delta_ns = int(stamps[i + 1]) - int(stamps[i])
                if delta_ns > 0:
                    sleep_s = (delta_ns / NS_PER_SEC) / speed_factor
                    # Cap pathological gaps (halts) so replay doesn't stall
                    sleep_s = min(sleep_s, 0.5)
                    if sleep_s > 0.0005:
                        await asyncio.sleep(sleep_s)
                    elif i % 500 == 0:
                        await asyncio.sleep(0)
                elif i % 500 == 0:
                    await asyncio.sleep(0)
            elif i % 2000 == 0:
                await asyncio.sleep(0)

        if self.running:
            await self.broadcast({
                "type": "replay_progress",
                "session_date": session_date,
                "index": count,
                "total": count,
                "pct": 100,
                "done": True,
            })
            await self.broadcast({
                "type": "status",
                "message": f"Replay finished for {session_date}",
            })
        self.running = False

    async def _broker_position_for_contract(self) -> Optional[dict]:
        if not self.topstep or not self.contract_id:
            return None
        positions = await self.topstep.search_open_positions()
        return next(
            (p for p in positions if p.get("contractId") == self.contract_id),
            None,
        )

    async def _broker_signed_for_contract(self) -> int:
        return _broker_signed_size(await self._broker_position_for_contract())

    async def _execute_live_actions(self, actions: list[dict]):
        """Broadcast engine actions and optionally place broker orders."""
        for action in actions:
            await self.broadcast({"type": "action", "data": action})

        if (
            self.mode != "live"
            or not self.topstep
            or not self.contract_id
            or not actions
        ):
            return

        enters = [a for a in actions if a.get("action") == "enter"]
        exits = [a for a in actions if a.get("action") == "exit"]
        fully_flat = any(a.get("action") == "trade_summary" for a in actions)

        try:
            for action in enters:
                # Clear any leftover working orders before a new entry
                await self.cancel_working_and_open_orders(reason="pre_enter")
                result = await self.topstep.execute_strategy_action(
                    action,
                    contract_id=self.contract_id,
                    enable_orders=TOPSTEP_ENABLE_ORDERS,
                )
                if result:
                    await self.broadcast({"type": "broker", "data": result})
                    if self.diag:
                        self.diag.info("broker", **_flatten_broker_log(result))
                    broker = result.get("broker") or {}
                    if broker.get("success") is False:
                        # Roll back engine position so we don't log a fake trade
                        if self.engine:
                            self.engine.abort_unfilled_entry(
                                ts=datetime.now(EASTERN)
                            )
                        await self.broadcast({
                            "type": "status",
                            "message": (
                                "Broker entry rejected — rolled back local "
                                "position"
                            ),
                        })
                    elif not result.get("skipped"):
                        try:
                            await self.place_live_exit_brackets(action)
                        except Exception as bracket_err:
                            if self.diag:
                                self.diag.info(
                                    "exit_brackets_error",
                                    error=str(bracket_err),
                                )
                            await self.broadcast({
                                "type": "status",
                                "message": (
                                    f"Exit brackets failed: {bracket_err}"
                                ),
                            })

            if exits:
                flatten = any(
                    a.get("reason") in (
                        "session_close",
                        "manual_close",
                        "external_flat",
                    )
                    for a in exits
                )
                stop_exit = any(a.get("reason") == "stop" for a in exits)

                if flatten or stop_exit or fully_flat:
                    # Done with the trade (or stopping out) — cancel exits
                    await self.cancel_working_and_open_orders(
                        reason="exit_flatten"
                    )
                    signed = await self._broker_signed_for_contract()
                    if signed != 0:
                        result = await self.topstep.execute_strategy_action(
                            {"action": "exit", "reason": "session_close"},
                            contract_id=self.contract_id,
                            enable_orders=TOPSTEP_ENABLE_ORDERS,
                        )
                        if result:
                            await self.broadcast({
                                "type": "broker",
                                "data": result,
                            })
                            if self.diag:
                                self.diag.info(
                                    "broker", **_flatten_broker_log(result)
                                )
                    await self.cancel_working_and_open_orders(
                        reason="post_flatten"
                    )
                else:
                    # Single leg target exit (TP1 or runner closed first)
                    for action in exits:
                        leg_key = (
                            "opp"
                            if action.get("kind") == "opp"
                            or action.get("leg") == 1
                            else "tp1"
                        )
                        order_id = self.working_exits.get(leg_key)
                        if order_id and TOPSTEP_ENABLE_ORDERS:
                            try:
                                await self.topstep.cancel_order(order_id)
                            except Exception:
                                pass
                            self.working_exits[leg_key] = None

                        signed = await self._broker_signed_for_contract()
                        if abs(signed) >= 2:
                            # Limit may not have filled yet — market out 1 lot
                            result = await self.topstep.execute_strategy_action(
                                action,
                                contract_id=self.contract_id,
                                enable_orders=TOPSTEP_ENABLE_ORDERS,
                            )
                            if result:
                                await self.broadcast({
                                    "type": "broker",
                                    "data": result,
                                })
                                if self.diag:
                                    self.diag.info(
                                        "broker",
                                        **_flatten_broker_log(result),
                                    )
                                broker = result.get("broker") or {}
                                if broker.get("success") is False:
                                    await self.sync_broker_position(
                                        reason="after_failed_exit"
                                    )
                        await self.on_single_leg_filled_broker(leg_key)

                if fully_flat:
                    await self.cancel_working_and_open_orders(
                        reason="trade_done"
                    )

        except Exception as e:
            if self.diag:
                self.diag.info("broker_error", error=str(e))
            await self.broadcast({
                "type": "status",
                "message": f"Broker action error: {e}",
            })
            if self._looks_like_auth_failure(e):
                await self.reclaim_topstep_session(reason=f"broker_action:{e}")
            else:
                try:
                    await self.sync_broker_position(reason="after_broker_error")
                except Exception:
                    pass

    async def _broadcast_live_tick(
        self, price: float, ts: datetime, force: bool = False
    ):
        if not self.engine:
            return
        now = asyncio.get_running_loop().time()
        last = getattr(self, "_last_live_tick_broadcast", 0.0)
        if not force and (now - last) < 0.1:
            return
        self._last_live_tick_broadcast = now
        await self.broadcast({
            "type": "tick",
            "price": price,
            "timestamp": ts.isoformat(),
            "status": self._engine_status_with_broker(),
        })

    async def _process_live_trade(self, payload: dict):
        trade = trade_from_payload(payload)
        if not trade or not self.engine:
            return
        # Ignore live prints until history seeding finishes
        if not self.engine.entries_enabled and not getattr(
            self, "_live_seed_done", False
        ):
            return

        self.value_area.maybe_roll_day(trade.timestamp)
        profile = self.value_area.add(trade.price_ticks, trade.volume)
        if profile:
            poc, vah, val = profile
            self.engine.update_profile(
                poc=poc, vah=vah, val=val, ts=trade.timestamp
            )

        t = trade.timestamp.astimezone(EASTERN).time()
        is_rth = RTH_START <= t < RTH_END
        actions = self.engine.on_tick(trade.price_ticks, trade.timestamp, is_rth)
        await self._execute_live_actions(actions)
        await self._broadcast_live_tick(
            trade.raw_price, trade.timestamp, force=bool(actions)
        )

    async def _seed_live_value_area(self, contract_id: str) -> dict:
        """Rebuild today's RTH VA from History/retrieveBars before trading."""
        now = datetime.now(EASTERN)
        day = now.date()
        rth_open = datetime.combine(day, RTH_START, tzinfo=EASTERN)
        rth_end = datetime.combine(day, RTH_END, tzinfo=EASTERN)
        end = min(now, rth_end)

        if now < rth_open:
            return {
                "seeded": False,
                "reason": "before_rth",
                "bars": 0,
                "message": "Before RTH open — VA will build from live trades",
            }

        await self.broadcast({
            "type": "status",
            "message": (
                f"Seeding value area from {rth_open.strftime('%H:%M')} ET "
                f"→ {end.strftime('%H:%M')} ET (no trades until ready)…"
            ),
        })

        # 1-second bars; chunk to stay under the 20k bar API limit
        chunk = timedelta(seconds=15_000)
        cursor = rth_open
        all_bars: list[dict] = []
        while cursor < end and self.running:
            chunk_end = min(cursor + chunk, end)
            try:
                bars = await self.topstep.retrieve_bars(
                    contract_id=contract_id,
                    start_time=cursor.astimezone(timezone.utc),
                    end_time=chunk_end.astimezone(timezone.utc),
                    unit=1,
                    unit_number=1,
                    limit=20000,
                    live=TOPSTEP_CONTRACT_LIVE,
                    include_partial_bar=True,
                )
            except Exception as e:
                # Fall back to 1-minute bars if second bars are unavailable
                await self.broadcast({
                    "type": "status",
                    "message": (
                        f"1s history failed ({e}); retrying with 1-minute "
                        "bars…"
                    ),
                })
                bars = await self.topstep.retrieve_bars(
                    contract_id=contract_id,
                    start_time=rth_open.astimezone(timezone.utc),
                    end_time=end.astimezone(timezone.utc),
                    unit=2,
                    unit_number=1,
                    limit=20000,
                    live=TOPSTEP_CONTRACT_LIVE,
                    include_partial_bar=True,
                )
                all_bars = bars
                break
            all_bars.extend(bars)
            cursor = chunk_end

        profile, last_close, used = await asyncio.to_thread(
            seed_value_area_from_bars, self.value_area, all_bars
        )

        ts = end
        if profile:
            poc, vah, val = profile
            self.engine.update_profile(poc=poc, vah=vah, val=val, ts=ts)
            if last_close is not None:
                px = price_to_ticks(last_close)
                self.engine.prev_price = px
                self.engine.mark_price = px
                if px > poc:
                    self.engine.origin = 1
                elif px < poc:
                    self.engine.origin = -1
                else:
                    self.engine.origin = 0
            await self._broadcast_live_tick(
                last_close or ticks_to_price(poc), ts, force=True
            )

        return {
            "seeded": used > 0,
            "bars": used,
            "profile": {
                "poc": self.engine.profile.poc / 4.0,
                "vah": self.engine.profile.vah / 4.0,
                "val": self.engine.profile.val / 4.0,
            }
            if self.engine.profile.poc
            else None,
            "message": (
                f"VA seeded from {used:,} bars"
                if used
                else "No history bars returned — VA will build from live trades"
            ),
        }

    async def _process_live_quote(self, payload: dict):
        """UI mark from quotes only — strategy is driven by trades."""
        if not self.engine:
            return
        last = payload.get("lastPrice")
        if last is None:
            bid = payload.get("bestBid")
            ask = payload.get("bestAsk")
            if bid is None or ask is None:
                return
            last = (float(bid) + float(ask)) / 2.0

        ts = datetime.now(EASTERN)
        raw = payload.get("timestamp")
        if raw:
            try:
                ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=EASTERN)
            except ValueError:
                pass

        # Refresh mark_price only; keep prev_price strictly for trade prints
        self.engine.mark_price = price_to_ticks(float(last))
        await self._broadcast_live_tick(float(last), ts)

    async def _live_data_loop(self):
        """Live TopstepX quotes/trades via SignalR + REST order routing."""
        if not self.topstep:
            raise RuntimeError("Topstep client not authenticated")

        contract = await self.topstep.resolve_contract(
            contract_id=TOPSTEP_CONTRACT_ID or None,
            search_text=TOPSTEP_SYMBOL_SEARCH,
            live=TOPSTEP_CONTRACT_LIVE,
        )
        self.contract_id = contract["id"]
        self.value_area = DevelopingValueArea()
        self._live_queue = asyncio.Queue(maxsize=5000)
        self._live_seed_done = False
        self.engine.entries_enabled = False

        await self.broadcast({
            "type": "status",
            "message": (
                f"Live mode on {contract.get('name')} ({self.contract_id}) "
                f"account={self.topstep.account_id} "
                f"orders={'ON' if TOPSTEP_ENABLE_ORDERS else 'OFF'}"
            ),
        })
        if self.diag:
            self.diag.info(
                "live_contract",
                name=contract.get("name"),
                contract=self.contract_id,
                account=self.topstep.account_id,
                orders=TOPSTEP_ENABLE_ORDERS,
            )

        # Rebuild today's VA before connecting trade stream / allowing entries
        try:
            seed = await self._seed_live_value_area(self.contract_id)
            await self.broadcast({
                "type": "status",
                "message": seed.get("message") or "VA seed finished",
            })
            if self.diag:
                self.diag.info(
                    "va_seed",
                    seeded=seed.get("seeded"),
                    bars=seed.get("bars"),
                    reason=seed.get("reason"),
                    poc=(seed.get("profile") or {}).get("poc"),
                    vah=(seed.get("profile") or {}).get("vah"),
                    val=(seed.get("profile") or {}).get("val"),
                    message=seed.get("message"),
                )
            if seed.get("profile"):
                await self.broadcast({
                    "type": "status",
                    "message": (
                        f"Seeded POC {seed['profile']['poc']:.2f} · "
                        f"VAH {seed['profile']['vah']:.2f} · "
                        f"VAL {seed['profile']['val']:.2f}"
                    ),
                })
        except Exception as e:
            if self.diag:
                self.diag.info("va_seed_failed", error=str(e))
            await self.broadcast({
                "type": "status",
                "message": (
                    f"VA seed failed ({e}); continuing with live-only profile"
                ),
            })

        self._live_seed_done = True
        self.engine.entries_enabled = True
        if self.diag:
            self.diag.info(
                "entries_enabled",
                contract=self.contract_id,
                account=self.topstep.account_id if self.topstep else None,
                orders=TOPSTEP_ENABLE_ORDERS,
            )
        await self.broadcast({
            "type": "status",
            "message": "VA ready — entries enabled",
        })

        loop = asyncio.get_running_loop()
        self.market_hub = TopstepMarketHub(
            token_factory=lambda: self.topstep.token,
            contract_id=self.contract_id,
            event_queue=self._live_queue,
            loop=loop,
            hub_url=TOPSTEP_RTC_URL,
        )
        await asyncio.to_thread(self.market_hub.start)
        self._last_session_health = loop.time()
        self._last_broker_sync = 0.0
        await self._broadcast_session(
            "ok", "Topstep session ok", reason="live_start"
        )
        try:
            while self.running:
                try:
                    event = await asyncio.wait_for(
                        self._live_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    event = None

                if event:
                    kind = event[0]
                    if kind == "status":
                        await self.broadcast({
                            "type": "status",
                            "message": event[1],
                        })
                    elif kind == "hub_lost":
                        await self.handle_hub_lost(
                            reason=str(
                                event[1] if len(event) > 1 else "hub_lost"
                            )
                        )
                    elif kind == "trade":
                        await self._process_live_trade(event[2])
                    elif kind == "quote":
                        await self._process_live_quote(event[2])

                # Clock-based RTH close guard in case trade feed goes quiet
                now_et = datetime.now(EASTERN)
                is_rth_now = RTH_START <= now_et.time() < RTH_END
                if (
                    not is_rth_now
                    and self.engine
                    and self.engine.position
                    and self.engine.position.direction != 0
                ):
                    mark = (
                        self.engine.mark_price
                        or self.engine.prev_price
                        or self.engine.position.entry_price
                    )
                    actions = self.engine._close_all(
                        "session_close", mark, now_et
                    )
                    await self._execute_live_actions(actions)
                    await self._broadcast_live_tick(
                        mark / 4.0, now_et, force=True
                    )

                now = loop.time()
                if now - self._last_broker_sync >= BROKER_SYNC_INTERVAL_SEC:
                    try:
                        await self.sync_broker_position(reason="poll")
                    except Exception as e:
                        if self.diag:
                            self.diag.info("broker_sync_error", error=str(e))
                        self._last_broker_sync = now
                        if self._looks_like_auth_failure(e):
                            await self.reclaim_topstep_session(
                                reason=f"broker_sync:{e}"
                            )

                if now - self._last_session_health >= (
                    SESSION_HEALTH_INTERVAL_SEC
                ):
                    await self.ensure_topstep_session(reason="health_poll")
        finally:
            if self.market_hub:
                await asyncio.to_thread(self.market_hub.stop)
                self.market_hub = None

    async def stop_async(self):
        """Stop the active loop and market hub without blocking the loop."""
        if self.diag and self.running:
            self.diag.info("session_stop", mode=self.mode)
        self.running = False

        task = self.replay_task
        self.replay_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        hub = self.market_hub
        self.market_hub = None
        if hub:
            try:
                await asyncio.to_thread(hub.stop)
            except Exception:
                pass

    def stop(self):
        """Synchronous stop helper."""
        if self.diag and self.running:
            self.diag.info("session_stop", mode=self.mode)
        self.running = False
        if self.replay_task and not self.replay_task.done():
            self.replay_task.cancel()
        if self.market_hub:
            try:
                self.market_hub.stop()
            except Exception:
                pass
            self.market_hub = None


manager = ConnectionManager()


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await manager.stop_async()
    if manager.topstep:
        await manager.topstep.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            cmd = data.get("command")

            if cmd == "start":
                mode = data.get("mode")
                if mode is None:
                    mode = "test" if data.get("test_mode", True) else "live"
                try:
                    await manager.start_strategy(
                        mode=mode,
                        session_date=data.get("session_date"),
                        speed=data.get("speed", 50),
                        test_mode=data.get("test_mode"),
                    )
                    label = mode
                    if mode == "replay":
                        label = (
                            f"replay {data.get('session_date')} "
                            f"@ {data.get('speed', 50)}x"
                        )
                    await websocket.send_json({
                        "type": "status",
                        "message": f"Strategy started ({label})",
                    })
                except Exception as e:
                    manager.running = False
                    await websocket.send_json({
                        "type": "status",
                        "message": f"Start failed: {e}",
                    })

            elif cmd == "stop":
                await manager.stop_async()
                await websocket.send_json({
                    "type": "status",
                    "message": "Strategy stopped",
                })

            elif cmd == "update_profile":
                if manager.engine:
                    manager.engine.update_profile(
                        poc=price_to_ticks(float(data["poc"])),
                        vah=price_to_ticks(float(data["vah"])),
                        val=price_to_ticks(float(data["val"])),
                        ts=datetime.fromisoformat(data["timestamp"]),
                    )

            elif cmd == "status":
                if manager.engine:
                    await websocket.send_json({
                        "type": "status",
                        "data": manager._engine_status_with_broker(),
                    })

            elif cmd == "close_all":
                try:
                    result = await manager.close_all_positions()
                    await websocket.send_json({
                        "type": "close_all_result",
                        "data": result,
                        "message": (
                            "All positions closed"
                            if result.get("success")
                            else "Close-all finished with errors"
                        ),
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "status",
                        "message": f"Close all failed: {e}",
                    })

            elif cmd == "sync":
                try:
                    result = await manager.sync_broker_position(reason="manual")
                    await websocket.send_json({
                        "type": "sync_result",
                        "status": result.get("status"),
                        "reconciled": result.get("reconciled_actions", 0),
                        "message": (
                            "Broker sync: "
                            f"{(result.get('broker') or {}).get('note', 'ok')}"
                        ),
                        "data": result,
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "status",
                        "message": f"Broker sync failed: {e}",
                    })

            elif cmd == "update_runner":
                try:
                    price = float(data.get("price"))
                    result = await manager.update_runner_tp(price)
                    await websocket.send_json({
                        "type": "runner_update",
                        "price": result.get("price"),
                        "status": result.get("status"),
                        "broker": result.get("broker"),
                        "message": (
                            f"Runner TP → {result.get('price'):.2f}"
                            if result.get("success")
                            else result.get("error") or "Runner update failed"
                        ),
                        "success": result.get("success"),
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "status",
                        "message": f"Runner update failed: {e}",
                    })

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/sessions")
async def get_sessions():
    """List local historical sessions available for replay."""
    sessions = list_sessions()
    return {
        "success": True,
        "sessions_dir": str(SESSIONS_DIR),
        "count": len(sessions),
        "sessions": sessions,
    }


@app.get("/api/account")
async def get_account():
    """Get TopstepX account info via POST /api/Account/search."""
    try:
        client = await manager.ensure_topstep()
        return await client.get_account()
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/positions")
async def get_positions():
    """List open positions via POST /api/Position/searchOpen."""
    try:
        client = await manager.ensure_topstep()
        positions = await client.search_open_positions()
        return {
            "success": True,
            "positions": positions,
            "accountId": client.account_id,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/positions/sync")
async def sync_positions():
    """Reconcile engine state with Topstep open positions."""
    try:
        return await manager.sync_broker_position(reason="api")
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/positions/close-all")
async def close_all_positions():
    """Flatten all open positions (engine + broker)."""
    try:
        return await manager.close_all_positions()
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/trade/runner-tp")
async def update_runner_tp(payload: RunnerTPRequest):
    """Update runner take-profit price: {\"price\": 26350.25}."""
    try:
        return await manager.update_runner_tp(payload.price)
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)