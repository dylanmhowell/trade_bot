"""FastAPI backend for K_two_lot_60 strategy with Topstep API."""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, time as clock, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from zoneinfo import ZoneInfo

from backtest_va import from_100ns, rth_profile_for_session
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

from config import settings

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

# K_two_lot_60 parameters
STOP_TICKS = 240          # 60 points
LOTS = 2
TP1_TICKS = 80            # 20 points
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


# ============================================================================
# Data Models
# ============================================================================

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
                self._check_entry(price, ts, log_only_block="outside_entry_window")
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

        t = ts.astimezone(EASTERN).time()
        date = ts.astimezone(EASTERN).date().isoformat()

        if date != self.session_date:
            self.session_date = date
            self.daily_trades = []
            self.daily_realized_cents = 0
            self.equity_peak_cents = 0
            self.max_drawdown_cents = 0
            self.total_commission_cents = 0

        entry_start = clock(ENTRY_HOUR, ENTRY_MINUTE)
        self.entry_window_open = entry_start <= t < ENTRY_END
        if self.diag:
            self.diag.entry_window(
                self.entry_window_open,
                ts=ts,
                time_et=t.isoformat(timespec="seconds"),
            )

    def _unrealized_cents(self) -> int:
        if not self.position or self.prev_price is None:
            return 0
        unrealized = 0
        for leg in self.position.legs:
            if leg.open:
                ch = (self.prev_price - self.position.entry_price) * self.position.direction
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

    def _update_crossing_state(self, price: int, ts: Optional[datetime] = None):
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

        if price > self.profile.poc:
            self.origin = 1
        elif price < self.profile.poc:
            self.origin = -1
        else:
            self.origin = 0

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
            if self.prev_price and self.prev_price < trigger and price >= trigger:
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
            if self.prev_price and self.prev_price > trigger and price <= trigger:
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

    def _enter_position(self, direction: int, price: int, ts: datetime,
                       trigger_price: int) -> dict:
        """Create entry action and initialize position."""
        self.position = Position(
            direction=direction,
            entry_price=price,
            entry_time=ts
        )
        self.state = State.IN_POSITION

        # Calculate targets and stops
        stop = price - direction * STOP_TICKS

        # Leg 1: TP1 at 20 points
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
                {"target": self.position.legs[0].target, "stop": stop, "kind": "tp1"},
                {"target": self.position.legs[1].target, "stop": stop, "kind": "opp"}
            ],
            "timestamp": ts.isoformat()
        }
    
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
                    "timestamp": ts.isoformat()
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
                    "timestamp": ts.isoformat()
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

        fill = price - STOP_SLIP_TICKS if pos.direction == 1 else price + STOP_SLIP_TICKS

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
                    "timestamp": ts.isoformat()
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
            exit_reasons=[{
                "leg": i,
                "kind": leg.kind,
                "price": leg.filled_price / 4.0 if leg.filled_price else 0,
                "time": leg.filled_time.isoformat() if leg.filled_time else ""
            } for i, leg in enumerate(pos.legs)]
        )
        
        self.daily_trades.append(summary)
        self.daily_realized_cents += net

        return [{
            "action": "trade_summary",
            "data": asdict(summary)
        }]

    def get_status(self) -> dict:
        """Get current engine status."""
        unrealized = self._unrealized_cents()
        self._mark_equity(unrealized)
        daily_pnl = self.daily_realized_cents + unrealized

        return {
            "state": self.state.value,
            "in_position": self.position is not None and self.position.direction != 0,
            "direction": "long" if self.position and self.position.direction == 1 else
                        "short" if self.position and self.position.direction == -1 else "flat",
            "entry_price": self.position.entry_price / 4.0 if self.position else 0,
            "unrealized_pnl": unrealized / 100.0,
            "daily_pnl": daily_pnl / 100.0,
            "realized_pnl": self.daily_realized_cents / 100.0,
            "max_drawdown": self.max_drawdown_cents / 100.0,
            "entries_enabled": self.entries_enabled,
            "daily_trades": len(self.daily_trades),
            "total_commission": self.total_commission_cents / 100.0,
            "profile": {
                "poc": self.profile.poc / 4.0,
                "vah": self.profile.vah / 4.0,
                "val": self.profile.val / 4.0
            } if self.profile.poc else None
        }


# ============================================================================
# TopstepX / ProjectX Gateway API Client
# Docs: https://gateway.docs.projectx.com/  |  https://api.topstepx.com/swagger/index.html
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
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=30.0)

    def _headers(self) -> dict:
        if not self.token:
            raise RuntimeError("Not authenticated — call authenticate() first")
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "accept": "text/plain",
        }

    async def authenticate(self) -> bool:
        """POST /api/Auth/loginKey — returns a JWT valid ~24h."""
        try:
            response = await self.client.post(
                "/api/Auth/loginKey",
                json={"userName": self.user_name, "apiKey": self.api_key},
            )
            data = response.json()
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
                chosen = tradeable[0] if tradeable else (accounts[0] if accounts else None)
                if chosen:
                    self.account_id = chosen["id"]
                    print(f"Using Topstep account {self.account_id} ({chosen.get('name')})")
                else:
                    print("Auth ok but no accounts found")
                    return False
            return True
        except Exception as e:
            print(f"Auth failed: {e}")
            return False

    async def validate_session(self) -> bool:
        """POST /api/Auth/validate — refresh token before 24h expiry."""
        response = await self.client.post("/api/Auth/validate", headers=self._headers())
        data = response.json()
        if data.get("success") and data.get("newToken"):
            self.token = data["newToken"]
            return True
        return bool(data.get("success"))

    async def search_accounts(self, only_active: bool = True) -> list[dict]:
        """POST /api/Account/search"""
        response = await self.client.post(
            "/api/Account/search",
            headers=self._headers(),
            json={"onlyActiveAccounts": only_active},
        )
        data = response.json()
        if not data.get("success"):
            raise RuntimeError(data.get("errorMessage") or "Account search failed")
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
        response = await self.client.post(
            "/api/Position/searchOpen",
            headers=self._headers(),
            json={"accountId": self.account_id},
        )
        data = response.json()
        if not data.get("success"):
            raise RuntimeError(data.get("errorMessage") or "Position search failed")
        return data.get("positions") or []

    async def search_open_orders(self) -> list[dict]:
        """POST /api/Order/searchOpen"""
        if self.account_id is None:
            raise RuntimeError("account_id is required")
        response = await self.client.post(
            "/api/Order/searchOpen",
            headers=self._headers(),
            json={"accountId": self.account_id},
        )
        data = response.json()
        if not data.get("success"):
            raise RuntimeError(data.get("errorMessage") or "Open order search failed")
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
        response = await self.client.post(
            "/api/Order/place",
            headers=self._headers(),
            json=payload,
        )
        return response.json()

    async def cancel_order(self, order_id: int) -> dict:
        """POST /api/Order/cancel"""
        response = await self.client.post(
            "/api/Order/cancel",
            headers=self._headers(),
            json={"accountId": self.account_id, "orderId": order_id},
        )
        return response.json()

    async def close_contract(self, contract_id: str) -> dict:
        """POST /api/Position/closeContract — flatten one contract."""
        response = await self.client.post(
            "/api/Position/closeContract",
            headers=self._headers(),
            json={"accountId": self.account_id, "contractId": contract_id},
        )
        return response.json()

    async def partial_close_contract(self, contract_id: str, size: int) -> dict:
        """POST /api/Position/partialCloseContract"""
        response = await self.client.post(
            "/api/Position/partialCloseContract",
            headers=self._headers(),
            json={
                "accountId": self.account_id,
                "contractId": contract_id,
                "size": size,
            },
        )
        return response.json()

    async def search_contracts(
        self, search_text: str, live: bool = False
    ) -> list[dict]:
        """POST /api/Contract/search"""
        response = await self.client.post(
            "/api/Contract/search",
            headers=self._headers(),
            json={"searchText": search_text, "live": live},
        )
        data = response.json()
        if not data.get("success"):
            raise RuntimeError(data.get("errorMessage") or "Contract search failed")
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
        response = await self.client.post(
            "/api/History/retrieveBars",
            headers=self._headers(),
            json={
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
        data = response.json()
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
            response = await self.client.post(
                "/api/Contract/searchById",
                headers=self._headers(),
                json={"contractId": contract_id},
            )
            data = response.json()
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
        """Map engine enter/exit actions onto Topstep order/position endpoints."""
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
            # Fall back to opposite-side market order
            direction = action.get("direction")
            # Engine exit actions don't always include direction; infer from side needed
            # Prefer closeContract as last resort
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
            "success": all(r.get("success", False) for r in closed) if closed else True,
            "accountId": self.account_id,
            "cancelledOrders": cancelled,
            "closedPositions": closed,
        }

    async def close(self):
        await self.client.aclose()


# ============================================================================
# Connection Manager
# ============================================================================

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


def resolve_session_file(session_date: str, sessions_dir: Path = SESSIONS_DIR) -> Path:
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

        self.stop()
        await asyncio.sleep(0)  # let prior loop observe running=False

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
                    "Live mode requires TOPSTEP_USERNAME and TOPSTEP_API_KEY in .env"
                )
            self.topstep = TopstepClient(
                user_name=TOPSTEP_USERNAME,
                api_key=TOPSTEP_API_KEY,
                base_url=TOPSTEP_BASE_URL,
                account_id=TOPSTEP_ACCOUNT_ID,
            )
            ok = await self.topstep.authenticate()
            if not ok:
                self.running = False
                raise RuntimeError("TopstepX authentication failed")

        self.replay_task = asyncio.create_task(
            self._market_data_loop(mode, session_date=session_date, speed=speed)
        )

    async def ensure_topstep(self) -> TopstepClient:
        """Authenticate a Topstep client even when the strategy is not running live."""
        if self.topstep and self.topstep.token:
            return self.topstep
        if not TOPSTEP_USERNAME or not TOPSTEP_API_KEY:
            raise RuntimeError(
                "TOPSTEP_USERNAME and TOPSTEP_API_KEY are required in .env"
            )
        self.topstep = TopstepClient(
            user_name=TOPSTEP_USERNAME,
            api_key=TOPSTEP_API_KEY,
            base_url=TOPSTEP_BASE_URL,
            account_id=TOPSTEP_ACCOUNT_ID,
        )
        ok = await self.topstep.authenticate()
        if not ok:
            raise RuntimeError("TopstepX authentication failed")
        return self.topstep

    async def close_all_positions(self, price: Optional[int] = None) -> dict:
        """Flatten broker positions (live) and/or local engine position (sim)."""
        result = {
            "engine": None,
            "broker": None,
        }

        if self.engine and self.engine.position and self.engine.position.direction != 0:
            fill_price = price if price is not None else (
                self.engine.prev_price or self.engine.position.entry_price
            )
            ts = datetime.now(EASTERN)
            actions = self.engine._close_all("manual_close", fill_price, ts)
            for action in actions:
                await self.broadcast({"type": "action", "data": action})
            result["engine"] = {
                "success": True,
                "actions": actions,
                "status": self.engine.get_status(),
            }
            await self.broadcast({
                "type": "tick",
                "price": fill_price / 4.0,
                "timestamp": ts.isoformat(),
                "status": self.engine.get_status(),
            })

        # Broker flatten only in live mode (replay/test stay local-engine only).
        if self.mode == "live" and TOPSTEP_USERNAME and TOPSTEP_API_KEY:
            try:
                client = await self.ensure_topstep()
                result["broker"] = await client.close_all_positions()
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
        engine_ok = result["engine"] is None or result["engine"].get("success", False)
        result["success"] = broker_ok and engine_ok
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
            vah=base_price + 80,
            val=base_price - 80,
            timestamp=datetime.now(EASTERN),
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
                    "data": action
                })

            await self.broadcast({
                "type": "tick",
                "price": price / 4.0,
                "timestamp": ts.isoformat(),
                "status": self.engine.get_status()
            })

            await asyncio.sleep(0.1)

    async def _replay_data_loop(
        self,
        session_date: str,
        speed: Union[str, float] = 50,
    ):
        """Replay a historical RTH session from output/sessions/*.npy."""
        path = resolve_session_file(session_date)
        data = np.load(path)

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

        speed_key = str(speed).lower() if not isinstance(speed, (int, float)) else None
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
                f"{'max' if speed_factor is None else f'{speed_factor:g}x'} speed"
            ),
        })

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

            now = asyncio.get_event_loop().time()
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
                    # Cap pathological gaps (halts) so replay doesn't stall forever
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

        try:
            for action in enters:
                result = await self.topstep.execute_strategy_action(
                    action,
                    contract_id=self.contract_id,
                    enable_orders=TOPSTEP_ENABLE_ORDERS,
                )
                if result:
                    await self.broadcast({"type": "broker", "data": result})
                    if self.diag:
                        self.diag.info("broker", **_flatten_broker_log(result))

            if exits:
                flatten = any(
                    a.get("reason") in ("session_close", "manual_close")
                    for a in exits
                )
                if flatten:
                    result = await self.topstep.execute_strategy_action(
                        {"action": "exit", "reason": "session_close"},
                        contract_id=self.contract_id,
                        enable_orders=TOPSTEP_ENABLE_ORDERS,
                    )
                    if result:
                        await self.broadcast({"type": "broker", "data": result})
                        if self.diag:
                            self.diag.info("broker", **_flatten_broker_log(result))
                else:
                    for action in exits:
                        result = await self.topstep.execute_strategy_action(
                            action,
                            contract_id=self.contract_id,
                            enable_orders=TOPSTEP_ENABLE_ORDERS,
                        )
                        if result:
                            await self.broadcast({"type": "broker", "data": result})
                            if self.diag:
                                self.diag.info("broker", **_flatten_broker_log(result))
        except Exception as e:
            if self.diag:
                self.diag.info("broker_error", error=str(e))
            await self.broadcast({
                "type": "status",
                "message": f"Broker action error: {e}",
            })

    async def _broadcast_live_tick(self, price: float, ts: datetime, force: bool = False):
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
            "status": self.engine.get_status(),
        })

    async def _process_live_trade(self, payload: dict):
        trade = trade_from_payload(payload)
        if not trade or not self.engine:
            return
        # Ignore live prints until history seeding finishes
        if not self.engine.entries_enabled and not getattr(self, "_live_seed_done", False):
            return

        self.value_area.maybe_roll_day(trade.timestamp)
        profile = self.value_area.add(trade.price_ticks, trade.volume)
        if profile:
            poc, vah, val = profile
            self.engine.update_profile(poc=poc, vah=vah, val=val, ts=trade.timestamp)

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
                    "message": f"1s history failed ({e}); retrying with 1-minute bars…",
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
                if px > poc:
                    self.engine.origin = 1
                elif px < poc:
                    self.engine.origin = -1
                else:
                    self.engine.origin = 0
            await self._broadcast_live_tick(last_close or ticks_to_price(poc), ts, force=True)

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

        # Refresh mark for unrealized P&L only; do not drive entry crosses from quotes
        if self.engine.position:
            self.engine.prev_price = price_to_ticks(float(last))
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

        # Rebuild today's VA before connecting the trade stream / allowing entries
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
                "message": f"VA seed failed ({e}); continuing with live-only profile",
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

        last_validate = loop.time()
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
                    elif kind == "trade":
                        await self._process_live_trade(event[2])
                    elif kind == "quote":
                        await self._process_live_quote(event[2])

                now = loop.time()
                if now - last_validate > 60 * 30:
                    try:
                        await self.topstep.validate_session()
                    except Exception as e:
                        await self.broadcast({
                            "type": "status",
                            "message": f"Session validate failed: {e}",
                        })
                    last_validate = now
        finally:
            if self.market_hub:
                await asyncio.to_thread(self.market_hub.stop)
                self.market_hub = None

    def stop(self):
        if self.diag and self.running:
            self.diag.info("session_stop", mode=self.mode)
        self.running = False
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
    manager.stop()
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
                        label = f"replay {data.get('session_date')} @ {data.get('speed', 50)}x"
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
                manager.stop()
                await websocket.send_json({
                    "type": "status",
                    "message": "Strategy stopped"
                })

            elif cmd == "update_profile":
                if manager.engine:
                    manager.engine.update_profile(
                        poc=int(data["poc"] * 4),
                        vah=int(data["vah"] * 4),
                        val=int(data["val"] * 4),
                        timestamp=datetime.fromisoformat(data["timestamp"])
                    )
                    
            elif cmd == "status":
                if manager.engine:
                    await websocket.send_json({
                        "type": "status",
                        "data": manager.engine.get_status()
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
        return {"success": True, "positions": positions, "accountId": client.account_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/positions/close-all")
async def close_all_positions():
    """Flatten all open positions (engine + broker)."""
    try:
        return await manager.close_all_positions()
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)