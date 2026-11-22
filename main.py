"""FastAPI backend for K_two_lot_60 strategy with Topstep API."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, time as clock, timezone
from enum import Enum
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from zoneinfo import ZoneInfo


# ============================================================================
# Configuration
# ============================================================================

from config import settings

TOPSTEP_USERNAME = settings.topstep_username
TOPSTEP_API_KEY = settings.topstep_api_key
TOPSTEP_BASE_URL = settings.topstep_base_url
TOPSTEP_ACCOUNT_ID = settings.topstep_account_id

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
RTH_END = clock(16, 0)

TICK_CENTS = 50
COMMISSION_CENTS = 50
STOP_SLIP_TICKS = 1


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
    
    def __init__(self):
        self.state = State.FLAT
        self.profile = Profile()
        self.position: Optional[Position] = None
        self.origin = 0           # 1 = above POC, -1 = below POC
        self.prev_price: Optional[int] = None
        
        # Session state
        self.session_date: Optional[str] = None
        self.entry_window_open = False
        self.rth_active = False
        
        # Statistics
        self.daily_trades: list[TradeSummary] = []
        self.total_commission_cents = 0
        
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
            return actions
        
        # Update crossing state
        self._update_crossing_state(price)
        
        # Check for exits first
        if self.position and self.position.direction != 0:
            exit_actions = self._check_exits(price, ts)
            if exit_actions:
                actions.extend(exit_actions)
                if self.state == State.FLAT:
                    return actions  # Position closed
        
        # Check for entry
        if self.state == State.FLAT and self.entry_window_open and not self.position:
            entry_action = self._check_entry(price, ts)
            if entry_action:
                actions.append(entry_action)
        
        self.prev_price = price
        return actions
    
    def _update_session_state(self, ts: datetime, is_rth: bool):
        """Update entry window and RTH state."""
        self.rth_active = is_rth
        
        t = ts.astimezone(EASTERN).time()
        date = ts.astimezone(EASTERN).date().isoformat()
        
        if date != self.session_date:
            self.session_date = date
            self.daily_trades = []
        
        entry_start = clock(ENTRY_HOUR, ENTRY_MINUTE)
        self.entry_window_open = entry_start <= t < ENTRY_END
    
    def _update_crossing_state(self, price: int):
        """Update POC crossing armed state."""
        if self.profile.poc == 0:
            return
            
        # Check for cross through POC
        if self.origin == -1 and price > self.profile.poc:
            # Crossed up through POC - arm short
            self.state = State.ARMED_SHORT
        elif self.origin == 1 and price < self.profile.poc:
            # Crossed down through POC - arm long
            self.state = State.ARMED_LONG
        
        # Update origin
        if price > self.profile.poc:
            self.origin = 1
        elif price < self.profile.poc:
            self.origin = -1
        else:
            self.origin = 0
    
    def _check_entry(self, price: int, ts: datetime) -> Optional[dict]:
        """Check if entry conditions met."""
        if self.profile.vah == 0 or self.profile.val == 0:
            return None
        
        # Check POC distance requirement
        if self.state == State.ARMED_SHORT:
            trigger = self.profile.vah
            # Must be below VAH and cross up through it
            if self.prev_price and self.prev_price < trigger and price >= trigger:
                # Check POC distance
                if abs(self.profile.poc - trigger) >= MIN_POC_DIST:
                    return self._enter_position(-1, price, ts, trigger)
                    
        elif self.state == State.ARMED_LONG:
            trigger = self.profile.val
            # Must be above VAL and cross down through it
            if self.prev_price and self.prev_price > trigger and price <= trigger:
                # Check POC distance
                if abs(self.profile.poc - trigger) >= MIN_POC_DIST:
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
                continue
            
            all_closed = False
        
        if all_closed:
            actions.extend(self._record_trade(ts))
            self.position = None
            self.state = State.FLAT
        
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
        
        actions.extend(self._record_trade(ts))
        self.position = None
        self.state = State.FLAT
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
        
        return [{
            "action": "trade_summary",
            "data": asdict(summary)
        }]
    
    def get_status(self) -> dict:
        """Get current engine status."""
        unrealized = 0
        if self.position and self.prev_price:
            for leg in self.position.legs:
                if leg.open:
                    ch = (self.prev_price - self.position.entry_price) * self.position.direction
                    unrealized += ch * TICK_CENTS
        
        return {
            "state": self.state.value,
            "in_position": self.position is not None and self.position.direction != 0,
            "direction": "long" if self.position and self.position.direction == 1 else 
                        "short" if self.position and self.position.direction == -1 else "flat",
            "entry_price": self.position.entry_price / 4.0 if self.position else 0,
            "unrealized_pnl": unrealized / 100.0,
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

class ConnectionManager:
    """Manages WebSocket connections and strategy execution."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.engine: Optional[KTwoLot60Engine] = None
        self.topstep: Optional[TopstepClient] = None
        self.running = False
        self.test_mode = True
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)
    
    async def start_strategy(self, test_mode: bool = True):
        """Start the strategy engine."""
        self.engine = KTwoLot60Engine()
        self.running = True
        self.test_mode = test_mode

        if not test_mode:
            if not TOPSTEP_USERNAME or not TOPSTEP_API_KEY:
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

        asyncio.create_task(self._market_data_loop(test_mode))

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

        # Always attempt broker flatten when credentials are configured —
        # emergency close should work even if the strategy is in test mode.
        if TOPSTEP_USERNAME and TOPSTEP_API_KEY:
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

    async def _market_data_loop(self, test_mode: bool):
        """Main loop for processing market data."""
        if test_mode:
            await self._simulated_data_loop()
        else:
            await self._live_data_loop()

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
            is_rth = clock(9, 30) <= ts.time() < clock(16, 0)

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

    async def _live_data_loop(self):
        """Live market data from Topstep SignalR hubs (not yet wired)."""
        await self.broadcast({
            "type": "status",
            "message": (
                "Live mode authenticated. Wire rtc.topstepx.com SignalR hubs "
                "for market data; REST order/position endpoints are ready."
            ),
        })
        while self.running:
            try:
                if self.topstep:
                    await self.topstep.validate_session()
            except Exception as e:
                await self.broadcast({
                    "type": "status",
                    "message": f"Session validate failed: {e}",
                })
            await asyncio.sleep(60)

    def stop(self):
        self.running = False


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
                test_mode = data.get("test_mode", True)
                try:
                    await manager.start_strategy(test_mode)
                    await websocket.send_json({
                        "type": "status",
                        "message": (
                            f"Strategy started "
                            f"({'test' if test_mode else 'live'} mode)"
                        ),
                    })
                except Exception as e:
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