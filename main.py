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

TOPSTEP_API_KEY = settings.topstep_api_key
TOPSTEP_API_SECRET = settings.topstep_api_secret

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
# Topstep API Client
# ============================================================================

class TopstepClient:
    """Client for Topstep API."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = httpx.AsyncClient(base_url=TOPSTEP_BASE_URL)
        self.ws: Optional[any] = None  # WebSocket connection
        
    async def authenticate(self) -> bool:
        """Authenticate with Topstep API."""
        # Implement based on Topstep's actual auth flow
        # Likely OAuth2 or API key + signature
        try:
            response = await self.client.post(
                "/auth/token",
                json={"api_key": self.api_key, "api_secret": self.api_secret}
            )
            data = response.json()
            self.access_token = data.get("access_token")
            return True
        except Exception as e:
            print(f"Auth failed: {e}")
            return False
    
    async def get_account(self) -> dict:
        """Get account information."""
        response = await self.client.get(
            "/account",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        return response.json()
    
    async def place_order(self, symbol: str, side: str, quantity: int, 
                         order_type: str = "market", price: Optional[float] = None) -> dict:
        """Place an order."""
        payload = {
            "symbol": symbol,
            "side": side,  # "buy" or "sell"
            "quantity": quantity,
            "type": order_type,
            "price": price
        }
        response = await self.client.post(
            "/orders",
            headers={"Authorization": f"Bearer {self.access_token}"},
            json=payload
        )
        return response.json()
    
    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an order."""
        response = await self.client.delete(
            f"/orders/{order_id}",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        return response.json()
    
    async def connect_market_data(self, callback):
        """Connect to market data WebSocket."""
        # Implement WebSocket connection for real-time data
        pass
    
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
        
        if not test_mode:
            self.topstep = TopstepClient(TOPSTEP_API_KEY, TOPSTEP_API_SECRET)
            await self.topstep.authenticate()
        
        # Start market data loop
        asyncio.create_task(self._market_data_loop(test_mode))
        
    async def _market_data_loop(self, test_mode: bool):
        """Main loop for processing market data."""
        if test_mode:
            await self._simulated_data_loop()
        else:
            await self._live_data_loop()
    
    async def _simulated_data_loop(self):
        """Simulated market data for testing."""
        import random
        
        # Initialize with sample profile
        base_price = 85000  # ~212.50 in MNQ ticks
        self.engine.update_profile(
            poc=base_price,
            vah=base_price + 80,   # +20 points
            val=base_price - 80,   # -20 points
            timestamp=datetime.now(EASTERN)
        )
        
        price = base_price
        while self.running:
            # Simulate random walk
            price += random.randint(-2, 2)
            
            ts = datetime.now(EASTERN)
            is_rth = clock(9, 30) <= ts.time() < clock(16, 0)
            
            actions = self.engine.on_tick(price, ts, is_rth)
            
            for action in actions:
                await self.broadcast({
                    "type": "action",
                    "data": action
                })
            
            # Broadcast status every tick
            await self.broadcast({
                "type": "tick",
                "price": price / 4.0,
                "timestamp": ts.isoformat(),
                "status": self.engine.get_status()
            })
            
            await asyncio.sleep(0.1)  # 10 ticks per second
    
    async def _live_data_loop(self):
        """Live market data from Topstep."""
        # Implement actual WebSocket connection to Topstep
        pass
    
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
                await manager.start_strategy(test_mode)
                await websocket.send_json({
                    "type": "status",
                    "message": f"Strategy started ({'test' if test_mode else 'live'} mode)"
                })
                
            elif cmd == "stop":
                manager.stop()
                await websocket.send_json({
                    "type": "status",
                    "message": "Strategy stopped"
                })
                
            elif cmd == "update_profile":
                # Manual profile update from UI
                if manager.engine:
                    manager.engine.update_profile(
                        poc=int(data["poc"] * 4),  # Convert points to ticks
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
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/account")
async def get_account():
    """Get Topstep account info."""
    if manager.topstep:
        return await manager.topstep.get_account()
    return {"error": "Not connected"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)