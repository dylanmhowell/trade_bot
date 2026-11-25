"""Append-only diagnostic log for live trading sessions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
LIVE_LOG_DIR = Path("output/live")


class LiveDiagLog:
    def __init__(self, log_dir: Path = LIVE_LOG_DIR):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path: Optional[Path] = None
        self._last_state: Optional[str] = None
        self._last_window: Optional[bool] = None
        self._last_near_miss_key: Optional[str] = None
        self._last_near_miss_ts: float = 0.0

    def open_day(self, day: Optional[str] = None) -> Path:
        day = day or datetime.now(EASTERN).date().isoformat()
        self.path = self.log_dir / f"{day}.log"
        self.info(
            "session_start",
            day=day,
            file=str(self.path),
        )
        return self.path

    def _stamp(self, ts: Optional[datetime] = None) -> str:
        when = ts.astimezone(EASTERN) if ts else datetime.now(EASTERN)
        return when.isoformat(timespec="milliseconds")

    def write(self, event: str, ts: Optional[datetime] = None, **fields: Any):
        if self.path is None:
            self.open_day()
        parts = [f"[{self._stamp(ts)}]", event]
        for key, value in fields.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
        line = " ".join(parts)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def info(self, event: str, ts: Optional[datetime] = None, **fields: Any):
        self.write(event, ts=ts, **fields)

    def state_change(
        self,
        new_state: str,
        ts: Optional[datetime] = None,
        **fields: Any,
    ):
        if new_state == self._last_state:
            return
        prev = self._last_state
        self._last_state = new_state
        self.write(
            "state_change",
            ts=ts,
            from_state=prev,
            to_state=new_state,
            **fields,
        )

    def entry_window(self, open_: bool, ts: Optional[datetime] = None, **fields: Any):
        if open_ == self._last_window:
            return
        self._last_window = open_
        self.write(
            "entry_window",
            ts=ts,
            open=open_,
            **fields,
        )

    def near_miss(
        self,
        reason: str,
        ts: Optional[datetime] = None,
        throttle_sec: float = 5.0,
        **fields: Any,
    ):
        """Log a blocked/near entry, throttled so chatty ticks don't flood the file."""
        import time

        key = f"{reason}|{fields.get('side')}|{fields.get('trigger')}"
        now = time.monotonic()
        if key == self._last_near_miss_key and (now - self._last_near_miss_ts) < throttle_sec:
            return
        self._last_near_miss_key = key
        self._last_near_miss_ts = now
        self.write("near_miss", ts=ts, reason=reason, **fields)
