"""Dump the developing profile at fixed times for one session."""
import numpy as np
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from sweep import (
    BASE, SIZE, EASTERN, from_100ns, price_text,
    rth_profile_for_session, eastern_day_id,
)

sessions_dir = Path("output/sessions")
manifest = __import__("json").loads(
    (sessions_dir / "manifest.json").read_text()
)
row = next(
    r for r in manifest["sessions"]
    if r["date_eastern"] == "2026-08-13"
)
data = np.load(sessions_dir / row["file"])
prices = np.ascontiguousarray(data["last"], dtype=np.int32)
sizes = np.ascontiguousarray(data["volume"], dtype=np.int64)
stamps = np.ascontiguousarray(data["ts_100ns"], dtype=np.int64)

poc, vah, val = rth_profile_for_session(prices, sizes, stamps)

targets = [(9, 45), (10, 15), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (15, 59)]
seen = set()
for i, s in enumerate(stamps):
    et = from_100ns(s).astimezone(EASTERN)
    key = (et.hour, et.minute)
    if key in targets and key not in seen:
        seen.add(key)
        print(
            f"{et.strftime('%H:%M')}  price={price_text(int(prices[i])):>9}"
            f"  POC={price_text(int(poc[i])):>9}"
            f"  VAH={price_text(int(vah[i])):>9}"
            f"  VAL={price_text(int(val[i])):>9}"
            f"  width_pts={(int(vah[i]) - int(val[i])) / 4:6.1f}"
        )