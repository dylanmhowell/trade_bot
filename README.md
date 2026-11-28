# Dillybar's Value Area Trade Bot

Live / replay / test harness for an MNQ value-area strategy, with a dark React dashboard and optional TopstepX (ProjectX Gateway) order routing.

**Not financial advice.** Paper/sim first. Live orders can lose real money.

## What it does

- Builds a developing **value area** (POC / VAH / VAL) from RTH ticks
- Arms after a POC cross, then enters on a **VAH** (short) or **VAL** (long) cross
- Trades **2 MNQ lots** (champion: ATR5×0.75 + BE@TP1 + runner +25pt + no Friday):
  - **TP1**: `0.75 ×` mean of prior **5** sessions' **ATR(14) on 15m RTH** bars (clamp 15–100 pt). Enter the 5d ATR manually in the UI if history isn't available.
  - **Runner**: opposite VA edge **+25 points** beyond (adjustable in the UI)
  - **Stop**: −60 points (2 lots at entry; resized to 1 and moved to **breakeven** after TP1)
  - **No Friday** entries
- In **Live** mode, rests broker limit TPs + a stop on entry, syncs to Topstep positions, and can reclaim a stolen API session

All session rules and UI times are **US/Eastern**.

| Window | Eastern |
| --- | --- |
| RTH | 09:30–16:00 |
| Entries | 10:15–15:00 |
| Flatten | by 16:00 |

POC distance filter: VA trigger must be ≥ **30 points** from POC.

## Modes

| Mode | Purpose |
| --- | --- |
| **Test** | Random-walk ticks + sample VA (no broker) |
| **Replay** | Historical RTH ticks from `output/sessions/*.npy` |
| **Live** | Topstep quotes/trades + optional real orders |

## Quick start

### 1. Python backend

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install fastapi uvicorn httpx numpy numba pydantic-settings signalrcore python-dotenv
```

Copy env template and fill in Topstep credentials (for Live):

```bash
cp .env.example .env
```

Run the API (serves WebSocket + REST on port **8000**):

```bash
python main.py
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** (Vite proxies `/api` and `/ws` to the backend).

### 3. Topstep Live setup

1. Create an API key in TopstepX settings.
2. Set in `.env`:
   - `TOPSTEP_USERNAME`
   - `TOPSTEP_API_KEY`
3. Practice/sim accounts usually need:
   - `TOPSTEP_CONTRACT_LIVE=false`
4. Optional: pin `TOPSTEP_ACCOUNT_ID` and/or `TOPSTEP_CONTRACT_ID`.
5. `TOPSTEP_ENABLE_ORDERS=true` places broker orders when the strategy fires; set `false` to stream only.

See `.env.example` for the full list.

## Using the dashboard

1. Choose **Test**, **Replay**, or **Live**.
2. **Start strategy** / **Start replay**.
3. Watch **Rules** pills (VA seeded, entry window, armed, broker sync, Topstep session).
4. In a live trade you’ll see Entry / TP1 / Runner TP / Stop on the chart.
5. Drag **Runner TP** to amend the resting limit (and the engine target).
6. **Sync broker** reconciles local state with Topstep open size.
7. **Close all positions** cancels working orders and flattens.

Live diagnostics append to `output/live/YYYY-MM-DD.log`.

## Important Topstep notes

- Topstep allows **one concurrent session**. Logging into the Topstep chart can kick the bot (and the reverse). Prefer watching this UI while the bot runs.
- The bot detects a lost session, plays an alert, and tries to **re-login** (which can kick the chart again).
- Turn off account-level auto take-profit / flatten rules that fight the strategy (e.g. a blanket $120 TP).
- After exits, the bot cancels open orders so leftover brackets don’t fire later.

## Replay data

Replay reads session files under `output/sessions/` (see `manifest.json`). Those artifacts are gitignored; generate them with the convert/backtest scripts in this repo if you need history locally.

## Repo layout

| Path | Role |
| --- | --- |
| `main.py` | FastAPI app, strategy engine, Topstep REST/order routing |
| `topstep_live.py` | SignalR market hub + developing VA |
| `live_log.py` | Live session diagnostic log |
| `config.py` | Settings from `.env` |
| `frontend/` | Vite + React dashboard |
| `sweep.py` / `backtest_va.py` | Offline research / variants |

## Safety

- Keep `.env` private; never commit API keys.
- Start with **Test** / **Replay**, then Live with `TOPSTEP_ENABLE_ORDERS=false`, then enable orders on a practice account.
- You are responsible for risk limits, contract month rolls, and prop-firm rules.

## License

MIT — see [LICENSE](LICENSE).
