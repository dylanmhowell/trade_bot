import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { playTradeAction } from './sounds.js';

const WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws`;
const ET = 'America/New_York';

/** Format timestamps in US/Eastern so UI matches strategy rules. */
function formatEtTime(value) {
  const d = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleTimeString('en-US', {
    timeZone: ET,
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  });
}

function App() {
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState('replay');
  const [sessions, setSessions] = useState([]);
  const [sessionDate, setSessionDate] = useState('');
  const [speed, setSpeed] = useState('50');
  const [status, setStatus] = useState(null);
  const [trades, setTrades] = useState([]);
  const [ticks, setTicks] = useState([]);
  const [pnl, setPnl] = useState(0);
  const [toast, setToast] = useState(null);
  const [closing, setClosing] = useState(false);
  const [replayProgress, setReplayProgress] = useState(null);
  const [soundOn, setSoundOn] = useState(true);
  const wsRef = useRef(null);
  const soundOnRef = useRef(true);

  useEffect(() => {
    soundOnRef.current = soundOn;
  }, [soundOn]);

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      setRunning(false);
    };
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    };

    return () => ws.close();
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/api/sessions');
        const data = await res.json();
        if (cancelled || !data.success) return;
        const list = data.sessions || [];
        setSessions(list);
        if (list.length && !sessionDate) {
          const primary = list.find((s) => s.include_in_primary) || list[0];
          setSessionDate(primary.date);
        }
      } catch {
        // Backend may not be up yet; user can retry after starting main.py
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [connected]);

  const showToast = (message, error = false) => {
    setToast({ message, error });
    window.setTimeout(() => setToast(null), 4000);
  };

  const handleMessage = (msg) => {
    switch (msg.type) {
      case 'tick':
        setStatus(msg.status);
        setTicks((prev) => [
          ...prev.slice(-200),
          {
            time: formatEtTime(msg.timestamp),
            price: msg.price,
            poc: msg.status?.profile?.poc ?? null,
            vah: msg.status?.profile?.vah ?? null,
            val: msg.status?.profile?.val ?? null,
          },
        ]);
        break;
      case 'action':
        if (soundOnRef.current) {
          playTradeAction(msg.data);
        }
        if (msg.data.action === 'trade_summary') {
          setTrades((prev) => [msg.data.data, ...prev]);
          setPnl((prev) => prev + msg.data.data.net_pnl);
        }
        break;
      case 'replay_progress':
        setReplayProgress(msg);
        if (msg.done) {
          setRunning(false);
        }
        break;
      case 'close_all_result':
        setClosing(false);
        if (msg.data?.engine?.status) {
          setStatus(msg.data.engine.status);
        }
        showToast(msg.message || 'Close all finished', !msg.data?.success);
        break;
      case 'broker': {
        const broker = msg.data?.broker;
        const skipped = msg.data?.skipped;
        if (skipped) {
          showToast(`Order skipped: ${msg.data?.reason || 'disabled'}`);
        } else if (broker) {
          const ok = broker.success !== false;
          showToast(
            ok
              ? `Broker ${msg.data?.action || 'order'} ok`
              : `Broker error: ${broker.errorMessage || broker.errorCode || 'failed'}`,
            !ok
          );
        }
        break;
      }
      case 'status':
        showToast(msg.message, String(msg.message || '').toLowerCase().includes('fail'));
        if (String(msg.message || '').toLowerCase().includes('start failed')) {
          setRunning(false);
        }
        if (String(msg.message || '').toLowerCase().includes('replay finished')) {
          setRunning(false);
        }
        break;
      default:
        break;
    }
  };

  const send = (payload) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      showToast('WebSocket not connected', true);
      return false;
    }
    wsRef.current.send(JSON.stringify(payload));
    return true;
  };

  const startStrategy = () => {
    if (mode === 'replay' && !sessionDate) {
      showToast('Select a session date to replay', true);
      return;
    }
    setRunning(true);
    setTicks([]);
    setTrades([]);
    setPnl(0);
    setReplayProgress(null);
    setStatus(null);
    send({
      command: 'start',
      mode,
      session_date: mode === 'replay' ? sessionDate : undefined,
      speed: mode === 'replay' ? speed : undefined,
    });
  };

  const stopStrategy = () => {
    setRunning(false);
    send({ command: 'stop' });
  };

  const closeAllPositions = async () => {
    const confirmed = window.confirm(
      'Close all open positions and cancel working orders?'
    );
    if (!confirmed) return;

    setClosing(true);

    try {
      const res = await fetch('/api/positions/close-all', { method: 'POST' });
      const data = await res.json();
      if (data?.engine?.status) {
        setStatus(data.engine.status);
      }
      showToast(
        data.success ? 'All positions closed' : (data.error || 'Close-all finished with errors'),
        !data.success
      );
      setClosing(false);
      return;
    } catch {
      // Fall back to websocket command
    }

    if (!send({ command: 'close_all' })) {
      setClosing(false);
    }
  };

  const inPosition = Boolean(status?.in_position);
  const canStart =
    connected &&
    !running &&
    (mode !== 'replay' || Boolean(sessionDate));

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Dillybar's Value Area Trade Bot</h1>
          <p>
            MNQ value-area · times in Eastern · entries 10:15–15:00 ET · flatten by 16:00 ET
          </p>
        </div>
        <div className={`pill ${connected ? 'online' : ''}`}>
          <span className="pill-dot" />
          {connected ? 'Connected' : 'Disconnected'}
        </div>
      </header>

      {toast && (
        <div className={`toast ${toast.error ? 'error' : ''}`}>{toast.message}</div>
      )}

      {replayProgress && (
        <div className="toast">
          Replaying {replayProgress.session_date} · {replayProgress.pct}%
          {replayProgress.done ? ' · done' : ''}
        </div>
      )}

      <section className="panel controls">
        <div className="mode-block">
          <div className="mode-switch" role="group" aria-label="Strategy mode">
            {[
              { id: 'test', label: 'Test' },
              { id: 'replay', label: 'Replay' },
              { id: 'live', label: 'Live' },
            ].map((opt) => (
              <button
                key={opt.id}
                type="button"
                className={`mode-btn ${mode === opt.id ? 'active' : ''}`}
                disabled={running}
                onClick={() => setMode(opt.id)}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {mode === 'replay' && (
            <div className="replay-fields">
              <label className="field">
                <span>Session</span>
                <select
                  value={sessionDate}
                  onChange={(e) => setSessionDate(e.target.value)}
                  disabled={running || sessions.length === 0}
                >
                  {sessions.length === 0 && (
                    <option value="">No sessions found</option>
                  )}
                  {sessions.map((s) => (
                    <option key={s.date} value={s.date}>
                      {s.date}
                      {s.records ? ` · ${(s.records / 1e6).toFixed(2)}M ticks` : ''}
                      {s.include_in_primary ? '' : ' · secondary'}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Speed</span>
                <select
                  value={speed}
                  onChange={(e) => setSpeed(e.target.value)}
                  disabled={running}
                >
                  <option value="10">10x</option>
                  <option value="50">50x</option>
                  <option value="max">Max</option>
                </select>
              </label>
            </div>
          )}

          {mode === 'test' && (
            <p className="mode-hint">
              Random-walk ticks with a fixed sample value area. Session clock uses Eastern.
            </p>
          )}
          {mode === 'replay' && (
            <p className="mode-hint">
              Replays historical RTH ticks. Chart times and entry rules are Eastern.
            </p>
          )}
          {mode === 'live' && (
            <p className="mode-hint">
              Seeds today&apos;s VA from history, then streams live MNQ. All session rules are Eastern.
            </p>
          )}
        </div>

        {!running ? (
          <button
            className="btn btn-primary"
            onClick={startStrategy}
            disabled={!canStart}
          >
            {mode === 'replay' ? 'Start replay' : 'Start strategy'}
          </button>
        ) : (
          <button className="btn btn-danger" onClick={stopStrategy}>
            Stop
          </button>
        )}

        <button
          className="btn btn-warning"
          onClick={closeAllPositions}
          disabled={!connected || closing}
          title="Flatten positions and cancel open orders"
        >
          {closing ? 'Closing…' : 'Close all positions'}
        </button>

        <button
          type="button"
          className={`btn btn-ghost ${soundOn ? 'active-sound' : ''}`}
          onClick={() => setSoundOn((v) => !v)}
          title="Toggle trade sounds"
        >
          {soundOn ? 'Sound on' : 'Sound off'}
        </button>
      </section>

      {status && (
        <>
          <RuleIndicators status={status} />

          <section className="status-grid">
            <StatusCard
              title="Position"
              value={
                inPosition
                  ? `${status.direction.toUpperCase()} @ ${status.entry_price.toFixed(2)}`
                  : 'FLAT'
              }
              color={
                inPosition
                  ? status.direction === 'long'
                    ? 'var(--green)'
                    : 'var(--red)'
                  : 'var(--text-dim)'
              }
            />
            <StatusCard
              title="Unrealized P&L"
              value={`$${Number(status.unrealized_pnl || 0).toFixed(2)}`}
              color={(status.unrealized_pnl || 0) >= 0 ? 'var(--green)' : 'var(--red)'}
            />
            <StatusCard
              title="Daily P&L"
              value={`$${Number(status.daily_pnl || 0).toFixed(2)}`}
              color={(status.daily_pnl || 0) >= 0 ? 'var(--green)' : 'var(--red)'}
            />
            <StatusCard
              title="Max drawdown"
              value={`-$${Number(status.max_drawdown || 0).toFixed(2)}`}
              color={(status.max_drawdown || 0) > 0 ? 'var(--red)' : 'var(--text-dim)'}
            />
            <StatusCard title="Daily trades" value={status.daily_trades} />
            <StatusCard
              title="Commission"
              value={`$${Number(status.total_commission || 0).toFixed(2)}`}
            />
          </section>
        </>
      )}

      {status?.profile && (
        <section className="panel profile-panel">
          <h3 className="section-title">Value area profile</h3>
          <div className="profile-row">
            <div className="profile-item">
              <span>POC</span>
              <strong>${status.profile.poc.toFixed(2)}</strong>
            </div>
            <div className="profile-item">
              <span>VAH</span>
              <strong>${status.profile.vah.toFixed(2)}</strong>
            </div>
            <div className="profile-item">
              <span>VAL</span>
              <strong>${status.profile.val.toFixed(2)}</strong>
            </div>
          </div>
        </section>
      )}

      <section className="panel chart-panel">
        <h3 className="section-title">Price (Eastern)</h3>
        <div className="chart-wrap">
          {ticks.length === 0 ? (
            <div className="empty">
              {mode === 'replay'
                ? 'Start a session replay to stream historical ticks.'
                : 'Start the strategy to stream ticks.'}
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={ticks}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="time"
                  stroke="#5f7085"
                  tick={{ fill: '#8b9bb0', fontSize: 11 }}
                  minTickGap={24}
                />
                <YAxis
                  domain={['auto', 'auto']}
                  stroke="#5f7085"
                  tick={{ fill: '#8b9bb0', fontSize: 11 }}
                  width={64}
                />
                <Tooltip
                  contentStyle={{
                    background: '#121820',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 10,
                    color: '#e8eef6',
                  }}
                />
                <Legend
                  wrapperStyle={{ color: '#8b9bb0', fontSize: 12 }}
                  iconType="plainline"
                />
                <Line
                  type="monotone"
                  dataKey="price"
                  name="Price"
                  stroke="#5b9cff"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="poc"
                  name="POC"
                  stroke="#f0b429"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="vah"
                  name="VAH"
                  stroke="#3dd68c"
                  strokeWidth={1.5}
                  strokeDasharray="2 4"
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="val"
                  name="VAL"
                  stroke="#ff6b7a"
                  strokeWidth={1.5}
                  strokeDasharray="2 4"
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>

      <section className="panel trades-panel">
        <h3 className="section-title">
          Trade history
          {trades.length > 0 ? ` · total $${pnl.toFixed(2)}` : ''}
        </h3>
        {trades.length === 0 ? (
          <div className="empty">No closed trades yet.</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Direction</th>
                  <th>Entry</th>
                  <th>Gross</th>
                  <th>Commission</th>
                  <th>Net</th>
                  <th>Exit</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((t, i) => (
                  <tr key={`${t.exit_time}-${i}`}>
                    <td>{formatEtTime(t.exit_time)}</td>
                    <td>{t.direction.toUpperCase()}</td>
                    <td>{t.entry_price.toFixed(2)}</td>
                    <td className={t.gross_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}>
                      ${t.gross_pnl.toFixed(2)}
                    </td>
                    <td>${t.commission.toFixed(2)}</td>
                    <td className={t.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}>
                      ${t.net_pnl.toFixed(2)}
                    </td>
                    <td>
                      {t.exit_reasons?.map((r, j) => (
                        <span key={j} style={{ marginRight: 10 }}>
                          {r.kind}: {r.price.toFixed(2)}
                        </span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function StatusCard({ title, value, color = 'var(--text)' }) {
  return (
    <div className="stat-card" style={{ '--card-accent': color }}>
      <div className="stat-label">{title}</div>
      <div className="stat-value" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

function RuleIndicators({ status }) {
  const inPosition = status.in_position;
  const armed = !!status.armed;
  const triggerSide = status.trigger_side;
  const triggerPrice = status.trigger_price;
  const lastPrice = status.last_price;
  const canEnter = !!status.can_enter;

  let armDetail = 'Waiting for POC cross to arm';
  if (inPosition) {
    armDetail = `In ${status.direction} — no new entries`;
  } else if (armed && triggerSide === 'short' && triggerPrice != null) {
    armDetail = `Armed SHORT — fires on cross up through VAH ${triggerPrice.toFixed(2)}`;
  } else if (armed && triggerSide === 'long' && triggerPrice != null) {
    armDetail = `Armed LONG — fires on cross down through VAL ${triggerPrice.toFixed(2)}`;
  } else if (status.state === 'flat' && status.origin === 0) {
    armDetail = 'Flat — establishing origin vs POC';
  }

  const lastVsTrigger =
    armed && lastPrice != null && triggerPrice != null
      ? `Last ${lastPrice.toFixed(2)} · trigger ${triggerPrice.toFixed(2)}`
      : null;

  return (
    <section className="panel rules-panel">
      <div className="rules-header">
        <h3 className="section-title">Rules</h3>
        <span className={`rules-ready ${canEnter ? 'on' : ''}`}>
          {canEnter ? 'Ready to fire' : inPosition ? 'In trade' : 'Not ready'}
        </span>
      </div>
      <div className="rules-grid">
        <RulePill
          label="VA seeded"
          on={!!status.entries_enabled}
          hint="Entries unlock after value area is built"
        />
        <RulePill
          label="Entry window"
          on={!!status.entry_window_open}
          hint="10:15–15:00 ET"
        />
        <RulePill
          label="RTH"
          on={!!status.rth_active}
          hint="09:30–16:00 ET"
        />
        <RulePill
          label={
            armed
              ? triggerSide === 'short'
                ? 'Armed SHORT @ VAH'
                : 'Armed LONG @ VAL'
              : 'Armed'
          }
          on={armed && !inPosition}
          warn={inPosition}
          hint={armDetail}
        />
        <RulePill
          label="POC distance"
          on={status.poc_distance_ok === true}
          muted={status.poc_distance_ok == null}
          hint="VA line must be ≥30 pts from POC"
        />
      </div>
      <p className="rules-detail">
        {armDetail}
        {lastVsTrigger ? ` · ${lastVsTrigger}` : ''}
      </p>
    </section>
  );
}

function RulePill({ label, on, warn = false, muted = false, hint }) {
  const tone = muted ? 'muted' : warn ? 'warn' : on ? 'on' : 'off';
  return (
    <div className={`rule-pill ${tone}`} title={hint}>
      <span className="rule-dot" aria-hidden />
      <span className="rule-label">{label}</span>
    </div>
  );
}

export default App;
