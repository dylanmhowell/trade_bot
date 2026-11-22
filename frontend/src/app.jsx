import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws`;

function App() {
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [testMode, setTestMode] = useState(true);
  const [status, setStatus] = useState(null);
  const [trades, setTrades] = useState([]);
  const [ticks, setTicks] = useState([]);
  const [pnl, setPnl] = useState(0);
  const [toast, setToast] = useState(null);
  const [closing, setClosing] = useState(false);
  const wsRef = useRef(null);

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

  const showToast = (message, error = false) => {
    setToast({ message, error });
    window.setTimeout(() => setToast(null), 4000);
  };

  const handleMessage = (msg) => {
    switch (msg.type) {
      case 'tick':
        setStatus(msg.status);
        setTicks((prev) => [
          ...prev.slice(-100),
          {
            time: new Date(msg.timestamp).toLocaleTimeString(),
            price: msg.price,
            pnl: msg.status?.unrealized_pnl || 0,
          },
        ]);
        break;
      case 'action':
        if (msg.data.action === 'trade_summary') {
          setTrades((prev) => [msg.data.data, ...prev]);
          setPnl((prev) => prev + msg.data.data.net_pnl);
        }
        break;
      case 'close_all_result':
        setClosing(false);
        if (msg.data?.engine?.status) {
          setStatus(msg.data.engine.status);
        }
        showToast(msg.message || 'Close all finished', !msg.data?.success);
        break;
      case 'status':
        showToast(msg.message, String(msg.message || '').toLowerCase().includes('fail'));
        if (String(msg.message || '').toLowerCase().includes('start failed')) {
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
    setRunning(true);
    setTicks([]);
    setTrades([]);
    setPnl(0);
    send({ command: 'start', test_mode: testMode });
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

    // Prefer live REST close so it works even if the strategy WS loop is idle.
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

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Dillybar's Value Area Trade Bot</h1>
          <p>MNQ value-area strategy · TopstepX gateway</p>
        </div>
        <div className={`pill ${connected ? 'online' : ''}`}>
          <span className="pill-dot" />
          {connected ? 'Connected' : 'Disconnected'}
        </div>
      </header>

      {toast && (
        <div className={`toast ${toast.error ? 'error' : ''}`}>{toast.message}</div>
      )}

      <section className="panel controls">
        <label className="toggle">
          <input
            type="checkbox"
            checked={testMode}
            onChange={(e) => setTestMode(e.target.checked)}
            disabled={running}
          />
          Test mode (simulated ticks)
        </label>

        {!running ? (
          <button
            className="btn btn-primary"
            onClick={startStrategy}
            disabled={!connected}
          >
            Start strategy
          </button>
        ) : (
          <button className="btn btn-danger" onClick={stopStrategy}>
            Stop strategy
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
      </section>

      {status && (
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
            value={`$${status.unrealized_pnl.toFixed(2)}`}
            color={status.unrealized_pnl >= 0 ? 'var(--green)' : 'var(--red)'}
          />
          <StatusCard title="Daily trades" value={status.daily_trades} />
          <StatusCard
            title="Commission"
            value={`$${status.total_commission.toFixed(2)}`}
          />
        </section>
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
        <h3 className="section-title">Price & P&L</h3>
        <div className="chart-wrap">
          {ticks.length === 0 ? (
            <div className="empty">Start the strategy to stream ticks.</div>
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
                  yAxisId="left"
                  domain={['auto', 'auto']}
                  stroke="#5f7085"
                  tick={{ fill: '#8b9bb0', fontSize: 11 }}
                  width={64}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#5f7085"
                  tick={{ fill: '#8b9bb0', fontSize: 11 }}
                  width={56}
                />
                <Tooltip
                  contentStyle={{
                    background: '#121820',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 10,
                    color: '#e8eef6',
                  }}
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="price"
                  stroke="#5b9cff"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="pnl"
                  stroke="#3dd68c"
                  strokeWidth={2}
                  dot={false}
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
                    <td>{new Date(t.exit_time).toLocaleTimeString()}</td>
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

export default App;
