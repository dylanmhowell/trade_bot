import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const WS_URL = 'ws://localhost:8000/ws';

function App() {
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [testMode, setTestMode] = useState(true);
  const [status, setStatus] = useState(null);
  const [trades, setTrades] = useState([]);
  const [ticks, setTicks] = useState([]);
  const [pnl, setPnl] = useState(0);
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    };

    return () => ws.close();
  }, []);

  const handleMessage = (msg) => {
    switch (msg.type) {
      case 'tick':
        setStatus(msg.status);
        setTicks(prev => [...prev.slice(-100), {
          time: new Date(msg.timestamp).toLocaleTimeString(),
          price: msg.price,
          pnl: msg.status?.unrealized_pnl || 0
        }]);
        break;
      case 'action':
        if (msg.data.action === 'trade_summary') {
          setTrades(prev => [msg.data.data, ...prev]);
          setPnl(prev => prev + msg.data.data.net_pnl);
        }
        break;
      case 'status':
        console.log(msg.message);
        break;
    }
  };

  const startStrategy = () => {
    setRunning(true);
    setTicks([]);
    setTrades([]);
    setPnl(0);
    wsRef.current.send(JSON.stringify({ command: 'start', test_mode: testMode }));
  };

  const stopStrategy = () => {
    setRunning(false);
    wsRef.current.send(JSON.stringify({ command: 'stop' }));
  };

  const updateProfile = (poc, vah, val) => {
    wsRef.current.send(JSON.stringify({
      command: 'update_profile',
      poc, vah, val,
      timestamp: new Date().toISOString()
    }));
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'system-ui, sans-serif', maxWidth: '1200px', margin: '0 auto' }}>
      <h1>K_two_lot_60 Strategy Dashboard</h1>
      
      {/* Connection Status */}
      <div style={{ 
        display: 'inline-block', 
        padding: '8px 16px', 
        borderRadius: '4px',
        background: connected ? '#22c55e' : '#ef4444',
        color: 'white',
        marginBottom: '20px'
      }}>
        {connected ? 'Connected' : 'Disconnected'}
      </div>

      {/* Controls */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px', alignItems: 'center' }}>
        <label>
          <input 
            type="checkbox" 
            checked={testMode} 
            onChange={(e) => setTestMode(e.target.checked)}
            disabled={running}
          />
          Test Mode (Simulated Data)
        </label>
        
        {!running ? (
          <button 
            onClick={startStrategy}
            disabled={!connected}
            style={{
              padding: '10px 20px',
              background: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Start Strategy
          </button>
        ) : (
          <button 
            onClick={stopStrategy}
            style={{
              padding: '10px 20px',
              background: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Stop Strategy
          </button>
        )}
      </div>

      {/* Status Panel */}
      {status && (
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '15px',
          marginBottom: '20px'
        }}>
          <StatusCard 
            title="Position" 
            value={status.in_position ? `${status.direction.toUpperCase()} @ ${status.entry_price.toFixed(2)}` : 'FLAT'} 
            color={status.in_position ? (status.direction === 'long' ? '#22c55e' : '#ef4444') : '#6b7280'}
          />
          <StatusCard 
            title="Unrealized P&L" 
            value={`$${status.unrealized_pnl.toFixed(2)}`} 
            color={status.unrealized_pnl >= 0 ? '#22c55e' : '#ef4444'}
          />
          <StatusCard 
            title="Daily Trades" 
            value={status.daily_trades} 
          />
          <StatusCard 
            title="Total Commission" 
            value={`$${status.total_commission.toFixed(2)}`} 
          />
        </div>
      )}

      {/* Profile Display */}
      {status?.profile && (
        <div style={{ 
          background: '#f3f4f6', 
          padding: '15px', 
          borderRadius: '8px',
          marginBottom: '20px'
        }}>
          <h3>Value Area Profile</h3>
          <div style={{ display: 'flex', gap: '30px' }}>
            <span>POC: <strong>${status.profile.poc.toFixed(2)}</strong></span>
            <span>VAH: <strong>${status.profile.vah.toFixed(2)}</strong></span>
            <span>VAL: <strong>${status.profile.val.toFixed(2)}</strong></span>
          </div>
        </div>
      )}

      {/* Price Chart */}
      <div style={{ height: '300px', marginBottom: '20px' }}>
        <h3>Price & P&L</h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={ticks}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis yAxisId="left" domain={['auto', 'auto']} />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Line yAxisId="left" type="monotone" dataKey="price" stroke="#3b82f6" dot={false} />
            <Line yAxisId="right" type="monotone" dataKey="pnl" stroke="#22c55e" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Trade History */}
      <div>
        <h3>Trade History {trades.length > 0 && `(Total P&L: $${pnl.toFixed(2)})`}</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#e5e7eb' }}>
              <th style={thStyle}>Time</th>
              <th style={thStyle}>Direction</th>
              <th style={thStyle}>Entry</th>
              <th style={thStyle}>Gross P&L</th>
              <th style={thStyle}>Commission</th>
              <th style={thStyle}>Net P&L</th>
              <th style={thStyle}>Exit Details</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #e5e7eb' }}>
                <td style={tdStyle}>{new Date(t.exit_time).toLocaleTimeString()}</td>
                <td style={tdStyle}>{t.direction.toUpperCase()}</td>
                <td style={tdStyle}>{t.entry_price.toFixed(2)}</td>
                <td style={{...tdStyle, color: t.gross_pnl >= 0 ? '#22c55e' : '#ef4444'}}>
                  ${t.gross_pnl.toFixed(2)}
                </td>
                <td style={tdStyle}>${t.commission.toFixed(2)}</td>
                <td style={{...tdStyle, color: t.net_pnl >= 0 ? '#22c55e' : '#ef4444', fontWeight: 'bold'}}>
                  ${t.net_pnl.toFixed(2)}
                </td>
                <td style={tdStyle}>
                  {t.exit_reasons?.map((r, j) => (
                    <span key={j} style={{ marginRight: '10px' }}>
                      {r.kind}: {r.price.toFixed(2)}
                    </span>
                  ))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StatusCard({ title, value, color = '#374151' }) {
  return (
    <div style={{ 
      background: '#f9fafb', 
      padding: '15px', 
      borderRadius: '8px',
      borderLeft: `4px solid ${color}`
    }}>
      <div style={{ fontSize: '12px', color: '#6b7280', textTransform: 'uppercase' }}>{title}</div>
      <div style={{ fontSize: '24px', fontWeight: 'bold', color }}>{value}</div>
    </div>
  );
}

const thStyle = { padding: '10px', textAlign: 'left', fontWeight: '600' };
const tdStyle = { padding: '10px' };

export default App;