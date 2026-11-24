/** Lightweight Web Audio cues for trade events (no asset files). */

let audioCtx = null;

function getCtx() {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return null;
  if (!audioCtx) audioCtx = new AC();
  if (audioCtx.state === 'suspended') {
    audioCtx.resume().catch(() => {});
  }
  return audioCtx;
}

function tone(freq, durationSec, { type = 'sine', gain = 0.08, when = 0 } = {}) {
  const ctx = getCtx();
  if (!ctx) return;

  const start = ctx.currentTime + when;
  const osc = ctx.createOscillator();
  const amp = ctx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, start);
  amp.gain.setValueAtTime(0.0001, start);
  amp.gain.exponentialRampToValueAtTime(gain, start + 0.015);
  amp.gain.exponentialRampToValueAtTime(0.0001, start + durationSec);
  osc.connect(amp);
  amp.connect(ctx.destination);
  osc.start(start);
  osc.stop(start + durationSec + 0.02);
}

/** New position opened */
export function playEntry() {
  tone(520, 0.09, { type: 'triangle', gain: 0.07 });
  tone(780, 0.12, { type: 'triangle', gain: 0.06, when: 0.08 });
}

/** Stop hit */
export function playStop() {
  tone(220, 0.18, { type: 'sawtooth', gain: 0.05 });
  tone(160, 0.22, { type: 'sawtooth', gain: 0.045, when: 0.1 });
}

/** Take-profit / target hit */
export function playTakeProfit() {
  tone(660, 0.08, { type: 'sine', gain: 0.07 });
  tone(880, 0.1, { type: 'sine', gain: 0.06, when: 0.07 });
  tone(1175, 0.14, { type: 'sine', gain: 0.05, when: 0.15 });
}

/** Flatten / session close */
export function playFlatten() {
  tone(400, 0.1, { type: 'square', gain: 0.04 });
  tone(300, 0.12, { type: 'square', gain: 0.035, when: 0.09 });
}

export function playTradeAction(action) {
  if (!action) return;
  if (action.action === 'enter') {
    playEntry();
    return;
  }
  if (action.action !== 'exit') return;

  const reason = action.reason;
  if (reason === 'stop') {
    playStop();
  } else if (reason === 'target') {
    playTakeProfit();
  } else if (
    reason === 'session_close' ||
    reason === 'manual_close' ||
    reason === 'external_flat' ||
    reason === 'external_partial'
  ) {
    playFlatten();
  }
}
