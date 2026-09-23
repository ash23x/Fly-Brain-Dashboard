// worker.js -- runs one fly circuit in a Web Worker and streams frames to the
// page, in exactly the shape the local Python server sends over its WebSocket
// (server/sim_server.py), so the pages do not care which one they are talking to.
//
//   page -> worker:  {type: 'init', mode: 'compass'|'mushroom', base: <URL of data/<circuit>/>}
//                    {type: 'cmd', cmd: {...}}   the same commands the WebSocket accepts
//   worker -> page:  {type: 'progress', text}  {type: 'meta', meta}  {type: 'frame', frame}  {type: 'error', text}
'use strict';
importScripts('flyb.js', 'compass_sim.js', 'mushroom_sim.js');

const FRAME_HZ = 30, SUBSTEPS = 3;
let sim = null, mode = null, frame = 0, timer = null;

async function init(msg) {
  mode = msg.mode;
  const base = msg.base;
  const progress = (text) => postMessage({ type: 'progress', text });
  progress('fetching the wiring…');
  const [meta, buf] = await Promise.all([
    fetch(base + 'meta.json').then(r => { if (!r.ok) throw new Error('meta.json ' + r.status); return r.json(); }),
    fetch(base + 'circuit.bin').then(r => { if (!r.ok) throw new Error('circuit.bin ' + r.status); return r.arrayBuffer(); }),
  ]);
  progress('wiring up ' + meta.n.toLocaleString() + ' neurons…');
  const pack = Flyb.parse(buf);
  const seed = (Math.random() * 2 ** 32) >>> 0;          // every tab gets its own fly
  sim = mode === 'compass' ? new CompassSim(pack, seed) : new MushroomSim(pack, meta, seed);
  meta.step_hz = FRAME_HZ * SUBSTEPS; meta.frame_hz = FRAME_HZ; meta.engine = 'browser';
  postMessage({ type: 'meta', meta });
  run();
}

function apply(cmd) {
  if (!sim || typeof cmd !== 'object' || cmd === null) return;
  const num = (k, lo, hi) => { const v = parseFloat(cmd[k]); return Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : null; };
  if (mode === 'compass') {
    if ('turn' in cmd) { const v = num('turn', -1, 1); if (v !== null) sim.setTurn(v); }
    if ('cue' in cmd) { const v = num('cue', -1e9, 1e9); if (v !== null) sim.cue(((v % 360) + 360) % 360, 2.0, 30, 60); }
  } else {
    if ('odour' in cmd && (cmd.odour === null || cmd.odour === '' || cmd.odour in sim.odours)) sim.setOdour(cmd.odour || null);
    if ('reward' in cmd) { const v = num('reward', 0, 1); if (v !== null) sim.setReward(v); }
    if ('punish' in cmd) { const v = num('punish', 0, 1); if (v !== null) sim.setPunish(v); }
    if (cmd.forget) sim.forget();
  }
  if (cmd.reset) sim.reset();
}

// A drift-corrected loop: each tick runs SUBSTEPS steps and posts one frame. If the
// tab was asleep it does not try to catch up -- the fly just paused.
function run() {
  const period = 1000 / FRAME_HZ;
  let due = performance.now();
  const tick = () => {
    const fired = new Set();
    let out = null;
    for (let k = 0; k < SUBSTEPS; k++) { out = sim.step(); for (const i of out.spikes) fired.add(i); }
    out.spikes = Array.from(fired).sort((a, b) => a - b);
    out.t = Date.now() / 1000;
    if (mode === 'compass') out.reignitions = sim.reignitions;
    else if (frame % 6 === 0) out.mbon_trace = sim.mbonTrace();
    frame++;
    postMessage({ type: 'frame', frame: out });
    due += period;
    const now = performance.now();
    let delay = due - now;
    if (delay < -period) { due = now; delay = 0; }
    timer = setTimeout(tick, Math.max(0, delay));
  };
  tick();
}

onmessage = (e) => {
  const msg = e.data || {};
  if (msg.type === 'init') init(msg).catch(err => postMessage({ type: 'error', text: String(err && err.message || err) }));
  else if (msg.type === 'cmd') apply(msg.cmd);
};
