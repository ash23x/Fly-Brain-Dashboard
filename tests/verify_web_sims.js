// Runs the browser simulators (web/sim/*.js) in Node through the same protocols the
// Python probes use, and prints the numbers that matter. Compare with
// tests/verify_web_sims.py, which runs the Python reference through the same script.
//   node tests/verify_web_sims.js [compass|mushroom|all]
'use strict';
const fs = require('fs'), path = require('path');
const Flyb = require('../web/sim/flyb.js');
const CompassSim = require('../web/sim/compass_sim.js');
const MushroomSim = require('../web/sim/mushroom_sim.js');
const ROOT = path.join(__dirname, '..', 'web', 'data');

function load(circuit) {
  const buf = fs.readFileSync(path.join(ROOT, circuit, 'circuit.bin'));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const meta = JSON.parse(fs.readFileSync(path.join(ROOT, circuit, 'meta.json'), 'utf8'));
  return { pack: Flyb.parse(ab), meta };
}
const mean = a => a.reduce((s, x) => s + x, 0) / Math.max(1, a.length);
const circ = (a, b) => { let d = (a - b) % 360; if (d > 180) d -= 360; if (d < -180) d += 360; return d; };

function compass(seed) {
  const { pack } = load('compass');
  const sim = new CompassSim(pack, seed);
  const t0 = Date.now();
  const run = (steps, turn) => { const out = []; sim.setTurn(turn); for (let i = 0; i < steps; i++) out.push(sim.step()); return out; };
  run(300, 0);                                  // settle: the ring wakes and re-ignites somewhere
  sim.cue(90, 2.0, 30, 60);
  const hold1 = run(900, 0);                    // 10 s hold
  const h1 = hold1.slice(300).map(f => f.heading), s1 = hold1.slice(300).map(f => f.strength);
  const right = run(300, 1);                    // 3.3 s turn right
  const hold2 = run(600, 0);
  const left = run(300, -1);
  const hold3 = run(600, 0);
  const rateSteps = (frames) => { const r = frames.map(f => f.epg_rate.reduce((s, x) => s + x, 0) / f.epg_rate.length); return mean(r); };
  const pathLen = (fr) => { let t = 0; for (let i = 1; i < fr.length; i++) if (fr[i - 1].strength > 0 && fr[i].strength > 0) t += circ(fr[i].heading, fr[i - 1].heading); return t; };
  const drift = pathLen(hold1.slice(300));
  const dRight = pathLen(right);
  const dLeft = pathLen(left);
  const deaths = [...hold1, ...right, ...hold2, ...left, ...hold3].filter(f => f.strength === 0).length;
  const ms = (Date.now() - t0) / 3000;
  return { seed, cue_error: circ(hold1[120].heading, 90).toFixed(1), hold_drift: drift.toFixed(1), hold_strength: mean(s1).toFixed(2),
           turn_right_deg: dRight.toFixed(0), turn_left_deg: dLeft.toFixed(0), silent_steps: deaths,
           mean_epg_rate: rateSteps(hold1.slice(300)).toFixed(3), reignitions: sim.reignitions, ms_per_step: ms.toFixed(3) };
}

function mushroom(seed) {
  const { pack, meta } = load('mushroom');
  const sim = new MushroomSim(pack, meta, seed);
  const t0 = Date.now();
  let n = 0;
  const run = (steps) => { let out = null; for (let i = 0; i < steps; i++) { out = sim.step(); n++; } return out; };
  const present = (od, steps = 300) => { sim.setOdour(od); const out = run(steps); sim.setOdour(null); run(90); return out; };
  const codes = {};
  for (const od of ['A', 'B', 'C']) {
    sim.setOdour(od); const active = new Set();
    for (let i = 0; i < 300; i++) { const f = sim.step(); n++; if (i >= 150) for (const id of f.spikes) if (sim.kc_idx.includes(id)) active.add(id); }
    codes[od] = active; sim.setOdour(null); run(90);
  }
  const overlap = (a, b) => { let c = 0; for (const x of a) if (b.has(x)) c++; return c / Math.max(1, Math.min(a.size, b.size)); };
  const kcSparsity = present('A').kc_sparsity;
  const naiveA = sim.naive.A, naiveB = present('B').learned.B;
  const pair = (od, kind, times = 3) => { for (let i = 0; i < times; i++) { sim.setOdour(od); run(110); if (kind === 'reward') sim.setReward(1); else sim.setPunish(1); run(225); sim.setReward(0); sim.setPunish(0); sim.setOdour(null); run(90); } };
  pair('A', 'reward');
  const testA = present('A').learned.A;
  const testB1 = present('B').learned.B;
  pair('B', 'punish');
  const testB2 = present('B').learned.B;
  const testA2 = present('A').learned.A;
  const depressed = sim.step().n_depressed;
  const ms = (Date.now() - t0) / n;
  return { seed, kc_active_per_odour: [codes.A.size, codes.B.size, codes.C.size], kc_sparsity: kcSparsity,
           overlap_AB: overlap(codes.A, codes.B).toFixed(2), overlap_AC: overlap(codes.A, codes.C).toFixed(2),
           naive_A: naiveA.toFixed(2), after_sugar_A: testA.toFixed(2), control_B: `${naiveB.toFixed(2)} -> ${testB1.toFixed(2)}`,
           after_shock_B: testB2.toFixed(2), A_retained: testA2.toFixed(2), n_depressed: depressed, ms_per_step: ms.toFixed(2) };
}

const which = process.argv[2] || 'all';
if (which === 'compass' || which === 'all') for (const s of [1, 2, 3]) console.log('compass', JSON.stringify(compass(s)));
if (which === 'mushroom' || which === 'all') for (const s of (process.argv[3] ? process.argv[3].split(",").map(Number) : [1, 2])) console.log("mushroom", JSON.stringify(mushroom(s)));
