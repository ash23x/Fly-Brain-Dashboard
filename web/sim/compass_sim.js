// compass_sim.js -- the fly's head-direction circuit (EPG / PEN / PEG / Delta7)
// as a leaky integrate-and-fire network, running in the browser. A step-for-step
// port of server/compass_sim.py; the wiring comes pre-normalised from
// scripts/08_export_web.py, so this file is only the dynamics:
//
//     s_j  <- s_j (1 - 1/tau_s) + spike_j                 synaptic activation
//     E_i  =  sum_j s_j W_ji / sum_j W_ji   (excitatory j)  fraction of input active
//     I_i  =  same over the inhibitory (glutamatergic Delta7) j
//     v_i  <- v_i + (-v_i + gE E_i - gI I_i + ext_i + noise) / tau_m
//     spike when v_i >= threshold_i, reset to 0, 3 refractory steps
//
// A bump forms on the EPG ring and stays with no input at all; gain-modulating
// the PENs on one side of the protocerebral bridge pushes it round: the fly turns.
'use strict';
(function (root) {
  class CompassSim {
    constructor(pack, seed = 1) {
      const c = pack.constants, A = pack.arrays;
      this.n = c.n;
      this.tau_m = c.tau_m; this.tau_s = c.tau_s; this.gE = c.gE; this.gI = c.gI;
      this.epg_drive = c.epg_drive; this.pen_drive = c.pen_drive;
      this.turn_gain = c.turn_gain; this.turn_tau = c.turn_tau; this.noise_std = c.noise_std;
      this.reignite_after = c.reignite_after; this.refractory_steps = c.refractory_steps;
      this.rate_alpha = c.rate_alpha;
      this.rng = root.Flyb.rng(seed);
      this.threshold = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) this.threshold[i] = 1 + c.threshold_jitter * this.rng.normal();

      this.W_E = A.W_E; this.W_I = A.W_I;                     // dense n x n, rows = presynaptic
      this.epg_idx = A.epg_idx; this.pen_idx = A.pen_idx; this.pen_left = A.pen_left; this.pen_right = A.pen_right;
      this.d7_idx = A.d7_idx; this.peg_idx = A.peg_idx;
      this.epg_angle = A.epg_angle;
      this.epg_cos = new Float64Array(this.epg_idx.length); this.epg_sin = new Float64Array(this.epg_idx.length);
      for (let k = 0; k < this.epg_idx.length; k++) { this.epg_cos[k] = Math.cos(this.epg_angle[k]); this.epg_sin[k] = Math.sin(this.epg_angle[k]); }
      this.ext0 = new Float64Array(this.n);                   // the tonic drives, fixed
      for (const i of this.epg_idx) this.ext0[i] += this.epg_drive;
      for (const i of this.pen_idx) this.ext0[i] += this.pen_drive;

      const n = this.n;
      this.v = new Float64Array(n); this.s = new Float64Array(n); this.s_eff = new Float64Array(n);
      this.refractory = new Int32Array(n); this.spikes = new Uint8Array(n); this.rate = new Float64Array(n);
      this.E_in = new Float64Array(n); this.I_in = new Float64Array(n);
      this._cue = new Float64Array(n); this._cue_steps = 0;
      this.turn = 0; this.turn_cmd = 0;
      this._silent_steps = 0; this._reignite_random = true; this.reignitions = 0;
      this.heading = 0; this.strength = 0; this.step_count = 0;
      this.epg_rate = new Float64Array(this.epg_idx.length);
    }

    setTurn(omega) { this.turn_cmd = Math.max(-1, Math.min(1, +omega || 0)); }

    // A landmark: current into the EPGs near angle_deg for a while (the ring neurons' job in the fly).
    cue(angleDeg, strength = 0.8, widthDeg = 30, steps = 60) {
      const a0 = angleDeg * Math.PI / 180, w = widthDeg * Math.PI / 180;
      this._cue.fill(0);
      for (let k = 0; k < this.epg_idx.length; k++) {
        let d = this.epg_angle[k] - a0;
        d = Math.atan2(Math.sin(d), Math.cos(d));
        this._cue[this.epg_idx[k]] = strength * Math.exp(-0.5 * (d / w) * (d / w));
      }
      this._cue_steps = steps | 0;
    }

    reset() {
      this.v.fill(0); this.s.fill(0); this.refractory.fill(0); this.spikes.fill(0); this.rate.fill(0);
      this._cue_steps = 0; this._silent_steps = 0; this._reignite_random = true;
      this.turn = 0; this.turn_cmd = 0; this.step_count = 0;
    }

    step() {
      const n = this.n, s = this.s, s_eff = this.s_eff, v = this.v, refractory = this.refractory, spikes = this.spikes;
      this.turn += (this.turn_cmd - this.turn) / this.turn_tau;
      // Turning = gain-modulating the PENs: a pure sideways push, total ring drive unchanged.
      s_eff.set(s);
      const gl = 1 + this.turn_gain * this.turn, gr = 1 - this.turn_gain * this.turn;
      for (const i of this.pen_left) s_eff[i] *= gl;
      for (const i of this.pen_right) s_eff[i] *= gr;
      const E_in = this.E_in, I_in = this.I_in, W_E = this.W_E, W_I = this.W_I;
      E_in.fill(0); I_in.fill(0);
      for (let j = 0; j < n; j++) {                 // rows = presynaptic j; skip the silent ones
        const se = s_eff[j], sj = s[j];
        if (se === 0 && sj === 0) continue;
        const row = j * n;
        if (se !== 0) for (let i = 0; i < n; i++) E_in[i] += se * W_E[row + i];
        if (sj !== 0) for (let i = 0; i < n; i++) I_in[i] += sj * W_I[row + i];
      }
      const cueOn = this._cue_steps > 0;
      const gE = this.gE, gI = this.gI, ns = this.noise_std, tau_m = this.tau_m, thr = this.threshold;
      const decay = 1 - 1 / this.tau_s, alpha = this.rate_alpha;
      for (let i = 0; i < n; i++) {
        let I = gE * E_in[i] - gI * I_in[i] + this.ext0[i];
        if (cueOn) I += this._cue[i];
        if (ns > 0) I += ns * this.rng.normal();
        let vi = v[i] + (-v[i] + I) / tau_m;
        const refr = refractory[i] > 0;
        if (refr) { vi = 0; refractory[i]--; }
        if (vi < -1) vi = -1;
        let sp = 0;
        if (!refr && vi >= thr[i]) { sp = 1; vi = 0; refractory[i] = this.refractory_steps; }
        v[i] = vi; spikes[i] = sp;
        s[i] = s[i] * decay + sp;
        this.rate[i] += alpha * (sp - this.rate[i]);
      }
      if (cueOn) this._cue_steps--;
      this.step_count++;

      // Readout: the population vector of EPG firing is the needle.
      const r = this.epg_rate;
      let tot = 0, zx = 0, zy = 0;
      for (let k = 0; k < r.length; k++) { const rk = this.rate[this.epg_idx[k]]; r[k] = rk; tot += rk; zx += rk * this.epg_cos[k]; zy += rk * this.epg_sin[k]; }
      if (tot > 0.05) {
        zx /= tot; zy /= tot;
        this.heading = ((Math.atan2(zy, zx) * 180 / Math.PI) % 360 + 360) % 360;
        this.strength = Math.hypot(zx, zy);
        this._silent_steps = 0;
      } else {
        this.strength = 0;
        this._silent_steps++;
        if (this._cue_steps === 0 && this._silent_steps >= this.reignite_after) {
          // After a reset: anywhere, like a fly waking in the dark. After a rare spontaneous
          // collapse: where it was, so the needle flickers instead of jumping.
          const where = this._reignite_random ? this.rng.uniform() * 360 : this.heading;
          this.cue(where, 2.0, 30, 60);
          this._reignite_random = false; this.reignitions++; this._silent_steps = 0;
        }
      }
      const fired = [];
      for (let i = 0; i < n; i++) if (spikes[i]) fired.push(i);
      return { spikes: fired, heading: Math.round(this.heading * 100) / 100, strength: Math.round(this.strength * 1000) / 1000,
               turn: Math.round(this.turn * 1000) / 1000, epg_rate: Array.from(r, x => Math.round(x * 1000) / 1000) };
    }
  }
  root.CompassSim = CompassSim;
})(typeof self !== 'undefined' ? self : globalThis);
if (typeof module !== 'undefined') module.exports = globalThis.CompassSim;
