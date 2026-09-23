// mushroom_sim.js -- the fly's learning centre (mushroom body) as a leaky
// integrate-and-fire network with ONE plastic synapse type, running in the
// browser. A step-for-step port of server/mushroom_sim.py; the wiring comes
// pre-normalised from scripts/08_export_web.py.
//
// Odour -> projection neurons -> a sparse set of Kenyon cells (the APL neuron's
// feedback keeps it sparse) -> output neurons (MBONs), some driving approach,
// some avoidance. Dopamine (PAM = reward, PPL1 = punishment) arriving in a
// compartment while its Kenyon cells are active DEPRESSES those KC->MBON
// synapses. Reward compartments belong to avoidance MBONs, so odour + sugar =
// less avoidance = the fly now likes it. Nothing else changes.
'use strict';
(function (root) {
  class MushroomSim {
    constructor(pack, meta, seed = 1) {
      const c = pack.constants, A = pack.arrays;
      this.n = c.n; this.n_kc = c.n_kc;
      this.tau_m = c.tau_m; this.tau_s = c.tau_s; this.odour_drive = c.odour_drive; this.dan_drive = c.dan_drive;
      this.noise_std = c.noise_std; this.eta = c.eta; this.f_min = c.f_min; this.tau_forget = c.tau_forget;
      this.rate_alpha = c.rate_alpha; this.refractory_steps = c.refractory_steps;
      this.rng = root.Flyb.rng(seed);
      this.threshold = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) this.threshold[i] = 1 + c.threshold_jitter * this.rng.normal();

      // CSR, rows = postsynaptic cell: one row holds one cell's whole input
      this.E = { indptr: A.WT_E_indptr, indices: A.WT_E_indices, data: Float32Array.from(A.WT_E_data) };  // copy: it is modified by learning
      this.I = { indptr: A.WT_I_indptr, indices: A.WT_I_indices, data: A.WT_I_data };
      this.DA = { indptr: A.WT_da_indptr, indices: A.WT_da_indices, data: A.WT_da_data };
      this.gE_vec = A.gE_vec; this.gI_vec = A.gI_vec;
      this.plastic_pos = A.plastic_pos; this.plastic_mbon = A.plastic_mbon; this.plastic_kc = A.plastic_kc; this.base_w = A.base_w;
      this.n_plastic = this.plastic_pos.length;
      this.factor = new Float64Array(this.n_plastic).fill(1);
      this.pam_idx = A.pam_idx; this.ppl1_idx = A.ppl1_idx; this.kc_idx = A.kc_idx; this.mbon_idx = A.mbon_idx;
      this.mbon_approach = A.mbon_approach; this.mbon_avoid = A.mbon_avoid;
      this.odours = {};                                   // name -> the PN indices it drives
      for (const name of Object.keys(meta.odours)) this.odours[name] = A['odour_' + name];
      this.odourGloms = meta.odours;

      const n = this.n;
      this.v = new Float64Array(n); this.s = new Float64Array(n); this.refractory = new Int32Array(n);
      this.spikes = new Uint8Array(n); this.rate = new Float64Array(n);
      this.E_in = new Float64Array(n); this.I_in = new Float64Array(n); this.ext = new Float64Array(n); this.da = new Float64Array(n);
      this.odour = null; this.reward = 0; this.punish = 0; this.step_count = 0;
      this.naive = {}; this.learned = {};
      this._acc = new Float64Array(60); this._accN = 0;   // last 60 preference values while an odour is on
      // per-MBON totals of plastic input weight, for the memory trace
      this._mbonTot = new Float64Array(n);
      for (let k = 0; k < this.n_plastic; k++) this._mbonTot[this.plastic_mbon[k]] += this.base_w[k];
    }

    setOdour(name) {
      if (name !== null && name !== undefined && !(name in this.odours)) throw new Error('unknown odour ' + name);
      const nm = name || null;
      if (nm !== this.odour) this._accN = 0;
      this.odour = nm;
    }
    setReward(x) { this.reward = Math.max(0, Math.min(1, +x || 0)); }
    setPunish(x) { this.punish = Math.max(0, Math.min(1, +x || 0)); }

    // Wipe the memory: every KC->MBON synapse back to its connectome weight.
    forget() {
      this.factor.fill(1);
      for (let k = 0; k < this.n_plastic; k++) this.E.data[this.plastic_pos[k]] = this.base_w[k];
      this.naive = {}; this.learned = {};
      this.rate.fill(0); this._accN = 0;
    }

    reset() {
      this.v.fill(0); this.s.fill(0); this.refractory.fill(0); this.spikes.fill(0); this.rate.fill(0);
      this.odour = null; this.reward = 0; this.punish = 0; this._accN = 0; this.step_count = 0;
    }

    _matvec(M, x, out) {
      const ip = M.indptr, ix = M.indices, d = M.data;
      for (let i = 0, n = this.n; i < n; i++) {
        let acc = 0;
        for (let k = ip[i], e = ip[i + 1]; k < e; k++) acc += d[k] * x[ix[k]];
        out[i] = acc;
      }
    }

    _writePlastic() {
      const data = this.E.data, pos = this.plastic_pos, bw = this.base_w, f = this.factor;
      for (let k = 0; k < this.n_plastic; k++) data[pos[k]] = bw[k] * f[k];
    }

    step() {
      const n = this.n, s = this.s, v = this.v, refractory = this.refractory, spikes = this.spikes, ext = this.ext;
      ext.fill(0);
      if (this.odour !== null) for (const i of this.odours[this.odour]) ext[i] += this.odour_drive;
      if (this.reward > 0) for (const i of this.pam_idx) ext[i] += this.dan_drive * this.reward;
      if (this.punish > 0) for (const i of this.ppl1_idx) ext[i] += this.dan_drive * this.punish;

      this._matvec(this.E, s, this.E_in);
      this._matvec(this.I, s, this.I_in);
      const E_in = this.E_in, I_in = this.I_in, gE = this.gE_vec, gI = this.gI_vec, ns = this.noise_std;
      const tau_m = this.tau_m, thr = this.threshold, decay = 1 - 1 / this.tau_s, alpha = this.rate_alpha;
      for (let i = 0; i < n; i++) {
        let I = gE[i] * E_in[i] - gI[i] * I_in[i] + ext[i];
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
      this.step_count++;

      // The only plastic synapse: KC -> MBON, gated by dopamine landing on that MBON.
      if (this.reward > 0 || this.punish > 0) {
        this._matvec(this.DA, s, this.da);
        const da = this.da, f = this.factor, pk = this.plastic_kc, pm = this.plastic_mbon, eta = this.eta, fmin = this.f_min;
        let any = false;
        for (let k = 0; k < this.n_plastic; k++) {
          const co = s[pk[k]] * da[pm[k]];
          if (co !== 0) { any = true; let fk = f[k] * (1 - eta * co); if (fk < fmin) fk = fmin; f[k] = fk; }
        }
        if (any) this._writePlastic();
      }
      if (this.step_count % 10 === 0) {                 // slow recovery toward the connectome weights
        const f = this.factor, rec = 10 / this.tau_forget;
        let any = false;
        for (let k = 0; k < this.n_plastic; k++) if (f[k] < 0.999) { any = true; f[k] += rec * (1 - f[k]); }
        if (any) this._writePlastic();
      }

      // Readout: what does the fly make of this odour?
      let app = 0, avo = 0;
      for (const i of this.mbon_approach) app += this.rate[i];
      for (const i of this.mbon_avoid) avo += this.rate[i];
      app /= this.mbon_approach.length; avo /= this.mbon_avoid.length;
      const pref = (app - avo) / (app + avo + 1e-6);
      if (this.odour !== null) {
        this._acc[this._accN % 60] = pref; this._accN++;
        if (this._accN >= 150) {                         // skip the onset transient; a reading = mean of the last 60 steps
          let m = 0; for (let k = 0; k < 60; k++) m += this._acc[k];
          const reading = m / 60;
          this.learned[this.odour] = reading;
          if (!(this.odour in this.naive)) this.naive[this.odour] = reading;
        }
      }
      let active = 0; for (const i of this.kc_idx) if (this.rate[i] > 0.02) active++;
      let nDep = 0; for (let k = 0; k < this.n_plastic; k++) if (this.factor[k] < 0.9) nDep++;
      const fired = [];
      for (let i = 0; i < n; i++) if (spikes[i]) fired.push(i);
      const r3 = x => Math.round(x * 1000) / 1000, r4 = x => Math.round(x * 10000) / 10000;
      const naive = {}, learned = {};
      for (const k in this.naive) naive[k] = r3(this.naive[k]);
      for (const k in this.learned) learned[k] = r3(this.learned[k]);
      return { spikes: fired, odour: this.odour, reward: this.reward, punish: this.punish, pref: r3(pref),
               approach: r4(app), avoid: r4(avo), kc_sparsity: r4(active / this.kc_idx.length), n_depressed: nDep, naive, learned };
    }

    // Per MBON: the fraction of its Kenyon-cell input weight lost to depression (0 = untouched).
    mbonTrace() {
      const lost = new Float64Array(this.n);
      for (let k = 0; k < this.n_plastic; k++) lost[this.plastic_mbon[k]] += this.base_w[k] * (1 - this.factor[k]);
      const out = new Array(this.mbon_idx.length);
      for (let j = 0; j < this.mbon_idx.length; j++) {
        const i = this.mbon_idx[j], tot = this._mbonTot[i];
        out[j] = tot > 0 ? Math.round(lost[i] / Math.max(tot, 1e-9) * 10000) / 10000 : 0;
      }
      return out;
    }
  }
  root.MushroomSim = MushroomSim;
})(typeof self !== 'undefined' ? self : globalThis);
if (typeof module !== 'undefined') module.exports = globalThis.MushroomSim;
