"""
Spiking simulation of the fly's learning centre (mushroom body), built by
scripts/05_build_mushroom_body.py from the MaleCNS connectome.

Odour  -> a set of olfactory glomeruli -> their projection neurons fire
       -> Kenyon cells that get several of those PNs fire (a sparse code;
          the APL neuron's feedback inhibition keeps it sparse)
       -> output neurons (MBONs) sum the Kenyon cells: some drive approach,
          some drive avoidance. Their balance is the fly's opinion.

Learning: dopamine neurons (PAM = reward, PPL1 = punishment) each cover a
compartment of the mushroom body. When dopamine arrives in a compartment
while the Kenyon cells feeding it are active, those KC->MBON synapses are
DEPRESSED. Reward compartments belong to avoidance-driving MBONs, so
odour + reward = less avoidance = approach. Nothing else in the model
changes; this is the only plastic synapse, as in the animal.

Dynamics are the same current-based LIF as compass_sim.py, with one gain
per postsynaptic population (the normalised input to an MBON from 5 % of
4,000 Kenyon cells is tiny, so it needs a bigger gain than a PN does).
Dopamine synapses carry no fast current -- they are the teaching signal.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import sparse

ROLES = ("pn", "kc", "apl", "dpm", "mbon", "pam", "ppl1")


class MushroomSim:
    def __init__(self, mb_dir: pathlib.Path, seed: int | None = 0, **params):
        W_raw = sparse.load_npz(mb_dir / "adjacency.npz").tocsr().astype(np.float64)
        with open(mb_dir / "neurons.json") as f:
            self.neurons: list[dict] = json.load(f)
        with open(mb_dir / "glomeruli.json") as f:
            self.glomeruli: list[str] = json.load(f)
        self.n = W_raw.shape[0]
        self.rng = np.random.default_rng(seed)

        # ---- tunables ---------------------------------------------------------
        self.tau_m = params.get("tau_m", 10.0)
        self.tau_s = params.get("tau_s", 6.0)
        self.gE = {"pn": 0.0, "kc": params.get("gE_kc", 4.0), "apl": params.get("gE_apl", 30.0),
                   "dpm": 2.0, "mbon": params.get("gE_mbon", 80.0), "pam": 1.0, "ppl1": 1.0}
        self.gI = {"pn": 0.0, "kc": params.get("gI_kc", 16.0), "apl": 0.0, "dpm": 0.0,
                   "mbon": params.get("gI_mbon", 4.0), "pam": 0.0, "ppl1": 0.0}
        self.odour_drive = params.get("odour_drive", 1.6)      # current into PNs of an active glomerulus
        self.odour_fraction = params.get("odour_fraction", 0.25)  # share of glomeruli per odour
        self.dan_drive = params.get("dan_drive", 1.6)          # current into PAM (reward) or PPL1 (punish)
        self.noise_std = params.get("noise_std", 0.05)
        self.threshold_jitter = params.get("threshold_jitter", 0.2)
        self.eta = params.get("eta", 0.02)                     # depression rate per step at full KC x DA coincidence
        self.f_min = params.get("f_min", 0.05)                 # floor for a depressed synapse (fraction of baseline)
        self.tau_forget = params.get("tau_forget", 30000.0)    # steps for a depressed synapse to recover (~5 min at 90/s)
        self.refractory_steps = 3
        self.rate_alpha = params.get("rate_alpha", 0.05)
        self.threshold = 1.0 + self.threshold_jitter * self.rng.standard_normal(self.n)

        # ---- populations --------------------------------------------------------
        self.role = np.array([m["role"] for m in self.neurons])
        self.idx = {r: np.where(self.role == r)[0] for r in ROLES}
        self.sign = np.array([m.get("sign", 1) for m in self.neurons], dtype=np.float64)
        self.n_inhibitory = int((self.sign < 0).sum())
        glom = np.array([m.get("glom", "") for m in self.neurons])
        self.pn_by_glom = {g: np.where((self.role == "pn") & (glom == g))[0] for g in self.glomeruli}
        val = np.array([m.get("valence", "") for m in self.neurons])
        self.mbon_approach = np.where(val == "approach")[0]
        self.mbon_avoid = np.where(val == "avoid")[0]
        is_dan = np.isin(self.role, ["pam", "ppl1"])

        # ---- weights ----------------------------------------------------------
        # Glomeruli differ 30x in how many synapses their PNs make on Kenyon
        # cells (this fly, this reconstruction). Scale each PN's output so
        # every glomerulus delivers the same total, or an odour's strength is
        # an accident of which glomeruli it happened to include.
        self.nnz = int(W_raw.nnz)          # the anatomical count, before any modelling choices below
        kc_cols = self.idx["kc"]
        g_tot = {g: float(W_raw[pns][:, kc_cols].sum()) for g, pns in self.pn_by_glom.items()}
        g_med = float(np.median(list(g_tot.values())))
        out_scale = np.ones(self.n)
        for g, pns in self.pn_by_glom.items():
            if g_tot[g] > 0:
                out_scale[pns] = g_med / g_tot[g]
        W_raw = W_raw.multiply(out_scale[:, None]).tocsr()

        # KC->KC contacts are 58 % of a Kenyon cell's synapses by count -- they
        # are axo-axonal, in the lobes, and not what drives a KC to fire. Kept
        # out of the fast current (scaled by kc_kc_current, default 0).
        kc_kc = params.get("kc_kc_current", 0.0)
        is_kc = self.role == "kc"
        if kc_kc != 1.0:
            coo = W_raw.tocoo()
            m = is_kc[coo.row] & is_kc[coo.col]
            coo.data[m] *= kc_kc
            W_raw = coo.tocsr()
            W_raw.eliminate_zeros()

        # Dopamine synapses are modulatory: no fast current, kept aside as the
        # teaching signal (DAN -> MBON weights define each MBON's compartment).
        W_fast = W_raw.multiply((~is_dan)[:, None].astype(np.float64)).tocsr()
        exc = (self.sign > 0) & ~is_dan
        W_E = W_fast.multiply(exc[:, None].astype(np.float64)).tocsr()
        W_I = W_fast.multiply((~exc & ~is_dan)[:, None].astype(np.float64)).tocsr()
        colE = np.asarray(W_E.sum(axis=0)).ravel()
        colI = np.asarray(W_I.sum(axis=0)).ravel()
        self.W_E = W_E.multiply(1.0 / np.maximum(colE, 1e-9)[None, :]).tocsr()
        self.W_I = W_I.multiply(1.0 / np.maximum(colI, 1e-9)[None, :]).tocsr()
        self.W_E.eliminate_zeros(); self.W_I.eliminate_zeros()
        # Store the TRANSPOSE as CSR (rows = postsynaptic cell): a CSR matvec
        # is the fast path, and one row then holds one cell's whole input.
        self.WT_E = self.W_E.T.tocsr(); self.WT_E.sort_indices()
        self.WT_I = self.W_I.T.tocsr(); self.WT_I.sort_indices()
        self.gE_vec = np.array([self.gE[r] for r in self.role])
        self.gI_vec = np.array([self.gI[r] for r in self.role])

        # The plastic block: KC -> MBON entries of WT_E (row = MBON, col = KC),
        # addressed by position in .data (csr .data order == tocoo order).
        coo = self.WT_E.tocoo()
        plastic = (self.role[coo.row] == "mbon") & (self.role[coo.col] == "kc")
        self.plastic_pos = np.where(plastic)[0]
        self.plastic_mbon = coo.row[plastic]
        self.plastic_kc = coo.col[plastic]
        self.base_w = self.WT_E.data[self.plastic_pos].copy()
        self.factor = np.ones(len(self.plastic_pos))
        self.n_plastic = int(len(self.plastic_pos))

        # Dopamine reaching each MBON: DAN -> MBON weights, normalised per MBON
        W_da = W_raw.multiply(is_dan[:, None].astype(np.float64)).tocsr()
        colD = np.asarray(W_da.sum(axis=0)).ravel()
        self.WT_da = W_da.multiply(1.0 / np.maximum(colD, 1e-9)[None, :]).T.tocsr()

        # ---- odours: fixed random glomerulus sets, named -------------------------
        self.odours: dict[str, list[str]] = {}
        orng = np.random.default_rng(7)
        k = max(3, int(round(self.odour_fraction * len(self.glomeruli))))
        for name in ("A", "B", "C"):
            self.odours[name] = sorted(orng.choice(self.glomeruli, size=k, replace=False).tolist())

        # ---- state ----------------------------------------------------------------
        self.v = np.zeros(self.n)
        self.s = np.zeros(self.n)
        self.refractory = np.zeros(self.n, dtype=np.int32)
        self.spikes = np.zeros(self.n, dtype=bool)
        self.rate = np.zeros(self.n)
        self.odour: str | None = None
        self.reward = 0.0
        self.punish = 0.0
        self.step_count = 0
        self.naive: dict[str, float] = {}     # first-ever preference reading per odour
        self.learned: dict[str, float] = {}   # latest reading
        self._pref_acc: list[float] = []

    # ---- inputs ---------------------------------------------------------------
    def set_odour(self, name: str | None) -> None:
        if name is not None and name not in self.odours:
            raise KeyError(name)
        if name != self.odour:
            self._pref_acc = []
        self.odour = name

    def set_reward(self, x: float) -> None:
        self.reward = float(max(0.0, min(1.0, x)))

    def set_punish(self, x: float) -> None:
        self.punish = float(max(0.0, min(1.0, x)))

    def forget(self) -> None:
        """Wipe the memory: every KC->MBON synapse back to its connectome weight."""
        self.factor[:] = 1.0
        self.WT_E.data[self.plastic_pos] = self.base_w
        self.naive.clear()
        self.learned.clear()
        self.rate[:] = 0.0          # so the next "naive" reading isn't coloured by the old memory
        self._pref_acc = []

    def reset(self) -> None:
        self.v[:] = 0.0
        self.s[:] = 0.0
        self.refractory[:] = 0
        self.spikes[:] = False
        self.rate[:] = 0.0
        self.odour = None
        self.reward = 0.0
        self.punish = 0.0
        self._pref_acc = []
        self.step_count = 0

    # ---- dynamics ---------------------------------------------------------------
    def step(self) -> dict:
        ext = np.zeros(self.n)
        if self.odour is not None:
            for g in self.odours[self.odour]:
                ext[self.pn_by_glom[g]] += self.odour_drive
        if self.reward > 0:
            ext[self.idx["pam"]] += self.dan_drive * self.reward
        if self.punish > 0:
            ext[self.idx["ppl1"]] += self.dan_drive * self.punish

        E_in = self.WT_E @ self.s      # (n,) : input to each postsynaptic cell
        I_in = self.WT_I @ self.s
        I = self.gE_vec * E_in - self.gI_vec * I_in + ext
        if self.noise_std > 0:
            I = I + self.rng.normal(0.0, self.noise_std, self.n)

        self.v += (-self.v + I) / self.tau_m
        refr = self.refractory > 0
        self.v[refr] = 0.0
        self.refractory[refr] -= 1
        np.maximum(self.v, -1.0, out=self.v)
        self.spikes = (self.v >= self.threshold) & ~refr
        self.v[self.spikes] = 0.0
        self.refractory[self.spikes] = self.refractory_steps
        self.s *= (1.0 - 1.0 / self.tau_s)
        self.s[self.spikes] += 1.0
        self.rate += self.rate_alpha * (self.spikes.astype(np.float64) - self.rate)
        self.step_count += 1

        # ---- the only plastic synapse: KC -> MBON, gated by dopamine ---------------
        if self.reward > 0 or self.punish > 0:
            da = self.WT_da @ self.s                     # dopamine landing on each cell (MBONs matter)
            coincidence = self.s[self.plastic_kc] * da[self.plastic_mbon]
            if coincidence.any():
                self.factor *= (1.0 - self.eta * coincidence)
                np.maximum(self.factor, self.f_min, out=self.factor)
                self.WT_E.data[self.plastic_pos] = self.base_w * self.factor
        if self.step_count % 10 == 0:                     # slow recovery toward the connectome weights
            depressed = self.factor < 0.999
            if depressed.any():
                self.factor[depressed] += 10.0 * (1.0 - self.factor[depressed]) / self.tau_forget
                self.WT_E.data[self.plastic_pos] = self.base_w * self.factor

        # ---- readout: what does the fly make of this odour? ---------------------------
        app = float(self.rate[self.mbon_approach].mean())
        avo = float(self.rate[self.mbon_avoid].mean())
        pref = (app - avo) / (app + avo + 1e-6)
        if self.odour is not None:
            self._pref_acc.append(pref)
            if len(self._pref_acc) >= 150:            # skip the onset transient; a reading = mean of the last 60 steps
                reading = float(np.mean(self._pref_acc[-60:]))
                self.learned[self.odour] = reading
                self.naive.setdefault(self.odour, reading)
        kc = self.idx["kc"]
        return {
            "spikes": np.nonzero(self.spikes)[0].tolist(),
            "odour": self.odour,
            "reward": self.reward,
            "punish": self.punish,
            "pref": round(pref, 3),
            "approach": round(app, 4),
            "avoid": round(avo, 4),
            "kc_sparsity": round(float((self.rate[kc] > 0.02).mean()), 4),
            "n_depressed": int((self.factor < 0.9).sum()),
            "naive": {k: round(v, 3) for k, v in self.naive.items()},
            "learned": {k: round(v, 3) for k, v in self.learned.items()},
        }

    def mbon_trace(self) -> list[float]:
        """Per MBON: the fraction of its Kenyon-cell input weight lost to depression (0 = untouched)."""
        lost = np.zeros(self.n); tot = np.zeros(self.n)
        np.add.at(lost, self.plastic_mbon, self.base_w * (1.0 - self.factor))
        np.add.at(tot, self.plastic_mbon, self.base_w)
        m = self.idx["mbon"]
        return np.round(np.where(tot[m] > 0, lost[m] / np.maximum(tot[m], 1e-9), 0.0), 4).tolist()
