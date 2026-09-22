"""
Spiking simulation of the fly's head-direction circuit (EPG / PEN / PEG /
Delta7), built by scripts/03_build_compass.py from the MaleCNS connectome.

What it does: a single bump of activity forms on the ring of EPG neurons
and stays put with NO input (a ring attractor -- memory held by the
geometry of the wiring, no weight ever changes). Driving the PEN neurons
on one side of the protocerebral bridge harder than the other pushes the
bump round the ring: that is the fly turning. Readout is the population
vector of EPG firing -- the "needle".

Model (leaky integrate-and-fire, current-based synapses):
    s_j  <- s_j * (1 - 1/tau_s) + spike_j              synaptic activation
    E_i  =  sum_j s_j W_ji / sum_j W_ji  over excitatory j   (fraction active)
    I_i  =  same over inhibitory (glutamatergic Delta7) j
    v_i  <- v_i + (-v_i + gE*E_i - gI*I_i + ext_i + noise) / tau_m
    spike when v_i >= 1, reset to 0, 3 refractory steps

Per-neuron input normalisation means gE / gI are "input when 100% of my
excitatory / inhibitory partners are fully active". Synapse counts set
the *shape* of each neuron's input; gE / gI set the overall balance --
counts are not strengths, so the balance is a free parameter, tuned by
scripts/04_tune_compass.py.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import sparse


class CompassSim:
    def __init__(self, compass_dir: pathlib.Path, seed: int | None = 0, **params):
        W_raw = sparse.load_npz(compass_dir / "adjacency.npz").toarray().astype(np.float64)
        with open(compass_dir / "neurons.json") as f:
            self.neurons: list[dict] = json.load(f)
        self.n = W_raw.shape[0]
        if len(self.neurons) != self.n:
            raise ValueError("neurons.json / adjacency.npz size mismatch")
        self.rng = np.random.default_rng(seed)

        # ---- tunables (defaults = values found by 04_tune_compass.py) ------
        self.tau_m = params.get("tau_m", 10.0)
        self.tau_s = params.get("tau_s", 6.0)
        self.gE = params.get("gE", 4.5)
        self.gI = params.get("gI", 6.0)
        self.epg_drive = params.get("epg_drive", 0.3)   # tonic current into EPGs
        self.pen_drive = params.get("pen_drive", 0.4)   # tonic current into PENs
        self.turn_gain = params.get("turn_gain", 1.0)   # PEN gain asymmetry at |turn| = 1
        self.turn_tau = params.get("turn_tau", 15.0)    # steps: turn commands are ramped, not stepped
        self.noise_std = params.get("noise_std", 0.05)
        # If the EPG ring goes completely silent (it can, after a violent
        # input change -- a model artefact, the real bump never vanishes),
        # a bump is re-seeded at a random heading, as it forms spontaneously
        # in a real fly in darkness.
        self.reignite_after = int(params.get("reignite_after", 40))
        self.refractory_steps = 3
        # Identical cells with instantaneous coupling love to fire in
        # lock-step -- and a small ring that spikes in unison then falls
        # silent in unison (everyone refractory at once) and dies. Real
        # neurons are not identical; a little threshold spread kills the
        # pathological synchrony.
        self.threshold_jitter = params.get("threshold_jitter", 0.1)
        self.threshold = 1.0 + self.threshold_jitter * self.rng.standard_normal(self.n)
        self.rate_alpha = params.get("rate_alpha", 0.1)  # EMA for the readout

        # ---- populations ---------------------------------------------------
        role = np.array([m["role"] for m in self.neurons])
        pb_side = np.array([m.get("pb_side", "") for m in self.neurons])
        self.epg_idx = np.where(role == "epg")[0]
        self.pen_idx = np.where(role == "pen")[0]
        self.pen_left = np.where((role == "pen") & (pb_side == "L"))[0]
        self.pen_right = np.where((role == "pen") & (pb_side == "R"))[0]
        self.d7_idx = np.where(role == "delta7")[0]
        self.peg_idx = np.where(role == "peg")[0]
        self.sign = np.array([m.get("sign", 1) for m in self.neurons], dtype=np.float64)
        self.n_inhibitory = int((self.sign < 0).sum())
        ang = np.array([m["angle"] if m.get("angle") is not None else np.nan for m in self.neurons])
        self.epg_angle = np.radians(ang[self.epg_idx])
        self.epg_unit = np.exp(1j * self.epg_angle)

        # ---- equalise wedge "mass" -----------------------------------------
        # This one fly has 2, 3 or 4 EPGs per wedge (and 2-3 PENs per PB
        # glomerulus). Left as is, a 4-cell wedge shouts louder than a 2-cell
        # one and the bump slides downhill toward the loud wedges. Scale each
        # cell's OUTPUT by (mean cells per group / cells in its group) so every
        # wedge projects the same total -- the sample's cell count is
        # individual variation, not the circuit's design.
        self.equalize = bool(params.get("equalize_wedges", True))
        self.out_scale = np.ones(self.n)
        if self.equalize:
            groups: dict[tuple, list[int]] = {}
            for i, m in enumerate(self.neurons):
                if m["role"] in ("epg", "pen"):
                    groups.setdefault((m["type"], m.get("glom", "")), []).append(i)
            for role in ("epg", "pen"):
                sizes = [len(v) for k, v in groups.items() if self.neurons[v[0]]["role"] == role]
                mean_size = float(np.mean(sizes)) if sizes else 1.0
                for k, members in groups.items():
                    if self.neurons[members[0]]["role"] == role:
                        self.out_scale[members] = mean_size / len(members)
            W_raw = W_raw * self.out_scale[:, None]

        # ---- split + normalise weights (rows = presynaptic) ---------------
        exc = self.sign > 0
        W_E = W_raw * exc[:, None]
        W_I = W_raw * (~exc)[:, None]
        self.W_E = W_E / np.maximum(W_E.sum(axis=0, keepdims=True), 1e-9)
        self.W_I = W_I / np.maximum(W_I.sum(axis=0, keepdims=True), 1e-9)
        self.W_raw = W_raw
        self.nnz = int((W_raw > 0).sum())

        # ---- state -----------------------------------------------------------
        self.v = np.zeros(self.n)
        self.s = np.zeros(self.n)
        self.refractory = np.zeros(self.n, dtype=np.int32)
        self.spikes = np.zeros(self.n, dtype=bool)
        self.rate = np.zeros(self.n)
        self.turn = 0.0            # effective (ramped) command
        self.turn_cmd = 0.0        # requested command
        self._cue = np.zeros(self.n)
        self._cue_steps = 0
        self._silent_steps = 0
        self._reignite_random = True
        self.reignitions = 0
        self.heading = 0.0
        self.strength = 0.0
        self.step_count = 0

    # ---- inputs ------------------------------------------------------------
    def set_turn(self, omega: float) -> None:
        """omega in [-1, 1]: angular-velocity command (sign = direction)."""
        self.turn_cmd = float(max(-1.0, min(1.0, omega)))

    def cue(self, angle_deg: float, strength: float = 0.8, width_deg: float = 30.0, steps: int = 60) -> None:
        """A landmark: push current into the EPGs near `angle_deg` for a while.
        (In the fly this comes from the ring neurons of the visual system.)"""
        d = np.angle(np.exp(1j * (self.epg_angle - np.radians(angle_deg))))
        prof = strength * np.exp(-0.5 * (d / np.radians(width_deg)) ** 2)
        self._cue[:] = 0.0
        self._cue[self.epg_idx] = prof
        self._cue_steps = int(steps)

    def reset(self) -> None:
        self.v[:] = 0.0
        self.s[:] = 0.0
        self.refractory[:] = 0
        self.spikes[:] = False
        self.rate[:] = 0.0
        self._cue_steps = 0
        self._silent_steps = 0
        self._reignite_random = True
        self.turn = 0.0
        self.turn_cmd = 0.0
        self.step_count = 0

    # ---- dynamics ----------------------------------------------------------
    def step(self) -> dict:
        self.turn += (self.turn_cmd - self.turn) / self.turn_tau
        # Turning = gain-modulating the PENs: left-bridge PENs hand the bump
        # back to the ring shifted one way, right-bridge PENs the other, so
        # scaling one side up and the other down produces a pure sideways
        # push while the total drive on the ring stays the same. (An additive
        # current does the job too, but it also fattens or starves the bump.)
        s_eff = self.s.copy()
        s_eff[self.pen_left] *= 1.0 + self.turn_gain * self.turn
        s_eff[self.pen_right] *= 1.0 - self.turn_gain * self.turn
        E_in = s_eff @ self.W_E
        I_in = self.s @ self.W_I
        ext = np.zeros(self.n)
        ext[self.epg_idx] += self.epg_drive
        ext[self.pen_idx] += self.pen_drive
        if self._cue_steps > 0:
            ext += self._cue
            self._cue_steps -= 1
        I = self.gE * E_in - self.gI * I_in + ext
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

        # ---- readout: population vector of EPG firing = the needle ---------
        r = self.rate[self.epg_idx]
        tot = r.sum()
        if tot > 0.05:
            z = (r * self.epg_unit).sum() / tot
            self.heading = float(np.degrees(np.angle(z)) % 360)
            self.strength = float(abs(z))
            self._silent_steps = 0
        else:
            self.strength = 0.0
            self._silent_steps += 1
            if self._cue_steps == 0 and self._silent_steps >= self.reignite_after:
                # After an explicit reset: anywhere, like a fly waking in the
                # dark. After a spontaneous collapse (rare, a model artefact):
                # where it was, so the needle just flickers instead of jumping.
                where = float(self.rng.uniform(0, 360)) if self._reignite_random else self.heading
                self.cue(where, strength=2.0, steps=60)
                self._reignite_random = False
                self.reignitions += 1
                self._silent_steps = 0

        return {
            "spikes": np.nonzero(self.spikes)[0].tolist(),
            "heading": round(self.heading, 2),
            "strength": round(self.strength, 3),
            "turn": round(self.turn, 3),
            "epg_rate": np.round(r, 3).tolist(),
        }
