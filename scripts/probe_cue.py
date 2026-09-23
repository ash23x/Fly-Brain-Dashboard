"""How reliably does a landmark cue relocate an established bump?
For each variant: let a bump form and settle, drop a cue at a random angle at least
60 deg away, and measure the heading error 60, 120 and 240 steps after the cue starts.
    python scripts/probe_cue.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from compass_sim import CompassSim  # noqa: E402
from sim_server import read_params  # noqa: E402


def circ(a, b):
    d = (a - b) % 360
    return d - 360 if d > 180 else d


def trial(sim: CompassSim, rng: np.random.Generator, **cue_kw) -> tuple[float, float, float, bool]:
    for _ in range(400):
        sim.step()
    h0 = sim.heading
    target = (h0 + rng.uniform(60, 300)) % 360
    sim.cue(target, **cue_kw)
    errs = []
    died = False
    for k in range(1, 241):
        out = sim.step()
        if out["strength"] == 0:
            died = True
        if k in (60, 120, 240):
            errs.append(abs(circ(out["heading"], target)))
    return errs[0], errs[1], errs[2], died


def main() -> None:
    d = ROOT / "data" / "compass"
    params = read_params(d)
    variants = {
        "current (2.0 x 60)": dict(strength=2.0, steps=60),
        "stronger (3.0 x 60)": dict(strength=3.0, steps=60),
        "longer (2.0 x 120)": dict(strength=2.0, steps=120),
        "both (3.0 x 120)": dict(strength=3.0, steps=120),
    }
    if "suppress" in CompassSim.cue.__code__.co_varnames:
        variants["mask (2.0 x 60, suppress 1.0)"] = dict(strength=2.0, steps=60, suppress=1.0)
        variants["mask (2.0 x 60, suppress 0.5)"] = dict(strength=2.0, steps=60, suppress=0.5)
        variants["mask + hold 0.35"] = dict(strength=2.0, steps=60, suppress=1.0, hold=0.35)
        variants["mask + hold 0.2"] = dict(strength=2.0, steps=60, suppress=1.0, hold=0.2)
        variants["mask + hold 0.5"] = dict(strength=2.0, steps=60, suppress=1.0, hold=0.5)
    n_trials = 30
    variants = {k: v for k, v in variants.items() if "hold" in k or k.startswith("current")}
    for name, kw in variants.items():
        rng = np.random.default_rng(0)
        e60, e120, e240, deaths = [], [], [], 0
        for t in range(n_trials):
            sim = CompassSim(d, seed=100 + t, **params)
            a, b, c, died = trial(sim, rng, **kw)
            e60.append(a); e120.append(b); e240.append(c); deaths += died
        ok = np.mean(np.array(e240) < 25) * 100
        print(f"{name:32s} err@60 {np.median(e60):5.1f}  err@120 {np.median(e120):5.1f}  err@240 {np.median(e240):5.1f} deg   "
              f"landed (<25 deg) {ok:3.0f} %   silent steps in {deaths}/{n_trials} trials")
    # after an anchored landmark, does the fly still turn, and how far does the bump slide when released?
    rng = np.random.default_rng(1)
    turned, slid = [], []
    for t in range(15):
        sim = CompassSim(d, seed=200 + t, **params)
        trial(sim, rng, strength=2.0, steps=60, suppress=1.0, hold=0.35)
        h0 = sim.heading
        sim.set_turn(1.0)
        path = 0.0; prev = h0
        for _ in range(300):
            out = sim.step()
            if out["strength"] > 0:
                path += circ(out["heading"], prev); prev = out["heading"]
        turned.append(path)
        sim.set_turn(0.0); h1 = sim.heading
        for _ in range(600):
            sim.step()
        slid.append(abs(circ(sim.heading, h1)))
    print(f"turn after an anchored landmark: median {np.median(turned):.0f} deg in 300 steps (want ~360); "
          f"slide in the 600 steps after release: median {np.median(slid):.0f} deg")


if __name__ == "__main__":
    main()
