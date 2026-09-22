"""Watch population rates over time through the tuning protocol (debug aid).
usage: probe_dyn.py gE=6 gI=6 ... [turn=0.5] [every=50]"""
import pathlib, sys, json
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from compass_sim import CompassSim
params = {k: float(v) for k, v in (kv.split("=") for kv in sys.argv[1:])}
cue = params.pop("cue", 2.0); turn = params.pop("turn", 0.5); every = int(params.pop("every", 50))
w0 = int(params.pop("from", -1)); w1 = int(params.pop("to", -1)); fine = int(params.pop("fine", 10))
seed = int(params.pop("seed", 0))
sim = CompassSim(ROOT / "data" / "compass", seed=seed, **params)
sim.cue(90.0, strength=cue, steps=100)
print("   t turn heading  R     epg   penL  penR  peg   d7   | nEPG>0.02  v_epg")
for t in range(2600):
    if t == 1100: sim.set_turn(turn)
    elif t == 1600: sim.set_turn(-turn)
    elif t == 2100: sim.set_turn(0.0)
    out = sim.step()
    if t % every == 0 or (w0 <= t <= w1 and t % fine == 0):
        r = sim.rate
        print(f"{t:4d} {sim.turn:+.1f} {out['heading']:6.1f}  {out['strength']:.2f}  "
              f"{r[sim.epg_idx].mean():.3f} {r[sim.pen_left].mean():.3f} {r[sim.pen_right].mean():.3f} "
              f"{r[sim.peg_idx].mean():.3f} {r[sim.d7_idx].mean():.3f} | {(r[sim.epg_idx] > 0.02).sum():3d}  {sim.v[sim.epg_idx].mean():6.2f}")
print(f"reignitions: {sim.reignitions}")
