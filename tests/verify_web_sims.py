"""The Python reference simulators run through the same protocol as
tests/verify_web_sims.js, so the browser port can be compared number for number.
    python tests/verify_web_sims.py [compass|mushroom|all]
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from compass_sim import CompassSim  # noqa: E402
from mushroom_sim import MushroomSim  # noqa: E402
from sim_server import read_params  # noqa: E402


def circ(a, b):
    d = (a - b) % 360
    return d - 360 if d > 180 else d


def path_length(frames):
    return sum(circ(f2["heading"], f1["heading"]) for f1, f2 in zip(frames, frames[1:]) if f1["strength"] > 0 and f2["strength"] > 0)


def compass(seed):
    d = ROOT / "data" / "compass"
    sim = CompassSim(d, seed=seed, **read_params(d))
    t0 = time.perf_counter()

    def run(steps, turn):
        sim.set_turn(turn)
        return [sim.step() for _ in range(steps)]
    run(300, 0)
    sim.cue(90, strength=2.0, steps=60)
    hold1 = run(900, 0)
    right = run(300, 1)
    hold2 = run(600, 0)
    left = run(300, -1)
    hold3 = run(600, 0)
    allf = hold1 + right + hold2 + left + hold3
    ms = (time.perf_counter() - t0) * 1000 / len(allf) / 1.0
    return {"seed": seed, "cue_error": f"{circ(hold1[120]['heading'], 90):.1f}",
            "hold_drift": f"{path_length(hold1[300:]):.1f}", "hold_strength": f"{np.mean([f['strength'] for f in hold1[300:]]):.2f}",
            "turn_right_deg": f"{path_length(right):.0f}", "turn_left_deg": f"{path_length(left):.0f}",
            "silent_steps": sum(1 for f in allf if f["strength"] == 0),
            "mean_epg_rate": f"{np.mean([np.mean(f['epg_rate']) for f in hold1[300:]]):.3f}",
            "reignitions": sim.reignitions, "ms_per_step": f"{ms:.3f}"}


def mushroom(seed):
    d = ROOT / "data" / "mushroom"
    sim = MushroomSim(d, seed=seed, **read_params(d))
    t0 = time.perf_counter()
    n = [0]

    def run(steps):
        out = None
        for _ in range(steps):
            out = sim.step(); n[0] += 1
        return out

    def present(od, steps=300):
        sim.set_odour(od); out = run(steps); sim.set_odour(None); run(90); return out
    kc = set(sim.idx["kc"].tolist())
    codes = {}
    for od in "ABC":
        sim.set_odour(od); active = set()
        for i in range(300):
            f = sim.step(); n[0] += 1
            if i >= 150:
                active.update(x for x in f["spikes"] if x in kc)
        codes[od] = active; sim.set_odour(None); run(90)

    def overlap(a, b):
        return len(a & b) / max(1, min(len(a), len(b)))
    kc_sparsity = present("A")["kc_sparsity"]
    naive_a = sim.naive["A"]; naive_b = present("B")["learned"]["B"]

    def pair(od, kind, times=3):
        for _ in range(times):
            sim.set_odour(od); run(110)
            (sim.set_reward if kind == "reward" else sim.set_punish)(1.0); run(225)
            sim.set_reward(0); sim.set_punish(0); sim.set_odour(None); run(90)
    pair("A", "reward")
    test_a = present("A")["learned"]["A"]
    test_b1 = present("B")["learned"]["B"]
    pair("B", "punish")
    test_b2 = present("B")["learned"]["B"]
    test_a2 = present("A")["learned"]["A"]
    depressed = sim.step()["n_depressed"]
    ms = (time.perf_counter() - t0) * 1000 / n[0]
    return {"seed": seed, "kc_active_per_odour": [len(codes["A"]), len(codes["B"]), len(codes["C"])], "kc_sparsity": kc_sparsity,
            "overlap_AB": f"{overlap(codes['A'], codes['B']):.2f}", "overlap_AC": f"{overlap(codes['A'], codes['C']):.2f}",
            "naive_A": f"{naive_a:.2f}", "after_sugar_A": f"{test_a:.2f}", "control_B": f"{naive_b:.2f} -> {test_b1:.2f}",
            "after_shock_B": f"{test_b2:.2f}", "A_retained": f"{test_a2:.2f}", "n_depressed": depressed, "ms_per_step": f"{ms:.2f}"}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("compass", "all"):
        for s in (1, 2, 3):
            print("compass", json.dumps(compass(s)))
    if which in ("mushroom", "all"):
        seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else (1, 2)
        for s in seeds:
            print("mushroom", json.dumps(mushroom(s)))
