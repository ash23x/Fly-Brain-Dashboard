"""
Finds gain settings where the compass circuit behaves like a ring attractor.

Protocol per parameter set (all in simulation steps):
  0-100     cue at 90 deg (a "landmark"), then cue off
  100-1100  nothing: does ONE bump persist? (R = population-vector coherence)
  1100-1600 turn = +0.5: does the heading advance steadily?
  1600-2100 turn = -0.5: ...and back the other way?
  2100-2600 turn = 0: does it hold where it was left?

Score rewards: strong single bump while idle (R high, low drift), EPGs
neither silent nor saturated, symmetric rotation with the sign of the
command, bump still coherent while turning.

    python scripts/04_tune_compass.py              # grid search, writes best to data/compass/params.json
    python scripts/04_tune_compass.py --show gE=3 gI=6 epg_drive=0.3   # one run, verbose
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from compass_sim import CompassSim  # noqa: E402

COMPASS_DIR = ROOT / "data" / "compass"


def unwrap_deg(h: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(h)))


def run_protocol(params: dict, seed: int = 0, turn: float = 0.5) -> dict:
    params = dict(params)
    cue = params.pop("cue", 2.0)
    turn = params.pop("turn", turn)
    sim = CompassSim(COMPASS_DIR, seed=seed, **params)
    H, R, E = [], [], []
    sim.cue(90.0, strength=cue, steps=100)
    for t in range(2600):
        if t == 1100:
            sim.set_turn(turn)
        elif t == 1600:
            sim.set_turn(-turn)
        elif t == 2100:
            sim.set_turn(0.0)
        out = sim.step()
        H.append(out["heading"]); R.append(out["strength"]); E.append(np.mean(out["epg_rate"]))
    H, R, E = np.array(H), np.array(R), np.array(E)
    Hu = unwrap_deg(H)
    hold = slice(600, 1100)
    m = {
        "R_hold": float(R[hold].mean()),
        "drift_hold": float(abs(Hu[1099] - Hu[600])),
        "err_cue": float(abs(np.angle(np.exp(1j * np.radians(H[1099] - 90))) * 180 / np.pi)),
        "rate_hold": float(E[hold].mean()),
        "rot_pos": float(Hu[1599] - Hu[1300]),      # deg per 300 steps
        "rot_neg": float(Hu[2099] - Hu[1800]),
        "R_turn": float(R[1100:2100].mean()),
        "R_after": float(R[2300:2600].mean()),
        "drift_after": float(abs(Hu[2599] - Hu[2300])),
        "reignitions": int(sim.reignitions),
        # +turn then -turn for equal times should bring the needle home (path integration)
        "err_return": float(abs(np.angle(np.exp(1j * np.radians(H[2599] - H[1099]))) * 180 / np.pi)),
    }
    return m


def score(m: dict) -> float:
    s = 0.0
    s += 3.0 * min(m["R_hold"], 0.9)                       # coherent bump while idle
    s -= 0.02 * m["drift_hold"]                            # ...that stays put
    s -= 0.01 * m["err_cue"]                               # ...where the landmark said
    if not (0.02 <= m["rate_hold"] <= 0.25):               # EPGs alive but not saturated
        s -= 2.0
    if m["rot_pos"] > 15 and m["rot_neg"] < -15:           # turns both ways on command
        s += 2.0
        s += 1.0 * min(abs(m["rot_pos"]), abs(m["rot_neg"])) / max(abs(m["rot_pos"]), abs(m["rot_neg"]))
    s += 1.5 * min(m["R_turn"], 0.8)                       # still one bump while turning
    s += 1.0 * min(m["R_after"], 0.8) - 0.01 * m["drift_after"]
    return s


def grid() -> list[dict]:
    g = {
        "gE": [4.0, 6.0, 8.0, 10.0],
        "gI": [3.0, 6.0, 10.0, 15.0],
        "epg_drive": [0.3, 0.6, 0.9],
        "pen_drive": [0.2, 0.4],
        "turn_gain": [0.5],
        "noise_std": [0.05],
    }
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[g[k] for k in keys])]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--show", nargs="*", help="key=value pairs for a single verbose run")
    p.add_argument("--robust", nargs="*", help="key=value pairs: run seeds 0-5 at full turn, report deaths")
    p.add_argument("--top", type=int, default=12)
    a = p.parse_args(argv)

    if a.robust is not None:
        params = {k: float(v) for k, v in (kv.split("=") for kv in a.robust)}
        params.setdefault("turn", 1.0)
        print(json.dumps(params))
        tot = 0
        for seed in range(6):
            m = run_protocol(params, seed=seed)
            tot += m["reignitions"]
            print(f"  seed {seed}: deaths={m['reignitions']}  R_hold={m['R_hold']:.2f} rate={m['rate_hold']:.3f} "
                  f"rot+={m['rot_pos']:6.0f} rot-={m['rot_neg']:6.0f} R_after={m['R_after']:.2f} "
                  f"drift_after={m['drift_after']:5.1f} return_err={m['err_return']:5.1f}")
        print(f"  total deaths over 6 seeds: {tot}")
        return

    if a.show is not None:
        params = {k: float(v) for k, v in (kv.split("=") for kv in a.show)}
        m = run_protocol(params)
        print(json.dumps(params), "score", round(score(m), 2))
        for k, v in m.items():
            print(f"  {k:12s} {v:8.3f}")
        return

    results = []
    for i, params in enumerate(grid()):
        m = run_protocol(params)
        results.append((score(m), params, m))
        print(f"[{i + 1:3d}] score={score(m):6.2f} gE={params['gE']:<4} gI={params['gI']:<5} "
              f"epg={params['epg_drive']:<4} pen={params['pen_drive']:<4} "
              f"R={m['R_hold']:.2f} drift={m['drift_hold']:6.1f} rate={m['rate_hold']:.3f} "
              f"rot+={m['rot_pos']:7.1f} rot-={m['rot_neg']:7.1f} Rturn={m['R_turn']:.2f}", flush=True)
    results.sort(key=lambda t: -t[0])
    print("\nTOP:")
    for s, params, m in results[:a.top]:
        print(f"  {s:6.2f}  {json.dumps(params)}  R={m['R_hold']:.2f} rot+={m['rot_pos']:.0f} rot-={m['rot_neg']:.0f}")
    best = results[0][1]
    with open(COMPASS_DIR / "params.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nWrote best params to {COMPASS_DIR / 'params.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
