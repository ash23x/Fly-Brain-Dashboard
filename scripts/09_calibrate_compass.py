"""
Calibrates the compass ring's favourite headings away.

This one fly's wiring is not uniform round the ring: left alone, a bump slides
toward a handful of preferred wedges (tens of degrees in a few seconds), so a
heading set by a turn or a landmark does not hold. A real fly calibrates its
compass continually against vision and self-motion; we calibrate once, with the
smallest possible adjustment: one tonic bias current per wedge (18 numbers),
found by measuring the drift field and flattening it.

    python scripts/09_calibrate_compass.py            # writes epg_bias into data/compass/params.json
    python scripts/09_calibrate_compass.py --check    # just measure the current drift field

Measurement: anchor a bump at each of 36 headings with a landmark, let go, and
see how far it moves in 150 steps. Update: where the field converges (a basin)
lower that wedge's bias, where it diverges (a ridge) raise it; repeat.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from compass_sim import CompassSim  # noqa: E402
from sim_server import read_params  # noqa: E402

D = ROOT / "data" / "compass"
ANGLES = np.arange(0, 360, 10)


def circ(a, b):
    x = (a - b) % 360
    return x - 360 if x > 180 else x


def drift_field(params: dict, seeds=(0, 1), steps=150) -> tuple[np.ndarray, np.ndarray]:
    """Mean displacement (deg) of a released bump per start angle, and the slide over 900 steps."""
    v = np.zeros(len(ANGLES)); slide = np.zeros(len(ANGLES))
    for k, start in enumerate(ANGLES):
        for s in seeds:
            sim = CompassSim(D, seed=1000 + 7 * k + s, **params)
            for _ in range(300):
                sim.step()
            sim.cue(float(start), strength=2.0, steps=60, suppress=3.0, hold=0.5)
            for _ in range(200):
                sim.step()
            h0 = sim.heading
            sim._hold = None; sim.landmark = None
            for t in range(900):
                sim.step()
                if t == steps - 1:
                    v[k] += circ(sim.heading, h0)
            slide[k] += abs(circ(sim.heading, h0))
    return v / len(seeds), slide / len(seeds)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--eta", type=float, default=0.004)
    a = p.parse_args(argv)
    params = read_params(D)
    sim0 = CompassSim(D, **params)
    wedge_angles = np.array(sorted({round(m["angle"]) for m in sim0.neurons if m["role"] == "epg" and m.get("angle") is not None}), dtype=float)
    bias = np.array(params.get("epg_bias", [0.0] * len(wedge_angles)), dtype=float)

    def report(tag, v, slide):
        print(f"{tag}: |drift@150| median {np.median(np.abs(v)):5.1f} max {np.abs(v).max():5.1f} deg | slide@900 median {np.median(slide):5.1f} "
              f"90th {np.percentile(slide, 90):5.1f} max {slide.max():5.1f} deg | bias range {bias.min():+.3f}..{bias.max():+.3f}")

    v, slide = drift_field(params)
    report("start", v, slide)
    if a.check:
        return
    best = (np.median(slide) + np.percentile(slide, 90), bias.copy())
    for it in range(a.iters):
        # divergence of the field at each wedge: d(v)/d(theta), central differences on the 10-degree grid
        dv = (np.roll(v, -1) - np.roll(v, 1)) / 20.0
        at_wedge = np.interp(wedge_angles, ANGLES, dv, period=360)
        bias = np.clip(bias + a.eta * at_wedge * 20.0, -0.25, 0.35)
        bias -= bias.mean()                                       # keep the total tonic drive as tuned
        params["epg_bias"] = bias.tolist()
        v, slide = drift_field(params)
        report(f"iter {it + 1:2d}", v, slide)
        score = np.median(slide) + np.percentile(slide, 90)
        if score < best[0]:
            best = (score, bias.copy())
    bias = best[1]
    with open(D / "params.json") as f:
        full = json.load(f)
    full["epg_bias"] = [round(float(b), 4) for b in bias]
    full["_epg_bias_note"] = ("One tonic bias current per EPG wedge (wedges in ring order), found by scripts/09_calibrate_compass.py: "
                              "flattens this fly's favourite headings so a bump holds wherever a turn or a landmark leaves it.")
    with open(D / "params.json", "w") as f:
        json.dump(full, f, indent=2)
    print("wrote epg_bias to data/compass/params.json:", [round(float(b), 3) for b in bias])


if __name__ == "__main__":
    main(sys.argv[1:])
