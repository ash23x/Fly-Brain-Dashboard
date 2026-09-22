"""Sweep KC / APL / MBON gains for a sparse, odour-specific Kenyon-cell code."""
import pathlib, sys, itertools
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from mushroom_sim import MushroomSim

def measure(params):
    sim = MushroomSim(ROOT / "data" / "mushroom", seed=0, **params)
    codes = {}; app = {}; avo = {}; aplr = {}
    for od in ("A", "B", "C"):
        sim.set_odour(None)
        for _ in range(100): sim.step()
        sim.set_odour(od)
        for _ in range(300): out = sim.step()
        kc = sim.idx["kc"]
        codes[od] = set(np.where(sim.rate[kc] > 0.02)[0])
        app[od] = out["approach"]; avo[od] = out["avoid"]; aplr[od] = float(sim.rate[sim.idx["apl"]].mean())
    sp = [len(codes[o]) / 4064 for o in "ABC"]
    ov = [len(codes[a] & codes[b]) / max(1, min(len(codes[a]), len(codes[b]))) for a, b in (("A", "B"), ("A", "C"), ("B", "C"))]
    return sp, ov, app, avo, aplr

grid = {"gE_kc": [4.0, 5.0, 6.0], "gE_apl": [15.0, 30.0], "gI_kc": [8.0, 16.0], "gE_mbon": [40.0]}
keys = list(grid)
for vals in itertools.product(*[grid[k] for k in keys]):
    p = dict(zip(keys, vals))
    sp, ov, app, avo, aplr = measure(p)
    print(f"gE_kc={p['gE_kc']:<4} gE_apl={p['gE_apl']:<5} gI_kc={p['gI_kc']:<5} "
          f"sparsity A/B/C={sp[0]:.3f}/{sp[1]:.3f}/{sp[2]:.3f}  overlap AB/AC/BC={ov[0]:.2f}/{ov[1]:.2f}/{ov[2]:.2f}  "
          f"APL={np.mean(list(aplr.values())):.3f}  MBON app={np.mean(list(app.values())):.3f} avo={np.mean(list(avo.values())):.3f}", flush=True)
