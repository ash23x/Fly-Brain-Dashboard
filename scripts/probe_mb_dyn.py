"""Pavlov protocol on the mushroom body sim: sparsity, specificity, learning.
usage: probe_mb_dyn.py [key=value ...]  e.g. gE_kc=8 gI_kc=12 gE_mbon=40 eta=0.02"""
import pathlib, sys, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from mushroom_sim import MushroomSim

params = {k: float(v) for k, v in (kv.split("=") for kv in sys.argv[1:])}
t0 = time.time()
sim = MushroomSim(ROOT / "data" / "mushroom", seed=0, **params)
print(f"init {time.time() - t0:.1f}s  n={sim.n}  plastic KC->MBON synapses={sim.n_plastic:,}  "
      f"approach MBONs={len(sim.mbon_approach)} avoid={len(sim.mbon_avoid)}")
print("odours:", {k: len(v) for k, v in sim.odours.items()},
      "A∩B glomeruli:", len(set(sim.odours['A']) & set(sim.odours['B'])))

def run(steps, odour=None, reward=0.0, punish=0.0, label=""):
    sim.set_odour(odour); sim.set_reward(reward); sim.set_punish(punish)
    t0 = time.time(); out = None; prefs = []
    for _ in range(steps):
        out = sim.step()
        if _ >= steps // 3:
            prefs.append(out["pref"])
    dt = (time.time() - t0) / steps * 1000
    kc = sim.idx["kc"]; r = sim.rate
    active = set(np.where(r[kc] > 0.02)[0])
    print(f"{label:28s} {dt:5.2f} ms/step  PN={r[sim.idx['pn']].mean():.3f} KC={r[kc].mean():.4f} "
          f"sparsity={out['kc_sparsity']:.3f} APL={r[sim.idx['apl']].mean():.3f} "
          f"MBON app={out['approach']:.3f} avo={out['avoid']:.3f} pref={np.mean(prefs):+.3f} "
          f"PAM={r[sim.idx['pam']].mean():.3f} PPL1={r[sim.idx['ppl1']].mean():.3f} depressed={out['n_depressed']:,}")
    return active, float(np.mean(prefs))

run(150, None, label="silence")
kcA, pA0 = run(300, "A", label="odour A (naive)")
run(100, None, label="gap")
kcB, pB0 = run(300, "B", label="odour B (naive)")
run(100, None, label="gap")
kcA2, _ = run(300, "A", label="odour A again")
print(f"KC code: |A|={len(kcA)} |B|={len(kcB)} |A∩B|={len(kcA & kcB)} |A∩A'|={len(kcA & kcA2)} "
      f"(chance overlap ≈ {len(kcA) * len(kcB) / 4064:.0f})")
for i in range(3):
    run(100, None, label="gap")
    run(200, "A", reward=1.0, label=f"TRAIN A + reward #{i + 1}")
run(100, None, label="gap")
_, pA1 = run(300, "A", label="test A after reward")
run(100, None, label="gap")
_, pB1 = run(300, "B", label="test B (control)")
for i in range(3):
    run(100, None, label="gap")
    run(200, "B", punish=1.0, label=f"TRAIN B + punish #{i + 1}")
run(100, None, label="gap")
_, pB2 = run(300, "B", label="test B after punish")
run(100, None, label="gap")
_, pA2 = run(300, "A", label="test A (still likes it?)")
print(f"\nPREFERENCE  A: naive {pA0:+.3f} -> after reward {pA1:+.3f} -> later {pA2:+.3f}")
print(f"PREFERENCE  B: naive {pB0:+.3f} -> control {pB1:+.3f} -> after punish {pB2:+.3f}")
