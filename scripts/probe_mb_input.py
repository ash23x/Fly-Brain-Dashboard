"""Who feeds the Kenyon cells and the MBONs? Input composition by presynaptic role."""
import json, pathlib
import numpy as np
from scipy import sparse
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "data" / "mushroom"
W = sparse.load_npz(D / "adjacency.npz").tocsr()
meta = json.load(open(D / "neurons.json"))
role = np.array([m["role"] for m in meta]); glom = np.array([m["glom"] for m in meta])
for post in ("kc", "mbon", "apl", "pam", "ppl1"):
    cols = np.where(role == post)[0]
    sub = W[:, cols]
    tot = sub.sum()
    parts = {pre: float(sub[np.where(role == pre)[0]].sum()) / tot for pre in ("pn", "kc", "apl", "dpm", "mbon", "pam", "ppl1")}
    print(f"input to {post:5s}: " + "  ".join(f"{k}={v:.2f}" for k, v in parts.items() if v > 0.005))
# glomerulus strength: total PN->KC synapses per glomerulus
kc = np.where(role == "kc")[0]
g_tot = {}
for g in sorted(set(glom[role == "pn"])):
    pns = np.where((role == "pn") & (glom == g))[0]
    g_tot[g] = float(W[pns][:, kc].sum())
vals = np.array(list(g_tot.values()))
print(f"\nPN->KC synapses per glomerulus: min {vals.min():.0f} median {np.median(vals):.0f} max {vals.max():.0f}")
print("weakest:", sorted(g_tot.items(), key=lambda kv: kv[1])[:6])
print("strongest:", sorted(g_tot.items(), key=lambda kv: -kv[1])[:6])
