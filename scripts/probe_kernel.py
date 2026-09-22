"""Effective connection kernels of the compass circuit vs angular offset.
For each pair of EPG wedges, how much excitation (via PEN/PEG/EPG) and how
much inhibition (via Delta7) flows from wedge A to wedge B, as a function of
the angle between them. A ring attractor needs: excitation peaked at 0,
inhibition broad / dipped at 0. Read-only diagnostic."""
import json, pathlib, sys
import numpy as np
from scipy import sparse

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "data" / "compass"
A = sparse.load_npz(D / "adjacency.npz").toarray().astype(float)
meta = json.load(open(D / "neurons.json"))
role = np.array([m["role"] for m in meta]); ang = np.array([m["angle"] if m["angle"] is not None else np.nan for m in meta])
sign = np.array([m["sign"] for m in meta])
epg = np.where(role == "epg")[0]; pen = np.where(role == "pen")[0]; peg = np.where(role == "peg")[0]; d7 = np.where(role == "delta7")[0]
penL = np.array([i for i in pen if meta[i]["pb_side"] == "L"]); penR = np.array([i for i in pen if meta[i]["pb_side"] == "R"])

def kernel(M, name):
    """M: EPG x EPG effective matrix. Bin by angular offset (target - source)."""
    bins = {}
    for a in range(len(epg)):
        for b in range(len(epg)):
            d = (ang[epg[b]] - ang[epg[a]] + 180) % 360 - 180
            bins.setdefault(round(d / 20) * 20, []).append(M[a, b])
    ks = sorted(bins)
    print(f"{name:26s}" + " ".join(f"{k:>6d}" for k in ks))
    print(f"{'':26s}" + " ".join(f"{np.mean(bins[k]):6.1f}" for k in ks))

kernel(A[np.ix_(epg, epg)], "EPG->EPG direct")
kernel(A[np.ix_(epg, penL)] @ A[np.ix_(penL, epg)] / 100, "EPG->PEN(L)->EPG /100")
kernel(A[np.ix_(epg, penR)] @ A[np.ix_(penR, epg)] / 100, "EPG->PEN(R)->EPG /100")
kernel(A[np.ix_(epg, peg)] @ A[np.ix_(peg, epg)] / 100, "EPG->PEG->EPG /100")
kernel(A[np.ix_(epg, d7)] @ A[np.ix_(d7, epg)] / 100, "EPG->D7->EPG /100 (inhib)")
# Delta7 dendritic input profile: for each D7, EPG input weight vs offset from its own output angle
bins = {}
for i in d7:
    for a in epg:
        d = (ang[a] - ang[i] + 180) % 360 - 180
        bins.setdefault(round(d / 20) * 20, []).append(A[a, i])
ks = sorted(bins)
print(f"{'EPG->D7 in, vs D7 out angle':26s}" + " ".join(f"{k:>6d}" for k in ks))
print(f"{'':26s}" + " ".join(f"{np.mean(bins[k]):6.1f}" for k in ks))
print("D7->D7 total", A[np.ix_(d7, d7)].sum(), " D7->PEN_b", A[np.ix_(d7, pen)].sum(), " D7->PEG", A[np.ix_(d7, peg)].sum())
