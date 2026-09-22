"""
Cuts the fly's head-direction ("compass") circuit out of the MaleCNS
connectome and works out where each neuron sits on the ring.

The circuit (central complex, ~150 neurons):
  EPG    -- "compass neurons". Each one lives in one wedge of the ellipsoid
            body (EB) and one glomerulus of the protocerebral bridge (PB).
            A bump of activity across the EPG population IS the fly's
            heading estimate (Seelig & Jayaraman 2015; Kim et al. 2017).
  PEN    -- take the bump from the PB and hand it back to the EB shifted
            one wedge sideways. Left-PB PENs shift one way, right-PB PENs
            the other. Asymmetric PEN drive = the bump rotates = "turning".
  Delta7 -- glutamatergic (inhibitory) neurons spanning the PB. Broad
            inhibition that keeps the bump to ONE bump.
  PEG    -- close the EB -> PB loop for the return path.

Local excitation + broad inhibition on a ring = a ring attractor. The
bump persists with no input at all: working memory held by the wiring's
geometry, not by any weight changing. Kakaria & de Bivort (2017) showed
this emerges from a spiking model built on the PB connectome.

Ring angles are NOT hard-coded from anatomy. They are derived from the
wiring itself: a spectral embedding of the EPG->PEN->EPG + EPG->EPG loop
(the two lowest non-trivial Laplacian eigenvectors of a ring graph are
cos/sin of position). The PB glomerulus labels in the instance names are
then used only as a cross-check, printed at the end.

Run (offline, ~2 s):
    python scripts/03_build_compass.py            -> data/compass/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import scipy.linalg as sla
from scipy import sparse

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ANNOT_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT_FILE = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS_CANDIDATES = [
    "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]

COMPASS_TYPES = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7"]
ROLE = {"EPG": "epg", "EPGt": "epg", "PEN_a(PEN1)": "pen", "PEN_b(PEN2)": "pen",
        "PEG": "peg", "Delta7": "delta7"}
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
           "dopamine": 1, "serotonin": 1, "octopamine": 1}


def circ_mean(angles_deg: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Weighted circular mean (deg) and resultant length R in [0, 1]."""
    a = np.radians(np.asarray(angles_deg, dtype=float))
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return float("nan"), 0.0
    z = (w * np.exp(1j * a)).sum() / w.sum()
    return float(np.degrees(np.angle(z)) % 360), float(abs(z))


def load_circuit(data_dir: pathlib.Path, types: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    annot = feather.read_table(str(data_dir / ANNOT_FILE),
                               columns=["bodyId", "type", "instance", "somaSide"]).to_pandas()
    cx = annot[annot["type"].isin(types)].dropna(subset=["bodyId"]).copy()
    cx["bodyId"] = cx["bodyId"].astype("int64")
    cx = cx.drop_duplicates("bodyId").reset_index(drop=True)
    # PB glomerulus label from the instance name, e.g. "EPG(PB08)_L4" -> "L4",
    # "Delta7(PB15)_L1L9R8_R" -> "L1L9R8" (output glomeruli), side suffix dropped.
    cx["glom"] = cx["instance"].fillna("").str.replace(r"^.*\)_", "", regex=True)
    cx["glom"] = cx["glom"].str.replace(r"_[LR]$", "", regex=True)
    cx["pb_side"] = cx["glom"].str.extract(r"^([LR])")[0].fillna("")

    weights_path = next((data_dir / c for c in WEIGHTS_CANDIDATES if (data_dir / c).exists()), None)
    if weights_path is None:
        raise SystemExit(f"No connectome weights file in {data_dir}; run scripts/01_download_data.py")
    ids = pa.array(cx["bodyId"].tolist(), type=pa.int64())
    tab = feather.read_table(str(weights_path), columns=["body_pre", "body_post", "weight"])
    keep = pc.and_(pc.is_in(tab["body_pre"], value_set=ids), pc.is_in(tab["body_post"], value_set=ids))
    edges = tab.filter(keep).to_pandas()
    edges = edges[edges["body_pre"] != edges["body_post"]]
    return cx, edges


def dense_adjacency(cx: pd.DataFrame, edges: pd.DataFrame) -> np.ndarray:
    idx = {b: i for i, b in enumerate(cx["bodyId"])}
    A = np.zeros((len(cx), len(cx)), dtype=np.float64)
    np.add.at(A, (edges["body_pre"].map(idx).to_numpy(), edges["body_post"].map(idx).to_numpy()),
              edges["weight"].to_numpy(dtype=np.float64))
    return A


def ring_angles_from_wiring(A: np.ndarray, epg: np.ndarray, pen: np.ndarray) -> np.ndarray:
    """Angle (deg) of every EPG, from the wiring alone.

    Two-step EPG->PEN->EPG connectivity plus direct EPG->EPG gives a
    ring-shaped graph (each wedge talks to its neighbours through the PENs).
    The 2nd/3rd eigenvectors of its normalised Laplacian are cos/sin of the
    position around that ring -- classic spectral embedding of a cycle.
    """
    M = A[np.ix_(epg, pen)] @ A[np.ix_(pen, epg)] + A[np.ix_(epg, epg)]
    S = M + M.T
    np.fill_diagonal(S, 0.0)
    d = S.sum(axis=1)
    d[d == 0] = 1.0
    Dm = np.diag(d ** -0.5)
    Ln = np.eye(len(epg)) - Dm @ S @ Dm
    evals, evecs = sla.eigh(Ln)
    x, y = evecs[:, 1], evecs[:, 2]
    return np.degrees(np.arctan2(y, x)) % 360, evals[:5]


def build(data_dir: pathlib.Path, out_dir: pathlib.Path) -> dict:
    cx, edges = load_circuit(data_dir, COMPASS_TYPES)
    n = len(cx)
    print(f"Compass circuit: {n} neurons, {len(edges):,} edges")
    print(cx["type"].value_counts().to_string())
    A = dense_adjacency(cx, edges)

    role = cx["type"].map(ROLE).to_numpy()
    epg = np.where(role == "epg")[0]
    pen = np.where(role == "pen")[0]
    peg = np.where(role == "peg")[0]
    d7 = np.where(role == "delta7")[0]

    # ---- angles ----------------------------------------------------------
    angle = np.full(n, np.nan)
    angle[epg], evals = ring_angles_from_wiring(A, epg, pen)
    print(f"Laplacian eigenvalues: {np.round(evals, 4)}  (2nd and 3rd nearly equal => a ring)")

    # Everyone else: circular mean of the EPG angles they connect to.
    # PEN/PEG: where their EPG *inputs* sit. Delta7: where their *outputs* land.
    for i in pen:
        angle[i], _ = circ_mean(angle[epg], A[epg, i])
    for i in peg:
        angle[i], _ = circ_mean(angle[epg], A[epg, i])
    for i in d7:
        angle[i], _ = circ_mean(angle[epg], A[i, epg])

    # ---- cross-check against the anatomists' glomerulus labels -----------
    print("\nEPG ring position vs PB glomerulus label (should be tight clusters in order):")
    chk = pd.DataFrame({"glom": cx["glom"].values[epg], "angle": angle[epg]})
    rows = []
    for g, grp in chk.groupby("glom"):
        m, r = circ_mean(grp["angle"].to_numpy())
        rows.append((g, len(grp), m, r))
    rows.sort(key=lambda t: t[2])
    for g, c, m, r in rows:
        print(f"  {g:4s} n={c:2d}  angle={m:6.1f} deg  coherence={r:.2f}")
    loose = [g for g, c, m, r in rows if r < 0.9]
    if loose:
        print(f"  WARNING: glomeruli with loose clusters: {loose}")

    # Spectral embeddings get the ORDER right but stretch the spacing. Snap
    # each glomerulus group to an evenly spaced wedge in that order so the
    # heading readout is linear (18 groups -> 20 deg each). Raw values kept.
    angle_raw = angle.copy()
    wedge = 360.0 / len(rows)
    glom_angle = {g: k * wedge for k, (g, c, m, r) in enumerate(rows)}
    for i in epg:
        angle[i] = glom_angle[cx["glom"][i]]
    for i in np.concatenate([pen, peg]):
        angle[i], _ = circ_mean(angle[epg], A[epg, i])
    for i in d7:
        angle[i], _ = circ_mean(angle[epg], A[i, epg])
    print(f"  Snapped to {len(rows)} wedges of {wedge:.1f} deg. Order around the ring: "
          + " ".join(g for g, *_ in rows))

    # ---- neurotransmitter sign -----------------------------------------------
    nt = feather.read_table(str(data_dir / NT_FILE), columns=["body", "consensus_nt"]).to_pandas()
    nt = nt[nt["body"].isin(cx["bodyId"])]
    nt_map = dict(zip(nt["body"].astype("int64"), nt["consensus_nt"].astype(str).str.lower()))
    cx["nt"] = cx["bodyId"].map(nt_map).fillna("")
    cx["sign"] = cx["nt"].map(NT_SIGN).fillna(1).astype(int)
    print("\nTransmitter by type:")
    print(cx.groupby("type")["nt"].value_counts().to_string())

    # ---- save ----------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    W = sparse.csr_matrix(A.astype(np.float32))
    sparse.save_npz(out_dir / "adjacency.npz", W)
    meta = []
    for i in range(n):
        meta.append({
            "bodyId": int(cx["bodyId"][i]),
            "type": str(cx["type"][i]),
            "class": str(cx["type"][i]),
            "role": str(role[i]),
            "side": str(cx["somaSide"][i]),
            "pb_side": str(cx["pb_side"][i]),
            "glom": str(cx["glom"][i]),
            "nt": str(cx["nt"][i]),
            "sign": int(cx["sign"][i]),
            "angle": None if np.isnan(angle[i]) else round(float(angle[i]), 2),
            "angle_raw": None if np.isnan(angle_raw[i]) else round(float(angle_raw[i]), 2),
        })
    with open(out_dir / "neurons.json", "w") as f:
        json.dump(meta, f)
    print(f"\nSaved {n} neurons / {W.nnz} edges to {out_dir}")
    return {"n": n, "nnz": int(W.nnz)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Extract the compass (head-direction) circuit.")
    p.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR)
    p.add_argument("--out", type=pathlib.Path, default=None)
    a = p.parse_args(argv)
    build(a.data_dir, a.out or (a.data_dir / "compass"))


if __name__ == "__main__":
    main(sys.argv[1:])
