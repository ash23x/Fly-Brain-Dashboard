"""
Cuts the fly's learning centre -- the mushroom body -- out of the MaleCNS
connectome, with its inputs and its teachers.

  PN     -- uniglomerular projection neurons from the antennal lobe. Each
            carries one olfactory glomerulus. An "odour" is a set of
            glomeruli lit up together.
  KC     -- Kenyon cells (~4,000). Each samples a handful of PNs and fires
            only when several agree: a sparse, high-dimensional code for
            the odour. The APL neuron feeds back inhibition onto all of
            them and keeps the code sparse (a few % of cells per odour).
  MBON   -- mushroom body output neurons (97). They sum the Kenyon cells
            and push the fly toward or away from the odour.
  PAM    -- dopamine neurons signalling reward (sugar).
  PPL1   -- dopamine neurons signalling punishment (shock).

Learning lives at exactly one place: the KC->MBON synapse. Dopamine
arriving in a compartment while its KCs are active DEPRESSES those
synapses (Aso et al. 2014; Hige et al. 2015; Handler et al. 2019). Reward
DANs innervate compartments whose MBONs drive avoidance, so reward +
odour = that odour drives less avoidance = the fly approaches it.
Punishment DANs do the mirror image. The rule is three-factor
(KC x dopamine -> weight change) and it is the only thing in the model
that changes.

Each MBON's valence (approach / avoid) is not hand-labelled: it is read
off the connectome as whichever dopamine class (PAM vs PPL1) sends it more
synapses -- the compartment logic above, taken from the wiring.

Run (offline, ~5 s):
    python scripts/05_build_mushroom_body.py        -> data/mushroom/
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
from scipy import sparse

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ANNOT_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT_FILE = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS_CANDIDATES = [
    "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
           "dopamine": 1, "serotonin": 1, "octopamine": 1}
# Uniglomerular PN: "<glomerulus>_<lineage>PN", e.g. DA1_lPN, VM5d_adPN. Excludes
# multiglomerular (M_...), thermo/hygro (VP...) and odd ones (Z_...).
UNI_PN = re.compile(r"^(?!M_|Z_|VP)([A-Z]+\d*[a-z]?\d?)_(ad|l|il|lv)PN$")


def role_of(t: str) -> str | None:
    if UNI_PN.match(t):
        return "pn"
    if t.startswith("KC"):
        return "kc"
    if t.startswith("MBON"):
        return "mbon"
    if t.startswith("PAM"):
        return "pam"
    if t.startswith("PPL1"):
        return "ppl1"
    if t == "APL":
        return "apl"
    if t == "DPM":
        return "dpm"
    return None


def build(data_dir: pathlib.Path, out_dir: pathlib.Path) -> dict:
    annot = feather.read_table(str(data_dir / ANNOT_FILE),
                               columns=["bodyId", "type", "instance", "somaSide"]).to_pandas()
    annot["type"] = annot["type"].fillna("").astype(str)
    annot["role"] = annot["type"].map(role_of)
    mb = annot.dropna(subset=["role", "bodyId"]).copy()
    mb["bodyId"] = mb["bodyId"].astype("int64")
    mb = mb.drop_duplicates("bodyId").reset_index(drop=True)
    mb["glom"] = mb["type"].str.extract(UNI_PN.pattern)[0].fillna("")
    order = {"pn": 0, "kc": 1, "apl": 2, "dpm": 3, "mbon": 4, "pam": 5, "ppl1": 6}
    mb = mb.sort_values(["role", "type", "bodyId"], key=lambda s: s.map(order) if s.name == "role" else s).reset_index(drop=True)
    n = len(mb)
    print(f"Mushroom body circuit: {n} neurons")
    print(mb["role"].value_counts().to_string())
    print(f"  {mb.loc[mb.role == 'pn', 'glom'].nunique()} olfactory glomeruli among the PNs")

    weights_path = next((data_dir / c for c in WEIGHTS_CANDIDATES if (data_dir / c).exists()), None)
    if weights_path is None:
        raise SystemExit("No connectome weights file; run scripts/01_download_data.py")
    ids = pa.array(mb["bodyId"].tolist(), type=pa.int64())
    tab = feather.read_table(str(weights_path), columns=["body_pre", "body_post", "weight"])
    keep = pc.and_(pc.is_in(tab["body_pre"], value_set=ids), pc.is_in(tab["body_post"], value_set=ids))
    edges = tab.filter(keep).to_pandas()
    edges = edges[edges["body_pre"] != edges["body_post"]]
    idx = {b: i for i, b in enumerate(mb["bodyId"])}
    rows = edges["body_pre"].map(idx).to_numpy()
    cols = edges["body_post"].map(idx).to_numpy()
    W = sparse.csr_matrix((edges["weight"].to_numpy(dtype=np.float32), (rows, cols)), shape=(n, n))
    W.sum_duplicates()
    print(f"  {W.nnz:,} synaptic connections inside the circuit")

    role = mb["role"].to_numpy()
    R = {r: np.where(role == r)[0] for r in order}

    def block(a, b):
        return W[R[a]][:, R[b]]

    print("\nPathway synapse totals (rows -> cols):")
    for a, b in [("pn", "kc"), ("kc", "mbon"), ("kc", "apl"), ("apl", "kc"), ("pam", "mbon"), ("ppl1", "mbon"),
                 ("pam", "kc"), ("ppl1", "kc"), ("mbon", "pam"), ("mbon", "ppl1"), ("dpm", "kc"), ("kc", "dpm")]:
        print(f"  {a:>5s} -> {b:<5s} {block(a, b).sum():10,.0f}")
    fan_kc = np.asarray(block("pn", "kc").astype(bool).sum(axis=0)).ravel()
    print(f"\nPNs per Kenyon cell: median {np.median(fan_kc):.0f}, mean {fan_kc.mean():.1f}, "
          f"{(fan_kc == 0).sum()} KCs get no uniglomerular PN input")
    fan_mbon = np.asarray(block("kc", "mbon").astype(bool).sum(axis=0)).ravel()
    print(f"Kenyon cells per MBON: median {np.median(fan_mbon):.0f}, range {fan_mbon.min()}-{fan_mbon.max()}")

    # ---- MBON valence from the wiring ------------------------------------
    # Reward DANs (PAM) tile the compartments whose MBONs drive avoidance;
    # punishment DANs (PPL1) tile the compartments whose MBONs drive
    # approach. So: whichever class sends an MBON more synapses says which
    # compartment it lives in, and therefore what it drives.
    pam_in = np.asarray(block("pam", "mbon").sum(axis=0)).ravel()
    ppl_in = np.asarray(block("ppl1", "mbon").sum(axis=0)).ravel()
    valence = np.where(pam_in > ppl_in, "avoid", np.where(ppl_in > pam_in, "approach", "neutral"))
    print("\nMBON valence (from dominant dopamine input):")
    vt = pd.DataFrame({"type": mb.loc[R["mbon"], "type"].values, "valence": valence, "pam": pam_in, "ppl1": ppl_in})
    print(vt.groupby("type").agg(valence=("valence", "first"), n=("valence", "size"),
                                 pam=("pam", "sum"), ppl1=("ppl1", "sum")).to_string())
    print(pd.Series(valence).value_counts().to_string())

    # ---- transmitters -------------------------------------------------------
    nt = feather.read_table(str(data_dir / NT_FILE), columns=["body", "consensus_nt"]).to_pandas()
    nt = nt[nt["body"].isin(mb["bodyId"])]
    nt_map = dict(zip(nt["body"].astype("int64"), nt["consensus_nt"].astype(str).str.lower()))
    mb["nt"] = mb["bodyId"].map(nt_map).fillna("")
    mb["sign"] = mb["nt"].map(NT_SIGN).fillna(1).astype(int)
    print("\nTransmitter by role:")
    print(mb.groupby("role")["nt"].value_counts().to_string())

    # ---- save ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "adjacency.npz", W)
    val_by_idx = dict(zip(R["mbon"], valence))
    meta = []
    for i in range(n):
        meta.append({
            "bodyId": int(mb["bodyId"][i]),
            "type": str(mb["type"][i]),
            "class": str(mb["role"][i]),
            "role": str(mb["role"][i]),
            "side": str(mb["somaSide"][i]),
            "glom": str(mb["glom"][i]),
            "nt": str(mb["nt"][i]),
            "sign": int(mb["sign"][i]),
            "valence": val_by_idx.get(i, ""),
        })
    with open(out_dir / "neurons.json", "w") as f:
        json.dump(meta, f)
    gloms = sorted(set(mb.loc[mb.role == "pn", "glom"]))
    with open(out_dir / "glomeruli.json", "w") as f:
        json.dump(gloms, f)
    print(f"\nSaved {n} neurons / {W.nnz:,} connections / {len(gloms)} glomeruli to {out_dir}")
    return {"n": n, "nnz": int(W.nnz)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Extract the mushroom body learning circuit.")
    p.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR)
    p.add_argument("--out", type=pathlib.Path, default=None)
    a = p.parse_args(argv)
    build(a.data_dir, a.out or (a.data_dir / "mushroom"))


if __name__ == "__main__":
    main(sys.argv[1:])
