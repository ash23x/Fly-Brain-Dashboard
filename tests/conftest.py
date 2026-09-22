"""Test fixtures: tiny synthetic circuits so the simulators can be exercised
without the 560 MB dataset. They are structurally honest (same roles,
fields and files the real build scripts write) but numerically made up."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest
from scipy import sparse

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))


def _save(path: pathlib.Path, W: np.ndarray, meta: list[dict]) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path / "adjacency.npz", sparse.csr_matrix(W.astype(np.float32)))
    with open(path / "neurons.json", "w") as f:
        json.dump(meta, f)
    return path


@pytest.fixture
def compass_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """18 wedges x 2 EPG, one PEN per wedge (sides alternate), one Delta7 per wedge."""
    wedges = 18
    meta, W_rows = [], []
    epg = [[] for _ in range(wedges)]
    idx = 0
    for w in range(wedges):
        for _ in range(2):
            meta.append({"bodyId": idx, "type": "EPG", "role": "epg", "side": "L", "pb_side": "L",
                         "glom": f"L{w}", "nt": "acetylcholine", "sign": 1, "angle": w * 20.0})
            epg[w].append(idx); idx += 1
    pen = []
    for w in range(wedges):
        side = "L" if w % 2 == 0 else "R"
        meta.append({"bodyId": idx, "type": "PEN_a(PEN1)", "role": "pen", "side": side, "pb_side": side,
                     "glom": f"{side}{w}", "nt": "acetylcholine", "sign": 1, "angle": w * 20.0})
        pen.append(idx); idx += 1
    d7 = []
    for w in range(wedges):
        meta.append({"bodyId": idx, "type": "Delta7", "role": "delta7", "side": "L", "pb_side": "L",
                     "glom": f"L{w}", "nt": "glutamate", "sign": -1, "angle": w * 20.0})
        d7.append(idx); idx += 1
    n = idx
    W = np.zeros((n, n))
    for w in range(wedges):
        for a in epg[w]:
            for b in epg[w]:
                if a != b:
                    W[a, b] = 30
            W[a, pen[w]] = 40
            shift = 1 if w % 2 == 0 else -1
            for b in epg[(w + shift) % wedges]:
                W[pen[w], b] = 50
            for v in range(wedges):
                d = abs(((v - w + wedges // 2) % wedges) - wedges // 2)
                W[a, d7[v]] = 5 + 15 * d / (wedges // 2)      # Delta7 hears the far side of the ring
            W[d7[w], a] = 20
    return _save(tmp_path / "compass", W, meta)


@pytest.fixture
def mushroom_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """4 glomeruli x 2 PN, 60 KC, 1 APL, 6 MBON (3 approach / 3 avoid), 3 PAM, 3 PPL1."""
    rng = np.random.default_rng(1)
    meta = []
    gloms = ["DA1", "DM2", "VA1v", "DL5"]
    for g in gloms:
        for _ in range(2):
            meta.append({"type": f"{g}_adPN", "role": "pn", "glom": g, "nt": "acetylcholine", "sign": 1, "valence": ""})
    for _ in range(60):
        meta.append({"type": "KCg-m", "role": "kc", "glom": "", "nt": "acetylcholine", "sign": 1, "valence": ""})
    meta.append({"type": "APL", "role": "apl", "glom": "", "nt": "gaba", "sign": -1, "valence": ""})
    for i in range(6):
        meta.append({"type": f"MBON0{i + 1}", "role": "mbon", "glom": "", "nt": "acetylcholine", "sign": 1,
                     "valence": "approach" if i < 3 else "avoid"})
    for i in range(3):
        meta.append({"type": f"PAM0{i + 1}", "role": "pam", "glom": "", "nt": "dopamine", "sign": 1, "valence": ""})
    for i in range(3):
        meta.append({"type": f"PPL10{i + 1}", "role": "ppl1", "glom": "", "nt": "dopamine", "sign": 1, "valence": ""})
    for k, m in enumerate(meta):
        m.update(bodyId=k, side="L", pb_side="")
    role = np.array([m["role"] for m in meta])
    R = {r: np.where(role == r)[0] for r in ("pn", "kc", "apl", "mbon", "pam", "ppl1")}
    n = len(meta)
    W = np.zeros((n, n))
    for k in R["kc"]:
        for p in rng.choice(R["pn"], size=3, replace=False):
            W[p, k] = rng.integers(5, 30)
        W[k, R["apl"][0]] = 5
        W[R["apl"][0], k] = 8
        for m in R["mbon"]:
            if rng.random() < 0.6:
                W[k, m] = rng.integers(2, 12)
    for m in R["mbon"][:3]:                 # approach MBONs: taught by PPL1
        for d in R["ppl1"]:
            W[d, m] = 20
    for m in R["mbon"][3:]:                 # avoid MBONs: taught by PAM
        for d in R["pam"]:
            W[d, m] = 20
    path = _save(tmp_path / "mushroom", W, meta)
    with open(path / "glomeruli.json", "w") as f:
        json.dump(gloms, f)
    return path
