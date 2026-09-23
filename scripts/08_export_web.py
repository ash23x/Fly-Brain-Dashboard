"""
Exports a circuit for the browser-only build (web/): the wired-up, normalised
matrices exactly as the Python simulators construct them, the page metadata
the /neurons route would serve, and the packed skeletons with positions
quantised to int16 (half the bytes, 0.01 um resolution).

    python scripts/08_export_web.py compass
    python scripts/08_export_web.py mushroom

Writes web/data/<circuit>/meta.json, circuit.bin, skeletons.json,
skeleton_pos.bin. The JavaScript simulators in web/sim/
are step-for-step ports of server/*.py that read these files, so the fly
that runs in a visitor's tab is the same fly that runs on localhost.

circuit.bin layout:  b"FLYB" | uint32 header length | JSON header (utf-8, padded
to 4 bytes) | the arrays, each 4-byte aligned. The header names every array
with its dtype, byte offset, length and shape, and carries the tuned constants.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from compass_sim import CompassSim  # noqa: E402
from mushroom_sim import MushroomSim  # noqa: E402
from sim_server import read_params  # noqa: E402

DTYPES = {np.dtype("float32"): "float32", np.dtype("int32"): "int32", np.dtype("uint16"): "uint16",
          np.dtype("int16"): "int16", np.dtype("uint8"): "uint8"}


def pack(path: pathlib.Path, arrays: dict[str, np.ndarray], constants: dict) -> int:
    header: dict = {"constants": constants, "arrays": {}}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype not in DTYPES:
            raise TypeError(f"{name}: unsupported dtype {arr.dtype}")
        header["arrays"][name] = {"dtype": DTYPES[arr.dtype], "offset": offset, "length": int(arr.size), "shape": list(arr.shape)}
        b = arr.tobytes()
        pad = (-len(b)) % 4
        blobs.append(b + b"\0" * pad)
        offset += len(b) + pad
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hjson += b" " * ((-len(hjson)) % 4)
    with open(path, "wb") as f:
        f.write(b"FLYB")
        f.write(np.array([len(hjson)], dtype=np.uint32).tobytes())
        f.write(hjson)
        for b in blobs:
            f.write(b)
    return 8 + len(hjson) + offset


def csr_arrays(prefix: str, m) -> dict[str, np.ndarray]:
    m = m.tocsr()
    m.eliminate_zeros()
    m.sort_indices()
    idx_t = np.uint16 if m.shape[1] < 65536 else np.int32
    return {f"{prefix}_indptr": m.indptr.astype(np.int32), f"{prefix}_indices": m.indices.astype(idx_t),
            f"{prefix}_data": m.data.astype(np.float32)}


def export_compass(out: pathlib.Path) -> None:
    src = ROOT / "data" / "compass"
    sim = CompassSim(src, **read_params(src))
    meta = {"mode": "compass", "neurons": sim.neurons, "n": sim.n, "n_edges": sim.nnz, "step_hz": 90, "frame_hz": 30,
            "n_epg": len(sim.epg_idx), "n_pen": len(sim.pen_idx), "n_peg": len(sim.peg_idx), "n_delta7": len(sim.d7_idx),
            "n_inhibitory": sim.n_inhibitory,
            "params": {k: getattr(sim, k) for k in ("gE", "gI", "epg_drive", "pen_drive", "turn_gain", "turn_tau", "noise_std")}}
    constants = {k: float(getattr(sim, k)) for k in ("tau_m", "tau_s", "gE", "gI", "epg_drive", "pen_drive", "turn_gain",
                                                      "turn_tau", "noise_std", "threshold_jitter", "rate_alpha")}
    constants.update(reignite_after=int(sim.reignite_after), refractory_steps=int(sim.refractory_steps), n=int(sim.n))
    arrays = {
        "W_E": sim.W_E.astype(np.float32), "W_I": sim.W_I.astype(np.float32),        # rows = presynaptic
        "epg_idx": sim.epg_idx.astype(np.int32), "pen_idx": sim.pen_idx.astype(np.int32),
        "pen_left": sim.pen_left.astype(np.int32), "pen_right": sim.pen_right.astype(np.int32),
        "d7_idx": sim.d7_idx.astype(np.int32), "peg_idx": sim.peg_idx.astype(np.int32),
        "epg_angle": sim.epg_angle.astype(np.float32),                                 # radians, EPG order
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    size = pack(out / "circuit.bin", arrays, constants)
    print(f"compass: {sim.n} neurons, W_E/W_I dense {sim.n}x{sim.n}, circuit.bin {size / 1e6:.2f} MB")


def export_mushroom(out: pathlib.Path) -> None:
    src = ROOT / "data" / "mushroom"
    sim = MushroomSim(src, **read_params(src))
    meta = {"mode": "mushroom", "neurons": sim.neurons, "n": sim.n, "n_edges": sim.nnz, "step_hz": 90, "frame_hz": 30,
            "counts": {r: int(len(sim.idx[r])) for r in sim.idx}, "n_plastic": sim.n_plastic,
            "glomeruli": sim.glomeruli, "odours": sim.odours,
            "mbon_valence": [sim.neurons[i]["valence"] for i in sim.idx["mbon"]],
            "mbon_types": [sim.neurons[i]["type"] for i in sim.idx["mbon"]]}
    constants = {k: float(getattr(sim, k)) for k in ("tau_m", "tau_s", "odour_drive", "dan_drive", "noise_std",
                                                      "threshold_jitter", "eta", "f_min", "tau_forget", "rate_alpha")}
    constants.update(refractory_steps=int(sim.refractory_steps), n=int(sim.n), n_kc=int(len(sim.idx["kc"])))
    arrays = {}
    arrays.update(csr_arrays("WT_E", sim.WT_E))
    arrays.update(csr_arrays("WT_I", sim.WT_I))
    # Dopamine only matters where it lands on an MBON (the plastic rule reads da[plastic_mbon]):
    # keep just those rows, so the browser's matvec is a fraction of the size.
    keep = np.zeros(sim.n); keep[sim.idx["mbon"]] = 1.0
    arrays.update(csr_arrays("WT_da", sim.WT_da.multiply(keep[:, None]).tocsr()))
    # WT_E.data is addressed by plastic_pos: after sort_indices() above the order must still match.
    # MushroomSim already sorted indices before taking plastic_pos, so the positions are stable; verify.
    coo = sim.WT_E.tocoo()
    assert np.array_equal(coo.row[sim.plastic_pos], sim.plastic_mbon) and np.array_equal(coo.col[sim.plastic_pos], sim.plastic_kc)
    arrays.update({
        "plastic_pos": sim.plastic_pos.astype(np.int32), "plastic_mbon": sim.plastic_mbon.astype(np.uint16),
        "plastic_kc": sim.plastic_kc.astype(np.uint16), "base_w": sim.base_w.astype(np.float32),
        "gE_vec": sim.gE_vec.astype(np.float32), "gI_vec": sim.gI_vec.astype(np.float32),
        "pam_idx": sim.idx["pam"].astype(np.int32), "ppl1_idx": sim.idx["ppl1"].astype(np.int32),
        "kc_idx": sim.idx["kc"].astype(np.int32), "mbon_idx": sim.idx["mbon"].astype(np.int32),
        "mbon_approach": sim.mbon_approach.astype(np.int32), "mbon_avoid": sim.mbon_avoid.astype(np.int32),
    })
    for name, gloms in sim.odours.items():
        pns = np.concatenate([sim.pn_by_glom[g] for g in gloms])
        arrays[f"odour_{name}"] = np.sort(pns).astype(np.int32)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    size = pack(out / "circuit.bin", arrays, constants)
    print(f"mushroom: {sim.n} neurons, WT_E {sim.WT_E.nnz:,} / WT_I {sim.WT_I.nnz:,} / WT_da (MBON rows) {int(arrays['WT_da_data'].size):,} entries, "
          f"{sim.n_plastic:,} plastic, circuit.bin {size / 1e6:.2f} MB")


def resimplify(segs: np.ndarray, step: float) -> np.ndarray:
    """Coarsen one neuron's packed line segments (k, 2, 3): rebuild the tree, keep
    branch points and tips, and along each chain keep a vertex every `step` um."""
    if len(segs) == 0 or step <= 0:
        return segs
    key = np.round(segs.reshape(-1, 3) / 1e-3).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.reshape(-1, 2)
    xyz = np.zeros((vid.max() + 1, 3)); xyz[vid.ravel()] = segs.reshape(-1, 3)
    nbr: dict[int, list[int]] = {}
    for a, b in vid:
        if a == b:
            continue
        nbr.setdefault(int(a), []).append(int(b)); nbr.setdefault(int(b), []).append(int(a))
    out: list[tuple] = []
    seen: set[tuple[int, int]] = set()
    anchors = [v for v, ns in nbr.items() if len(ns) != 2]
    if not anchors:                       # a pure loop (should not happen in a tree): keep as is
        return segs
    for start in anchors:
        for first in nbr[start]:
            if (start, first) in seen:
                continue
            prev, cur, last_kept, dist = start, first, start, 0.0
            seen.add((start, first)); seen.add((first, start))
            while True:
                dist += float(np.linalg.norm(xyz[cur] - xyz[prev]))
                ns = nbr[cur]
                end = len(ns) != 2
                if end or dist >= step:
                    out.append((xyz[last_kept], xyz[cur])); last_kept = cur; dist = 0.0
                if end:
                    break
                nxt = ns[0] if ns[0] != prev else ns[1]
                seen.add((cur, nxt)); seen.add((nxt, cur))
                prev, cur = cur, nxt
    return np.array(out) if out else np.zeros((0, 2, 3))


def export_skeletons(circuit: str, out: pathlib.Path, step: float) -> None:
    src = ROOT / "data" / circuit
    if not (src / "skeletons.json").exists():
        print(f"{circuit}: no skeletons built (run scripts/06 and 07) -- the 3D panel will say so")
        return
    with open(src / "skeletons.json") as f:
        meta = json.load(f)
    pos = np.fromfile(src / "skeleton_pos.bin", dtype=np.float32).reshape(-1, 3)
    if step > meta.get("step_um", 0):
        parts, ranges = [], []
        start = 0
        for r0, cnt in meta["ranges"]:
            segs = resimplify(pos[r0:r0 + cnt].reshape(-1, 2, 3), step)
            parts.append(segs.reshape(-1, 3).astype(np.float32))
            ranges.append([start, int(len(segs) * 2)])
            start += int(len(segs) * 2)
        pos = np.concatenate(parts) if parts else pos[:0]
        meta.update(ranges=ranges, n_vertices=int(len(pos)), n_segments=int(len(pos) // 2), step_um=step)
    flat = pos.ravel()
    scale = float(np.abs(flat).max()) / 32767.0 if flat.size else 1.0
    q = np.round(flat / scale).astype(np.int16)
    q.tofile(out / "skeleton_pos.bin")
    # no skeleton_idx.bin: the browser rebuilds the per-vertex neuron index from `ranges`
    meta.update(pos_format="int16", pos_scale=scale)
    with open(out / "skeletons.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    print(f"{circuit}: skeletons {meta['n_segments']:,} segments at {meta['step_um']} um, positions int16 at "
          f"{scale * 1000:.1f} nm ({q.nbytes / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export a circuit for the browser-only build.")
    p.add_argument("circuit", choices=["compass", "mushroom"])
    p.add_argument("--step", type=float, default=None,
                   help="re-simplify skeletons to this step in microns for the web (default: compass as built, mushroom 4)")
    a = p.parse_args(argv)
    step = a.step if a.step is not None else (4.0 if a.circuit == "mushroom" else 0.0)
    out = ROOT / "web" / "data" / a.circuit
    out.mkdir(parents=True, exist_ok=True)
    (export_compass if a.circuit == "compass" else export_mushroom)(out)
    export_skeletons(a.circuit, out, step)


if __name__ == "__main__":
    main(sys.argv[1:])
