"""
Packs a circuit's SWC skeletons into two flat binary files the browser can
upload straight to the GPU, plus a small JSON of ranges and bounds.

  data/<circuit>/skeleton_pos.bin   float32 x,y,z per vertex (microns, centred), two vertices per line segment
  data/<circuit>/skeleton_idx.bin   uint16 neuron index per vertex (position in neurons.json)
  data/<circuit>/skeletons.json     counts, per-neuron vertex ranges, bounds

Skeletons are simplified on the way: a node is kept if it is a branch point,
a tip, or further than --step microns along the tree from the last kept
node. Topology is preserved; wiggles under the step are dropped.

    python scripts/07_build_skeletons.py compass
    python scripts/07_build_skeletons.py mushroom --step 3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "skeletons"
NM_PER_UNIT = 8.0            # the SWC coordinates are in 8 nm voxels
UM_PER_UNIT = NM_PER_UNIT / 1000.0


def load_swc(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ids, xyz in microns, parent ids) for one SWC file."""
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.size == 0:
        return np.zeros(0, int), np.zeros((0, 3)), np.zeros(0, int)
    ids = rows[:, 0].astype(np.int64)
    xyz = rows[:, 2:5] * UM_PER_UNIT
    parent = rows[:, 6].astype(np.int64)
    return ids, xyz, parent


def simplify(ids: np.ndarray, xyz: np.ndarray, parent: np.ndarray, step: float) -> np.ndarray:
    """Line segments (n, 2, 3) after dropping nodes closer than `step` along the tree."""
    if len(ids) == 0:
        return np.zeros((0, 2, 3))
    pos = {int(i): k for k, i in enumerate(ids)}
    children = np.zeros(len(ids), dtype=np.int32)
    for p in parent:
        if p != -1 and int(p) in pos:
            children[pos[int(p)]] += 1
    anchor = {}          # node id -> id of the last kept node on the path to the root
    segs = []
    # SWC files list parents before children in practice; sort by id to be safe.
    order = np.argsort(ids)
    for k in order:
        nid = int(ids[k]); pid = int(parent[k])
        if pid == -1 or pid not in pos:
            anchor[nid] = nid
            continue
        a = anchor.get(pid, pid)
        d = float(np.linalg.norm(xyz[k] - xyz[pos[a]]))
        keep = d >= step or children[k] != 1
        if keep:
            segs.append((xyz[pos[a]], xyz[k]))
            anchor[nid] = nid
        else:
            anchor[nid] = a
    return np.array(segs) if segs else np.zeros((0, 2, 3))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Pack SWC skeletons for the browser.")
    p.add_argument("circuit", choices=["compass", "mushroom"])
    p.add_argument("--step", type=float, default=1.5, help="simplification step in microns (default 1.5)")
    a = p.parse_args(argv)
    circuit_dir = ROOT / "data" / a.circuit
    with open(circuit_dir / "neurons.json") as f:
        neurons = json.load(f)

    all_pos, all_idx, ranges = [], [], []
    missing = 0
    total_nodes = 0
    for i, m in enumerate(neurons):
        path = CACHE / f"{m['bodyId']}.swc"
        if not path.exists():
            missing += 1
            ranges.append([0, 0])
            continue
        ids, xyz, parent = load_swc(path)
        total_nodes += len(ids)
        segs = simplify(ids, xyz, parent, a.step)
        start = sum(len(x) for x in all_pos)
        all_pos.append(segs.reshape(-1, 3).astype(np.float32))
        all_idx.append(np.full(len(segs) * 2, i, dtype=np.uint16))
        ranges.append([start, len(segs) * 2])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(neurons)}", flush=True)

    pos = np.concatenate(all_pos) if all_pos else np.zeros((0, 3), np.float32)
    idx = np.concatenate(all_idx) if all_idx else np.zeros(0, np.uint16)
    centre = (pos.min(axis=0) + pos.max(axis=0)) / 2 if len(pos) else np.zeros(3)
    pos = (pos - centre).astype(np.float32)
    lo, hi = (pos.min(axis=0).tolist(), pos.max(axis=0).tolist()) if len(pos) else ([0, 0, 0], [0, 0, 0])

    pos.tofile(circuit_dir / "skeleton_pos.bin")
    idx.tofile(circuit_dir / "skeleton_idx.bin")
    meta = {"n_neurons": len(neurons), "n_vertices": int(len(pos)), "n_segments": int(len(pos) // 2),
            "ranges": ranges, "bounds": {"min": lo, "max": hi}, "units": "microns, centred",
            "step_um": a.step, "source_nodes": int(total_nodes), "missing_skeletons": missing}
    with open(circuit_dir / "skeletons.json", "w") as f:
        json.dump(meta, f)
    print(f"{a.circuit}: {len(neurons)} neurons, {total_nodes:,} SWC nodes -> {meta['n_segments']:,} segments "
          f"({pos.nbytes / 1e6:.1f} MB positions), {missing} missing. Extent "
          f"{hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} x {hi[2] - lo[2]:.0f} um")


if __name__ == "__main__":
    main(sys.argv[1:])
