"""
Fetches the 3D skeleton (SWC) of every neuron in a built circuit, from the
same public Janelia bucket the connectome data came from -- no account, no
token. Files are cached in data/skeletons/<bodyId>.swc and shared between
circuits, so a neuron is only ever downloaded once.

    python scripts/06_fetch_skeletons.py compass       # 152 files, ~12 MB
    python scripts/06_fetch_skeletons.py mushroom      # 4,733 files, a few hundred MB

Then: python scripts/07_build_skeletons.py <circuit>
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc"
CACHE = ROOT / "data" / "skeletons"


def fetch_one(session: requests.Session, body_id: int, attempts: int = 4) -> tuple[int, str]:
    dest = CACHE / f"{body_id}.swc"
    if dest.exists() and dest.stat().st_size > 0:
        return body_id, "cached"
    url = f"{BUCKET}/{body_id}.swc"
    for attempt in range(1, attempts + 1):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                return body_id, "missing"
            r.raise_for_status()
            dest.write_bytes(r.content)
            return body_id, "fetched"
        except requests.RequestException as exc:
            if attempt == attempts:
                return body_id, f"failed: {exc}"
            time.sleep(2 * attempt)
    return body_id, "failed"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download SWC skeletons for a built circuit.")
    p.add_argument("circuit", choices=["compass", "mushroom"])
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)
    circuit_dir = ROOT / "data" / a.circuit
    with open(circuit_dir / "neurons.json") as f:
        ids = [int(m["bodyId"]) for m in json.load(f)]
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"{a.circuit}: {len(ids)} neurons -> {CACHE}")
    counts = {"cached": 0, "fetched": 0, "missing": 0, "failed": 0}
    t0 = time.time()
    with requests.Session() as s, cf.ThreadPoolExecutor(a.workers) as pool:
        for i, (bid, status) in enumerate(pool.map(lambda b: fetch_one(s, b), ids), 1):
            counts["failed" if status.startswith("failed") else status] += 1
            if status.startswith("failed") or status == "missing":
                print(f"  {bid}: {status}")
            if i % 200 == 0 or i == len(ids):
                print(f"  {i}/{len(ids)}  {counts}  {time.time() - t0:.0f} s", flush=True)
    total = sum(f.stat().st_size for f in CACHE.glob("*.swc"))
    print(f"Done: {counts}. Cache now {total / 1e6:,.0f} MB. Next: python scripts/07_build_skeletons.py {a.circuit}")


if __name__ == "__main__":
    main(sys.argv[1:])
