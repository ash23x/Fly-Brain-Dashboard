"""
Fetches the three MaleCNS v1.0 release files this project needs, straight
from the dataset's public Google Cloud Storage bucket (the location the
official download page at https://male-cns.janelia.org/download/ lists).
No account, no token: the data is CC-BY 4.0 and served over plain HTTPS.

    python scripts/01_download_data.py              # ~560 MB into ./data/
    python scripts/01_download_data.py --data-dir D:\somewhere\else

Interrupted transfers resume from the partial file; complete files are
skipped, so re-running is always safe.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import requests

BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",              # cell types, classes, sides   (~14 MB)
    "body-neurotransmitters-male-cns-v1.0.feather",                    # predicted transmitter per cell (~43 MB)
    "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",  # synapse counts, strong edges (~500 MB)
]
DEFAULT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
CHUNK = 1 << 20


def remote_size(url: str) -> int | None:
    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
        return int(r.headers["Content-Length"])
    except (requests.RequestException, KeyError, ValueError):
        return None


def fetch(url: str, dest: pathlib.Path, attempts: int = 5) -> None:
    total = remote_size(url)
    if dest.exists() and total is not None and dest.stat().st_size == total:
        print(f"  {dest.name}: already complete, skipping")
        return
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                if have and r.status_code != 206:      # server ignored the range: start over
                    have = 0
                r.raise_for_status()
                mode = "ab" if have else "wb"
                done = have
                t0 = time.time()
                with open(part, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100 * done / total
                            rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                            print(f"\r  {dest.name}: {pct:5.1f} %  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB  {rate:5.1f} MB/s", end="")
                print()
            if total is None or part.stat().st_size == total:
                part.replace(dest)
                return
            print(f"  short file ({part.stat().st_size} of {total} bytes), retrying")
        except (requests.RequestException, OSError) as exc:
            print(f"\n  attempt {attempt}/{attempts} failed: {exc}")
            time.sleep(min(30, 3 * attempt))
    raise SystemExit(f"Could not download {url}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download the MaleCNS v1.0 files (once).")
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DIR)
    a = p.parse_args(argv)
    a.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading into {a.data_dir}")
    for name in FILES:
        fetch(f"{BUCKET}/{name}", a.data_dir / name)
    print("Done. Next: python scripts/03_build_compass.py and python scripts/05_build_mushroom_body.py")


if __name__ == "__main__":
    main(sys.argv[1:])
