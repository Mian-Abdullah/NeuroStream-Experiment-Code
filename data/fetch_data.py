"""Download the exact dataset files used in the paper and verify their SHA-256.

The four datasets are public and are not redistributed in this repository. This script fetches the files from the
GitHub mirrors and commits listed in data/PROVENANCE.md, writes them to data/raw/ (or $NS_DATA), and stops with an
error if any checksum differs from data/SHA256SUMS.mirror. Files that are already present and valid are skipped.

Official sources (cite these; see the paper's Code and Data Availability section):
  Elec2      https://www.openml.org/d/151              (MOA: https://moa.cms.waikato.ac.nz/datasets/)
  Covertype  https://archive.ics.uci.edu/dataset/31/covertype   doi:10.24432/C50K5N
  Airlines   https://www.openml.org/d/1169             (MOA: https://moa.cms.waikato.ac.nz/datasets/)
  C-MAPSS    https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Usage:
  python data/fetch_data.py            # download missing files, verify all
  python data/fetch_data.py --verify   # verify only, download nothing
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.environ.get("NS_DATA", os.path.join(HERE, "raw"))

MIRRORS = {
    # file prefix in SHA256SUMS.mirror -> raw URL root at the pinned commit
    "": "https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/"
        "f1eb9f068c3450233721c3beac6fb0e880e9e7d7/",
    "cmapss/": "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/"
               "ed40e40e38f95fcfb84a8f48972a3f22a46fbb1d/",
}


def expected():
    out = {}
    with open(os.path.join(HERE, "SHA256SUMS.mirror")) as f:
        for line in f:
            if line.strip():
                h, name = line.split()
                out[name] = h
    return out


def url_of(name):
    if name.startswith("cmapss/"):
        return MIRRORS["cmapss/"] + name[len("cmapss/"):]
    return MIRRORS[""] + name


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path, tries=3):
    tmp = path + ".part"
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, path)
            return
        except Exception as e:  # network errors: retry, then report
            if k == tries - 1:
                raise RuntimeError(f"download failed for {url}: {e}")
            time.sleep(2 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="only verify files already in the data folder")
    args = ap.parse_args()
    bad = []
    for name, h in expected().items():
        path = os.path.join(DEST, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and sha256(path) == h:
            print(f"ok        {name}")
            continue
        if args.verify:
            bad.append(name)
            print(f"MISSING   {name}" if not os.path.exists(path) else f"MISMATCH  {name}")
            continue
        print(f"download  {name}")
        download(url_of(name), path)
        if sha256(path) != h:
            bad.append(name)
            print(f"MISMATCH  {name}")
    if bad:
        sys.exit(f"{len(bad)} file(s) missing or with a wrong checksum: {', '.join(bad)}")
    print(f"All {len(expected())} files verified in {DEST}")


if __name__ == "__main__":
    main()
