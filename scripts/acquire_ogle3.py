"""Acquire original OGLE-III RR Lyrae sources; streaming, atomic downloads."""

from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import argparse, json, urllib.request, shutil
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ogle3_rrlyrae_all"
BASE = "https://ftp.astrouw.edu.pl/ogle/ogle3/OIII-CVS/"


def fetch(region, name):
    path = DATA / "source" / region / name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + region + "/rrlyr/" + name
    if not path.exists():
        print("Downloading", region, name, flush=True)
        temporary = path.with_suffix(path.suffix + ".part")
        with (
            urllib.request.urlopen(url, timeout=180) as response,
            temporary.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        with temporary.open("rb") as stream:
            prefix = stream.read(100).lower()
        if b"<html" in prefix or b"<!doctype" in prefix:
            raise RuntimeError("Unexpected HTML response: " + url)
        temporary.replace(path)
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(
        region=region,
        file=str(path.relative_to(DATA)),
        url=url,
        bytes=path.stat().st_size,
        sha256=digest.hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", action="store_true")
    args = parser.parse_args()
    tasks = [
        (r, n)
        for r in ["blg", "lmc", "smc"]
        for n in ["README", "ident.dat", "RRab.dat", "RRc.dat", "RRd.dat"]
        + (["RRe.dat"] if r != "blg" else [])
        + (["phot.tar.gz"] if args.archives else [])
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda args: fetch(*args), tasks))
    manifest = dict(
        survey="OGLE-III original OIII-CVS, not OGLE-IV OCVS",
        retrieved_utc=datetime.now(timezone.utc).isoformat(),
        files=records,
    )
    (
        DATA / ("sources_complete.json" if args.archives else "sources_metadata.json")
    ).write_text(json.dumps(manifest, indent=2) + "\n")
    for r in ["blg", "lmc", "smc"]:
        lines = (DATA / "source" / r / "ident.dat").read_text().splitlines()
        print(r, len(lines), "identification rows", flush=True)


if __name__ == "__main__":
    main()
