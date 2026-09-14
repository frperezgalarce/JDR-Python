"""Acquire a separate, reproducible 500-object OGLE Galactic Disk RR Lyrae sample."""

from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import io
import json
import tarfile
import urllib.request
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.read_data import read_ogle_dat

BASE = "https://www.astrouw.edu.pl/ogle/ogle4/OCVS/gd/rrlyr/"
SEED = 20260912


def main():
    destination = ROOT / "data" / "ogle_gd_500"
    source = destination / "source"
    phot = destination / "photometry"
    source.mkdir(parents=True, exist_ok=True)
    phot.mkdir(exist_ok=True)
    records = []
    for name in [
        "README",
        "ident.dat",
        "RRab.dat",
        "RRc.dat",
        "RRd.dat",
        "phot.tar.gz",
    ]:
        path = source / name
        if not path.exists():
            print("Downloading", name, flush=True)
            with urllib.request.urlopen(BASE + name, timeout=120) as response:
                body = response.read()
            if body[:100].lower().find(b"<html") >= 0:
                raise RuntimeError("Unexpected HTML response")
            path.write_bytes(body)
        records.append(
            {
                "file": str(path.relative_to(ROOT)),
                "url": BASE + name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    metadata = {}
    for line in (source / "ident.dat").read_text().splitlines():
        ident = line[:19].strip()
        kind = line[21:25].strip()
        if ident and kind in {"RRab", "RRc", "RRd"}:
            metadata[ident] = {"id": ident, "type": kind}
    for kind in ["RRab", "RRc", "RRd"]:
        for line in (source / f"{kind}.dat").read_text().splitlines():
            ident = line[:19].strip()
            if ident not in metadata:
                continue
            primary = float(line[36:46])
            secondary = float(line[104:114]) if kind == "RRd" else None
            metadata[ident].update(
                catalog_period=primary, catalog_secondary_period=secondary
            )
    # Read regular I-band members without extractall or trusting archive paths.
    raw = {}
    with tarfile.open(source / "phot.tar.gz", "r:gz") as archive:
        for member in archive:
            path = Path(member.name)
            if (
                member.isfile()
                and path.parent.name == "I"
                and path.stem in metadata
                and path.suffix == ".dat"
            ):
                stream = archive.extractfile(member)
                raw[path.stem] = stream.read()
    ordered = np.random.default_rng(SEED).permutation(sorted(raw)).tolist()
    selected = []
    excluded = []
    seen = set()
    for rank, ident in enumerate(ordered):
        body = raw[ident]
        digest = sha256(body).hexdigest()
        # Validate in memory through the same raw-observation reader used by analysis.
        try:
            frame = read_ogle_dat(io.BytesIO(body))
            if len(frame) < 30:
                raise ValueError(
                    "Fewer than 30 observations for periodic shape fitting."
                )
            if digest in seen:
                raise ValueError("Byte-identical light curve already selected.")
            if "catalog_period" not in metadata[ident]:
                raise ValueError("Catalog period unavailable.")
        except ValueError as exc:
            excluded.append(
                {"object_id": ident, "candidate_rank": rank, "reason": str(exc)}
            )
            continue
        path = phot / f"{ident}.dat"
        if path.exists() and path.read_bytes() != body:
            raise ValueError(f"Existing file changed: {path}")
        path.write_bytes(body)
        seen.add(digest)
        selected.append(
            {
                "object_id": ident,
                "catalog_type": metadata[ident]["type"],
                "region": "GD",
                "file": str(path.relative_to(ROOT)),
                "sha256": digest,
                "n_observations": len(frame),
                "baseline_days": float(np.ptp(frame.time)),
                "has_errors": "mag_err" in frame,
                "candidate_rank": rank,
                "catalog_period": metadata[ident]["catalog_period"],
                "catalog_secondary_period": metadata[ident]["catalog_secondary_period"],
            }
        )
        if len(selected) == 500:
            break
    if len(selected) != 500:
        raise RuntimeError(f"Only {len(selected)} eligible curves.")
    table = pd.DataFrame(selected).sort_values("object_id").reset_index(drop=True)
    table.to_csv(destination / "cohort.csv", index=False)
    pd.DataFrame(excluded, columns=["object_id", "candidate_rank", "reason"]).to_csv(
        destination / "eligibility_exclusions.csv", index=False
    )
    manifest = {
        "source": BASE,
        "citation": "Soszynski et al. (2019), Acta Astronomica 69, 321; arXiv:2001.00025",
        "seed": SEED,
        "selection": "Uniform seeded candidate permutation; first 500 passing predeclared input checks, then sorted by ID.",
        "criteria": [
            "Subtype RRab, RRc or RRd (aRRd excluded).",
            "I-band photometry with >=30 observations.",
            "Finite values, positive errors, distinct times, nonconstant signal.",
            "Catalog period available; no duplicate IDs/content.",
        ],
        "eligible_archive_candidates": len(raw),
        "candidates_examined": rank + 1,
        "excluded_before_500": len(excluded),
        "class_counts": table.catalog_type.value_counts().to_dict(),
        "source_files": records,
    }
    (destination / "acquisition.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "eligible_archive_candidates",
                    "candidates_examined",
                    "excluded_before_500",
                    "class_counts",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
