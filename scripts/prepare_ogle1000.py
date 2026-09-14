"""Extend the frozen OGLE sample to 1,000 using the same candidate permutation."""

from pathlib import Path
from hashlib import sha256
import io
import json
import tarfile
import shutil
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.read_data import read_ogle_dat


def main():
    previous = ROOT / "data" / "ogle_gd_500"
    destination = ROOT / "data" / "ogle_gd_1000"
    source = destination / "source"
    phot = destination / "photometry"
    source.mkdir(parents=True, exist_ok=True)
    phot.mkdir(exist_ok=True)
    prior = json.loads((previous / "acquisition.json").read_text())
    records = []
    for record in prior["source_files"]:
        origin = ROOT / record["file"]
        if sha256(origin.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("The frozen source archive/catalog has changed.")
        target = source / origin.name
        if (
            target.exists()
            and sha256(target.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError("Existing 1000-object source differs.")
        if not target.exists():
            shutil.copy2(origin, target)
        records.append({**record, "file": str(target.relative_to(ROOT))})
    metadata = {}
    for line in (source / "ident.dat").read_text().splitlines():
        ident, kind = line[:19].strip(), line[21:25].strip()
        if kind in {"RRab", "RRc", "RRd"}:
            metadata[ident] = {"catalog_type": kind}
    for kind in ["RRab", "RRc", "RRd"]:
        for line in (source / f"{kind}.dat").read_text().splitlines():
            ident = line[:19].strip()
            if ident not in metadata:
                continue
            metadata[ident].update(
                catalog_period=float(line[36:46]),
                catalog_epoch=float(line[59:69]),
                catalog_secondary_period=(
                    float(line[104:114]) if kind == "RRd" else None
                ),
                catalog_secondary_epoch=float(line[127:137]) if kind == "RRd" else None,
                period_mode=(
                    "first overtone" if kind in {"RRc", "RRd"} else "fundamental"
                ),
            )
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
                if path.stem in raw:
                    raise ValueError("Duplicate I-band archive member.")
                raw[path.stem] = archive.extractfile(member).read()
    ordered = np.random.default_rng(prior["seed"]).permutation(sorted(raw)).tolist()
    selected, excluded, seen = [], [], set()
    for rank, ident in enumerate(ordered):
        body = raw[ident]
        digest = sha256(body).hexdigest()
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
            if (
                not np.isfinite(
                    [
                        metadata[ident]["catalog_period"],
                        metadata[ident]["catalog_epoch"],
                    ]
                ).all()
                or metadata[ident]["catalog_period"] <= 0
            ):
                raise ValueError("Invalid catalog period or epoch.")
        except ValueError as exc:
            excluded.append(
                {"object_id": ident, "candidate_rank": rank, "reason": str(exc)}
            )
            continue
        target = phot / f"{ident}.dat"
        if target.exists() and target.read_bytes() != body:
            raise ValueError("Selected file changed.")
        target.write_bytes(body)
        seen.add(digest)
        selected.append(
            {
                "object_id": ident,
                **metadata[ident],
                "region": "GD",
                "file": str(target.relative_to(ROOT)),
                "sha256": digest,
                "n_observations": len(frame),
                "baseline_days": float(np.ptp(frame.time)),
                "has_errors": "mag_err" in frame,
                "candidate_rank": rank,
            }
        )
        if len(selected) == 1000:
            break
    if len(selected) != 1000:
        raise ValueError("Insufficient eligible observations.")
    table = pd.DataFrame(selected).sort_values("object_id").reset_index(drop=True)
    previous_ids = set(pd.read_csv(previous / "cohort.csv").object_id)
    if not previous_ids.issubset(set(table.object_id)):
        raise AssertionError("1000 sample must contain the previous 500.")
    table.to_csv(destination / "cohort.csv", index=False)
    pd.DataFrame(excluded).to_csv(
        destination / "eligibility_exclusions.csv", index=False
    )
    manifest = {
        **prior,
        "source_files": records,
        "selection": "Same seeded permutation as the 500-object sample; first 1000 eligible objects, sorted by ID. Catalog epochs additionally parsed for phase folding.",
        "candidates_examined": rank + 1,
        "excluded_before_1000": len(excluded),
        "n_objects": 1000,
        "class_counts": table.catalog_type.value_counts().to_dict(),
        "contains_previous_500": True,
        "period_policy": "Use catalog period and epoch directly; RRd uses the catalog first-overtone mode. No period search or catalog-label optimization.",
    }
    manifest.pop("excluded_before_500", None)
    (destination / "acquisition.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "n_objects",
                    "class_counts",
                    "candidates_examined",
                    "excluded_before_1000",
                    "contains_previous_500",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
