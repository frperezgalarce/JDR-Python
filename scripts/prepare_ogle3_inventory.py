"""Inventory every original OGLE-III catalog ID, retaining original I-band bytes."""

from pathlib import Path
from hashlib import sha256
import argparse, io, json, tarfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ogle3_rrlyrae_all"
REGIONS = ["blg", "lmc", "smc"]
TYPES = ["RRab", "RRc", "RRd", "RRe"]


def catalog_records(data_dir):
    rows = []
    for region in REGIONS:
        folder = Path(data_dir) / "source" / region
        id_width = 19 if region == "smc" else 20
        type_start = 37 if region == "smc" else 38
        parameters = {}
        for kind in TYPES:
            path = folder / f"{kind}.dat"
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                ident = line[:id_width].strip()
                p_slice, e_slice = (
                    ((37, 47), (60, 70)) if region == "blg" else ((35, 45), (57, 67))
                )
                p = float(line[slice(*p_slice)])
                epoch = float(line[slice(*e_slice)])
                secondary = (
                    float(line[105:115] if region == "blg" else line[101:111])
                    if kind == "RRd"
                    else np.nan
                )
                if ident in parameters:
                    raise ValueError("Duplicate parameter ID: " + ident)
                parameters[ident] = (p, epoch, secondary, kind)
        for line in (folder / "ident.dat").read_text().splitlines():
            ident = line[:id_width].strip()
            kind = line[type_start : type_start + 4].strip()
            if kind not in TYPES:
                raise ValueError("Unrecognized subtype: " + kind)
            values = parameters.get(ident, (np.nan, np.nan, np.nan, kind))
            if values[3] != kind:
                raise ValueError("Conflicting catalog subtype: " + ident)
            rows.append(
                dict(
                    object_id=ident,
                    region=region.upper(),
                    catalog_type=kind,
                    catalog_period=values[0],
                    catalog_epoch=values[1],
                    catalog_secondary_period=values[2],
                    period_mode=(
                        "first overtone"
                        if kind in ["RRc", "RRd"]
                        else "catalog RRe" if kind == "RRe" else "fundamental"
                    ),
                )
            )
    table = pd.DataFrame(rows).sort_values("object_id").reset_index(drop=True)
    if not table.object_id.is_unique:
        raise ValueError("Duplicate catalog identification.")
    return table


def prepare(data_dir=DATA):
    data_dir = Path(data_dir)
    table = catalog_records(data_dir)
    phot = data_dir / "photometry"
    phot.mkdir(exist_ok=True)
    records = json.loads((data_dir / "sources_complete.json").read_text())["files"]
    for record in records:
        p = data_dir / record["file"]
        h = sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != record["sha256"]:
            raise ValueError("Source hash mismatch: " + record["file"])
    known = set(table.object_id)
    seen_members = set()
    extra = []
    for region in REGIONS:
        with tarfile.open(
            data_dir / "source" / region / "phot.tar.gz", "r:gz"
        ) as archive:
            for member in archive:
                p = Path(member.name)
                if not member.isfile() or p.parent.name != "I" or p.suffix != ".dat":
                    continue
                ident = p.stem
                if ident not in known:
                    extra.append(
                        dict(
                            region=region,
                            archive_member=member.name,
                            reason="I-band member has no current OGLE-III catalog ID",
                        )
                    )
                    continue
                if ident in seen_members:
                    raise ValueError("Duplicate I-band archive ID: " + ident)
                seen_members.add(ident)
                body = archive.extractfile(member).read()
                target = phot / p.name
                if target.exists() and target.read_bytes() != body:
                    raise ValueError("Existing raw file differs: " + ident)
                target.write_bytes(body)
        print("Extracted original I-band data:", region, flush=True)
    inventory = []
    content_seen = {}
    for index, row in enumerate(table.to_dict("records")):
        path = phot / f"{row['object_id']}.dat"
        reasons = []
        metadata = {}
        if (
            not np.isfinite([row["catalog_period"], row["catalog_epoch"]]).all()
            or row["catalog_period"] <= 0
        ):
            reasons.append("Missing or invalid catalog period/epoch")
        if not path.exists():
            reasons.append("No I-band file in official archive")
        else:
            body = path.read_bytes()
            digest = sha256(body).hexdigest()
            metadata.update(
                sha256=digest, file=f"photometry/{path.name}", bytes=len(body)
            )
            if digest in content_seen:
                reasons.append("Byte-identical content to " + content_seen[digest])
            else:
                content_seen[digest] = row["object_id"]
            try:
                raw = np.loadtxt(io.BytesIO(body), dtype=np.float64)
                if raw.ndim != 2 or raw.shape[1] != 3:
                    raise ValueError("Expected three numeric columns")
                t, x, e = raw.T
                metadata.update(
                    n_observations=len(raw),
                    baseline_days=float(np.ptp(t)),
                    has_errors=True,
                )
                if len(raw) < 30:
                    reasons.append("Fewer than 30 original observations")
                if not np.isfinite(raw).all():
                    reasons.append("Nonfinite observation")
                if np.any(e <= 0):
                    reasons.append("Nonpositive photometric uncertainty")
                if len(np.unique(t)) != len(t):
                    reasons.append("Duplicate observation times")
                if np.std(x) <= np.finfo(float).eps:
                    reasons.append("Constant or numerically degenerate magnitude")
                if not reasons:
                    phase = ((t - row["catalog_epoch"]) / row["catalog_period"]) % 1
                    if len(np.unique(phase)) != len(phase):
                        reasons.append("Exact duplicate folded phases")
            except (ValueError, IndexError) as exc:
                reasons.append("Unreadable photometry: " + str(exc))
        inventory.append(
            {
                **row,
                **metadata,
                "eligible": not reasons,
                "exclusion_reason": "; ".join(reasons),
            }
        )
        if (index + 1) % 5000 == 0:
            print("Inventoried", index + 1, "/", len(table), flush=True)
    inventory = pd.DataFrame(inventory)
    cohort = (
        inventory[inventory.eligible]
        .drop(columns=["eligible", "exclusion_reason"])
        .copy()
    )
    inventory.to_csv(data_dir / "inventory.csv", index=False)
    cohort.to_csv(data_dir / "cohort.csv", index=False)
    inventory[~inventory.eligible].to_csv(data_dir / "exclusions.csv", index=False)
    pd.DataFrame(extra, columns=["region", "archive_member", "reason"]).to_csv(
        data_dir / "unmatched_archive_members.csv", index=False
    )
    summary = dict(
        survey="Original OGLE-III OIII-CVS catalogs; I-band archive photometry",
        n_catalog_objects=len(inventory),
        n_eligible=len(cohort),
        n_excluded=int((~inventory.eligible).sum()),
        n_unmatched_archive_members=len(extra),
        catalog_region_counts=inventory.region.value_counts().to_dict(),
        catalog_type_counts=inventory.catalog_type.value_counts().to_dict(),
        eligible_region_counts=cohort.region.value_counts().to_dict(),
        eligible_type_counts=cohort.catalog_type.value_counts().to_dict(),
        n_measurements=int(cohort.n_observations.sum()),
        criteria=[
            "All catalog IDs across BLG/LMC/SMC, including historical RRe",
            "Original I-band data with at least 30 observations",
            "Finite observations, positive errors, distinct times and folded phases, nonconstant magnitude",
            "Catalog period and epoch present; RRd first-overtone mode",
            "No byte-identical duplicate light curves; no individual-point clipping or repair",
        ],
        interpretation="Inventory covers all catalog IDs; analysis cohort excludes only explicitly logged unusable inputs. No random downsampling for the full experiment.",
    )
    (data_dir / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return data_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    prepare(args.data)
