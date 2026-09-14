"""Package the frozen complete inputs and the validated pilot without claiming a full run."""

from pathlib import Path
from hashlib import sha256
import json, zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/representation_comparison_ogle3_all"


def main():
    with zipfile.ZipFile(
        OUT / "portable_reproduction_bundle.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as z:
        for folder, prefix in [
            (ROOT / "data/ogle3_rrlyrae_all", "inputs"),
            (OUT / "pilot", "outputs/pilot"),
        ]:
            for p in sorted(folder.rglob("*")):
                if p.is_file():
                    z.write(p, str(Path(prefix) / p.relative_to(folder)))
        for p, name in [
            (
                ROOT / "notebooks/jdr_vs_classical_features_ogle3_all.ipynb",
                "jdr_vs_classical_features_ogle3_all.ipynb",
            ),
            (OUT / "README.md", "README.md"),
            (OUT / "VALIDATION.md", "VALIDATION.md"),
            (OUT / "requirements-lock.txt", "requirements-lock.txt"),
            (ROOT / "scripts/run_ogle3_notebook.py", "run_notebook.py"),
            (ROOT / "review/jdr_paper_1000/reference.pdf", "reference.pdf"),
        ]:
            z.write(p, name)
        for name in ["acquire_ogle3.py", "prepare_ogle3_inventory.py"]:
            z.write(ROOT / "scripts" / name, "maintenance/scripts/" + name)
    with zipfile.ZipFile(
        OUT / "journal_figures.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as z:
        for p in sorted((OUT / "pilot/figures").iterdir()):
            z.write(p, p.name)
    records = []
    for p in [OUT / "portable_reproduction_bundle.zip", OUT / "journal_figures.zip"]:
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
        h = sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1024 * 1024), b""):
                h.update(b)
        records.append(dict(file=p.name, bytes=p.stat().st_size, sha256=h.hexdigest()))
    (OUT / "bundle_checksums.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
