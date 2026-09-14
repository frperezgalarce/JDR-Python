"""Convert an already-loaded light curve without guessing time offsets."""

from pathlib import Path
import numpy as np


def lightcurve_to_ogle(lc, path):
    # Astropy Time supplies the format conversion; no numerical offset guessing.
    time = np.asarray(lc.time.tdb.jd, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    err = getattr(lc, "flux_err", None)
    err = None if err is None else np.asarray(err.value, dtype=float)
    if (
        time.ndim != 1
        or flux.shape != time.shape
        or (err is not None and err.shape != time.shape)
    ):
        raise ValueError("Time, flux and errors must be matching 1D arrays.")
    mask = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    if err is not None:
        mask &= np.isfinite(err) & (err > 0)
    time, flux = time[mask], flux[mask]
    if len(time) < 3 or len(np.unique(time)) != len(time):
        raise ValueError(
            "At least three valid, distinct observation times are required."
        )
    mag = -2.5 * np.log10(flux / np.median(flux))
    values = [time, mag]
    if err is not None:
        values.append(2.5 / np.log(10) * err[mask] / flux)
    order = np.argsort(time)
    output = np.column_stack(values)[order]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        output,
        fmt="%.17g",
        header="Time in JD representation, TDB scale; original spatial reference retained.\n time mag"
        + (" mag_err" if err is not None else ""),
    )
    return {"path": str(path), "kept": len(time), "dropped": int((~mask).sum())}
