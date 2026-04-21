import pandas as pd
import numpy as np


def read_ogle_dat(path: str, fixed_length: int = 200) -> pd.DataFrame:
    """
    Read an OGLE-style .dat light curve and interpolate it to fixed_length points.

    Expected formats:
      - 2 columns: time, mag
      - 3 columns: time, mag, mag_err

    Returns
    -------
    pd.DataFrame
        Columns: time, mag [, mag_err]
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        comment="#",
        engine="python"
    )

    ncol = df.shape[1]
    if ncol == 2:
        df.columns = ["time", "mag"]
    elif ncol == 3:
        df.columns = ["time", "mag", "mag_err"]
    else:
        raise ValueError(
            f"{path}: unexpected number of columns ({ncol}). "
            "Expected 2 or 3 (time, mag [, mag_err])."
        )

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().sort_values("time").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(f"{path}: not enough valid points to resample.")

    new_time = np.linspace(df["time"].min(), df["time"].max(), fixed_length)

    out = {"time": new_time}
    out["mag"] = np.interp(new_time, df["time"], df["mag"])

    if "mag_err" in df.columns:
        out["mag_err"] = np.interp(new_time, df["time"], df["mag_err"])

    return pd.DataFrame(out)