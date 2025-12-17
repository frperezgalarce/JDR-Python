import pandas as pd
import numpy as np

def read_ogle_dat(path: str) -> pd.DataFrame:
    """
    Read an OGLE-style .dat light curve.

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

    # Enforce numeric dtype and drop invalid rows
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    return df
