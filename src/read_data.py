"""Validated OGLE input. Raw irregular observations are preserved by default."""

from numbers import Integral
import numpy as np
import pandas as pd


def read_ogle_dat(path, fixed_length=None):
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
    if df.shape[1] not in (2, 3):
        raise ValueError(f"{path}: expected time, mag [, mag_err].")
    df.columns = ["time", "mag", "mag_err"][: df.shape[1]]
    df = df.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(
            f"{path}: non-finite or nonnumeric observations; clean explicitly."
        )
    if "mag_err" in df and (df.mag_err <= 0).any():
        raise ValueError(f"{path}: measurement errors must be positive.")
    df = df.sort_values("time").reset_index(drop=True)
    if len(df) < 3 or df.time.nunique() < 3:
        raise ValueError(
            f"{path}: at least three distinct observation times are required."
        )
    if df.time.duplicated().any():
        raise ValueError(
            f"{path}: duplicate times; aggregate explicitly before analysis."
        )
    if np.std(df.mag.to_numpy(), ddof=1) <= np.finfo(float).eps * max(
        1.0, np.max(np.abs(df.mag))
    ):
        raise ValueError(f"{path}: constant or numerically constant magnitudes.")
    if fixed_length is not None:
        if (
            isinstance(fixed_length, bool)
            or not isinstance(fixed_length, Integral)
            or fixed_length < 3
        ):
            raise ValueError("fixed_length must be an integer >= 3.")
        if "mag_err" in df:
            raise ValueError(
                "Resampling correlated measurement errors is unsupported; use a fitted model."
            )
        time = np.linspace(df.time.min(), df.time.max(), fixed_length)
        df = pd.DataFrame({"time": time, "mag": np.interp(time, df.time, df.mag)})
    return df
