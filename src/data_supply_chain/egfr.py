from __future__ import annotations
import numpy as np
import pandas as pd

_BLACK_COEFF = 1.159

DEMO_PATIENT = {"age": 50, "scr": 1.5, "female": False, "black": True}


def egfr_2009_race(scr: pd.Series, age: pd.Series, female: pd.Series, black: pd.Series) -> pd.Series:
    scr = np.asarray(scr, float)
    age = np.asarray(age, float)
    female = np.asarray(female, bool)
    black = np.asarray(black, bool)
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.329, -0.411)
    r = scr / k
    return (
        141
        * np.minimum(r, 1) ** a
        * np.maximum(r, 1) ** -1.209
        * 0.993 ** age
        * np.where(female, 1.018, 1)
        * np.where(black, _BLACK_COEFF, 1)
    )


def egfr_2021_racefree(scr: pd.Series, age: pd.Series, female: pd.Series) -> pd.Series:
    scr = np.asarray(scr, float)
    age = np.asarray(age, float)
    female = np.asarray(female, bool)
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    r = scr / k
    return (
        142
        * np.minimum(r, 1) ** a
        * np.maximum(r, 1) ** -1.200
        * 0.9938 ** age
        * np.where(female, 1.012, 1)
    )


def egfr_2021_cystatin(scys: pd.Series, age: pd.Series, female: pd.Series) -> pd.Series:
    scys = np.asarray(scys, float)
    age = np.asarray(age, float)
    female = np.asarray(female, bool)
    r = scys / 0.8
    return (
        133
        * np.minimum(r, 1) ** -0.499
        * np.maximum(r, 1) ** -1.328
        * 0.996 ** age
        * np.where(female, 0.932, 1)
    )


def egfr_asr_nb(egfr_2009, black) -> np.ndarray:
    egfr_2009 = np.asarray(egfr_2009, float)
    black = np.asarray(black, bool)
    return np.where(black, egfr_2009 / _BLACK_COEFF, egfr_2009)


def crossings(
    df: pd.DataFrame,
    col: str,
    thr: float = 60,
    into: bool = True,
    baseline: str = "egfr_2009",
) -> np.ndarray:
    a = np.asarray(df[baseline].values, float)
    b = np.asarray(df[col].values, float) if isinstance(col, str) else np.asarray(col, float)
    return (a >= thr) & (b < thr) if into else (a < thr) & (b >= thr)


def removal_rates(
    df: pd.DataFrame,
    col: str,
    *,
    thr: float = 60,
    baseline: str = "egfr_2009",
    race_col: str = "race",
) -> dict[str, float]:
    black = df[race_col].values == "Black"
    into = crossings(df, col, thr=thr, into=True, baseline=baseline)
    out = crossings(df, col, thr=thr, into=False, baseline=baseline)
    return {
        "black_into": float(into[black].mean() * 100),
        "black_out": float(out[black].mean() * 100),
        "nonblack_into": float(into[~black].mean() * 100),
        "nonblack_out": float(out[~black].mean() * 100),
    }


def reclass_table(
    df: pd.DataFrame,
    e09_col: str = "egfr_2009",
    e21_col: str = "egfr_2021",
) -> pd.DataFrame:
    thresholds = {60: "CKD (eGFR<60)", 45: "G3b", 20: "Transplant referral proxy"}
    rows = []
    for thr, label in thresholds.items():
        crossed = (df[e09_col] >= thr) & (df[e21_col] < thr)
        by = df.assign(c=crossed).groupby("race")["c"].mean()
        rows.append(
            {
                "threshold": thr,
                "label": label,
                "Black_pct": by.get("Black", 0) * 100,
                "NonBlack_pct": by.get("Non-Black", 0) * 100,
            }
        )
    return pd.DataFrame(rows)


def ckd_stage(egfr: pd.Series) -> pd.Series:
    return pd.cut(
        np.atleast_1d(np.asarray(egfr, float)),
        bins=[-np.inf, 15, 30, 45, 60, 90, np.inf],
        labels=["G5", "G4", "G3b", "G3a", "G2", "G1"],
    )


def metformin_tier(egfr: pd.Series) -> np.ndarray:
    e = np.asarray(egfr, float)
    return np.select([e >= 45, e >= 30], [2, 1], default=0)


def add_egfr_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["egfr_2009"] = egfr_2009_race(df["scr"], df["age"], df["female"], df["black"])
    df["egfr_2021"] = egfr_2021_racefree(df["scr"], df["age"], df["female"])
    if df["scys"].notna().any():
        df["egfr_cys"] = egfr_2021_cystatin(df["scys"], df["age"], df["female"])
    else:
        df["egfr_cys"] = np.nan
    return df
