from __future__ import annotations

import numpy as np
import pandas as pd

from data_supply_chain.bootstrap import WorkshopState


class SyntheticProjectionError(RuntimeError):
    pass


def survey_design_columns(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = (
        df["pooled_wt"] if "pooled_wt" in df
        else df["wt_mec"] / df["cycle"].nunique()
    ).values
    STR = (
        df["SDMVSTRA"] if "SDMVSTRA" in df
        else pd.Series(0, index=df.index)
    ).values
    PSU = (
        df["SDMVPSU"] if "SDMVPSU" in df
        else pd.Series(np.arange(len(df)), index=df.index)
    ).values
    return W, STR, PSU


def psu_groups(STR: np.ndarray, PSU: np.ndarray) -> dict:
    groups: dict = {}
    for _i, (_s, _p) in enumerate(zip(STR, PSU)):
        groups.setdefault(_s, {}).setdefault(_p, []).append(_i)
    return groups


def design_ci(
    mask: np.ndarray,
    W: np.ndarray,
    groups: dict,
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ests = []
    for _ in range(n_boot):
        rows = []
        for _s, psus in groups.items():
            keys = list(psus)
            pick = rng.integers(0, len(keys), len(keys))
            for j in pick:
                rows.extend(psus[keys[j]])
        rows = np.asarray(rows)
        ests.append(W[rows][mask[rows]].sum())
    return np.nanpercentile(ests, [2.5, 97.5])


MIN_UNWEIGHTED_N = 30
MAX_RSE = 0.30


def _t_multiplier(df: int) -> float:
    from scipy import stats

    return float(stats.t.ppf(0.975, df))


def ultimate_cluster_se(mask, W: np.ndarray, groups: dict) -> float:
    mask = np.asarray(mask, bool)
    y = W * mask
    v = 0.0
    for _stratum, psus in groups.items():
        totals = np.array([y[np.asarray(rows)].sum() for rows in psus.values()], float)
        n_h = len(totals)
        if n_h > 1:
            v += n_h / (n_h - 1) * ((totals - totals.mean()) ** 2).sum()
    return float(np.sqrt(v))


def design_degrees_of_freedom(groups: dict) -> int:
    return sum(len(psus) for psus in groups.values()) - len(groups)


def project(
    mask,
    base,
    label: str,
    W: np.ndarray,
    groups: dict,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    mask = np.asarray(mask, bool)
    base = np.asarray(base, bool)

    n_un = int(mask.sum())
    count = float(W[mask].sum())
    denom = W[base].sum()
    pct = float(W[mask & base].sum() / denom) if denom > 0 else np.nan

    se = ultimate_cluster_se(mask, W, groups)
    half = _t_multiplier(design_degrees_of_freedom(groups)) * se
    lo, hi = max(0.0, count - half), count + half
    rse = se / count if count > 0 else np.inf

    return dict(
        level=label,
        unweighted_n=n_un,
        weighted_pct=pct,
        national_count=count,
        ci_low=lo,
        ci_high=hi,
        rse=rse,
        reliable=(n_un >= MIN_UNWEIGHTED_N) and (rse <= MAX_RSE),
    )


def national_projection_table(
    state: WorkshopState,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    from data_supply_chain.egfr import metformin_tier

    if not state.is_real:
        raise SyntheticProjectionError(
            "National projections are disabled on synthetic data. "
            f"This state came from source={state.source!r}. Nothing computed here "
            "would be true of the United States, and the number would look "
            "identical to one that is. Re-run with the real cohort."
        )

    df = state.df
    W, STR, PSU = survey_design_columns(df)
    groups = psu_groups(STR, PSU)

    t09 = metformin_tier(df["egfr_2009"])
    t21 = metformin_tier(df["egfr_2021"])
    blk = df["black"].values.astype(bool)
    dmb = blk & df["diabetic"].values.astype(bool)

    e09 = df["egfr_2009"].values
    e21 = df["egfr_2021"].values

    rows = [
        project(blk & (e09 >= 60) & (e21 < 60), blk,
                "Black adults reclassified into CKD (eGFR<60)", W, groups,
                n_boot=n_boot, seed=seed),
        project(dmb & (t09 > t21), dmb,
                "Black diabetics: metformin tier masked", W, groups,
                n_boot=n_boot, seed=seed),
        project(dmb & (e09 >= 30) & (e21 < 30), dmb,
                "Black diabetics: into contraindicated (<30)", W, groups,
                n_boot=n_boot, seed=seed),
    ]
    return pd.DataFrame(rows)


def render_projection_table(proj: pd.DataFrame) -> pd.DataFrame:
    return proj.assign(
        weighted_pct=proj.weighted_pct.map(
            lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "-"
        ),
        national=proj.national_count.map(lambda x: f"{x:,.0f}"),
        ci_95=proj.apply(lambda r: f"{r.ci_low:,.0f}-{r.ci_high:,.0f}", axis=1),
        status=proj.reliable.map(
            lambda b: "reliable" if b else "UNRELIABLE (NCHS: n<30 or RSE>30%)"
        ),
    )[["level", "unweighted_n", "weighted_pct", "national", "ci_95", "status"]]
