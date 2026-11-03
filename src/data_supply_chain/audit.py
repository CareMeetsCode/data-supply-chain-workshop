from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data_supply_chain.egfr import metformin_tier

_SHIP_AUROC_FLOOR = 0.65
_SHIP_ECE_CAP = 0.10


def ece(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = 0.0
    n = len(y)
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            out += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(out)


def recovery_guidance(min_n: int) -> list[str]:
    return [
        "To recover:",
        "  * widen the subgroup, or audit an attribute with larger groups —",
        "    this is almost always the right move",
        f"  * lowering min_n (currently {min_n}) is almost always the WRONG one:",
        "    it does not create evidence, it only removes the check that was",
        "    telling you there was none",
    ]


class EmptyAuditError(ValueError):
    pass


def subgroup_audit(
    sub: pd.DataFrame,
    y,
    p,
    attr: str,
    min_n: int = 30,
    n_boot: int = 500,
    seed: int = 1,
):
    g = np.random.default_rng(seed)
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    rows = []
    for grp, _ in sub.groupby(attr, observed=True):
        m = sub[attr].values == grp
        yt, yp = y[m], p[m]
        n = len(yt)
        if n < min_n or yt.min() == yt.max():
            rows.append(
                dict(group=grp, n=n, auroc=np.nan, lo=np.nan, hi=np.nan, ece=np.nan)
            )
            continue
        boots = [
            roc_auc_score(yt[i], yp[i])
            for i in (g.integers(0, n, n) for _ in range(n_boot))
            if yt[i].min() != yt[i].max()
        ]
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
        rows.append(
            dict(
                group=grp,
                n=n,
                auroc=roc_auc_score(yt, yp),
                lo=lo,
                hi=hi,
                ece=ece(yt, yp),
            )
        )
    tab = pd.DataFrame(rows).sort_values("auroc", na_position="last").reset_index(drop=True)
    rep = tab.dropna(subset=["auroc"])

    if rep.empty:
        raise EmptyAuditError(_empty_audit_message(tab, attr, min_n))

    summ = dict(
        worst_auroc=rep.auroc.min(),
        worst_group=rep.loc[rep.auroc.idxmin(), "group"],
        gap=rep.auroc.max() - rep.auroc.min(),
        worst_ece=rep.ece.max(),
    )
    return tab, summ


def _empty_audit_message(tab: pd.DataFrame, attr: str, min_n: int) -> str:
    lines = [
        f"subgroup_audit({attr!r}): every group was suppressed, so there is "
        f"nothing to audit and no worst group to report.",
        "",
        f"  {'group':<20} {'n':>6}   why",
    ]
    for _, r in tab.iterrows():
        why = f"n < min_n ({min_n})" if int(r["n"]) < min_n else "single outcome class"
        lines.append(f"  {str(r['group']):<20} {int(r['n']):>6}   {why}")
    lines += [
        "",
        "No AUROC could be computed for any group, so the audit has no result.",
        "This is a slice too small to say anything about, not a model that failed.",
        "",
        *recovery_guidance(min_n),
    ]
    return "\n".join(lines)


def ship_gate(
    global_auroc: float,
    worst_auroc: float,
    worst_ece: float,
    min_worst_auroc: float = 0.65,
    max_drop: float = 0.10,
    max_worst_ece: float = 0.10,
):
    if not (np.isfinite(worst_auroc) and np.isfinite(worst_ece)):
        return (
            "DON'T SHIP",
            ["audit produced no usable subgroups — worst AUROC is undefined"],
        )

    reasons = []
    if worst_auroc < min_worst_auroc:
        reasons.append(f"worst AUROC {worst_auroc:.3f} < {min_worst_auroc}")
    if global_auroc - worst_auroc > max_drop:
        reasons.append(f"worst is {global_auroc - worst_auroc:.3f} below global")
    if worst_ece > max_worst_ece:
        reasons.append(f"worst ECE {worst_ece:.3f} > {max_worst_ece}")
    return ("SHIP" if not reasons else "DON'T SHIP"), reasons


def mortality_validation(dm: pd.DataFrame, weights: pd.Series | None = None):
    dm = dm.copy()
    if "masked" not in dm.columns:
        dm["tier_2009"] = metformin_tier(dm.egfr_2009)
        dm["tier_2021"] = metformin_tier(dm.egfr_2021)
        dm["masked"] = dm.tier_2009 > dm.tier_2021
    crossers = dm[dm.black & dm.masked]
    rest = dm[dm.black & ~dm.masked]
    out = {
        "crossers_n": len(crossers),
        "rest_n": len(rest),
        "crossers_mort": crossers.death.mean() if len(crossers) else np.nan,
        "rest_mort": rest.death.mean() if len(rest) else np.nan,
        "crossers_egfr_2021_mean": crossers.egfr_2021.mean() if len(crossers) else np.nan,
        "rest_egfr_2021_mean": rest.egfr_2021.mean() if len(rest) else np.nan,
    }
    if weights is not None and len(crossers) and len(rest):
        w = weights.reindex(dm.index)
        out["crossers_mort_wt"] = np.average(crossers.death, weights=w.loc[crossers.index])
        out["rest_mort_wt"] = np.average(rest.death, weights=w.loc[rest.index])
    return out
