from __future__ import annotations

import pandas as pd

from data_supply_chain.egfr import metformin_tier

TIER_LABELS = {2: "standard (>=45)", 1: "caution (30-44)", 0: "contraindicated (<30)"}


def medication_masking_table(dm: pd.DataFrame) -> dict:
    dm = dm.copy()
    dm["tier_2009"] = metformin_tier(dm.egfr_2009)
    dm["tier_2021"] = metformin_tier(dm.egfr_2021)
    dm["masked"] = dm.tier_2009 > dm.tier_2021
    by = dm.groupby("race")["masked"].agg(["mean", "count"])
    bk = dm[dm.black & dm.masked]
    return {
        "dm": dm,
        "by_race_pct": (by["mean"] * 100).to_dict(),
        "by_race_n": by["count"].to_dict(),
        "black_cross_45": int(((bk.egfr_2009 >= 45) & (bk.egfr_2021 < 45)).sum()),
        "black_cross_30": int(((bk.egfr_2009 >= 30) & (bk.egfr_2021 < 30)).sum()),
        "black_masked_n": len(bk),
    }


def decision_gate(black_masked_pct: float, max_masked_pct: float = 0.0):
    if black_masked_pct > max_masked_pct:
        return "REVIEW", [f"Black diabetic masking {black_masked_pct:.2f}% > {max_masked_pct}%"]
    return "PASS", []


def masking_directions(dm: pd.DataFrame) -> dict:
    dm = dm.copy()
    dm["tier_2009"] = metformin_tier(dm.egfr_2009)
    dm["tier_2021"] = metformin_tier(dm.egfr_2021)
    dm["masked"] = dm.tier_2009 > dm.tier_2021
    dm["reverse"] = dm.tier_2021 > dm.tier_2009

    by = dm.groupby("race")[["masked", "reverse"]].agg(["mean", "count"])
    return {
        "dm": dm,
        "masked_pct": (by[("masked", "mean")] * 100).to_dict(),
        "masked_n": dm[dm.masked].groupby("race").size().reindex(by.index, fill_value=0).to_dict(),
        "reverse_pct": (by[("reverse", "mean")] * 100).to_dict(),
        "reverse_n": dm[dm.reverse].groupby("race").size().reindex(by.index, fill_value=0).to_dict(),
        "denominator": by[("masked", "count")].to_dict(),
    }


def tier_transition_table(dm: pd.DataFrame, direction: str = "masked") -> pd.DataFrame:
    if direction not in ("masked", "reverse"):
        raise ValueError(f"direction must be 'masked' or 'reverse', not {direction!r}")

    dm = dm.copy()
    dm["tier_2009"] = metformin_tier(dm.egfr_2009)
    dm["tier_2021"] = metformin_tier(dm.egfr_2021)

    if direction == "masked":
        dm["masked"] = dm.tier_2009 > dm.tier_2021
        sub = dm[dm.black & dm.masked].copy()
        sub["from"] = sub.tier_2009.map(TIER_LABELS)
        sub["to"] = sub.tier_2021.map(TIER_LABELS)
    else:
        sub = dm[dm.tier_2021 > dm.tier_2009].copy()
        sub["from"] = sub.tier_2009.map(TIER_LABELS)
        sub["to"] = sub.tier_2021.map(TIER_LABELS)

    return sub.groupby(["from", "to"]).size().reset_index(name="n")
