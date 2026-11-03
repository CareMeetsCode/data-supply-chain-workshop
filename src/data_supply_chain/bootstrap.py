from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

SEED = 7
_SPLIT_SEED = SEED + 1


@dataclass
class WorkshopState:
    df: pd.DataFrame
    train_idx: np.ndarray
    test_idx: np.ndarray
    source: str
    is_real: bool

    predictions: np.ndarray | None = None
    y_test: np.ndarray | None = None
    audit_sub: pd.DataFrame | None = None
    auroc: float | None = None
    egfr_col: str | None = None


def reset(verbose: bool = True) -> WorkshopState:
    from data_supply_chain.models import fit_audit_model

    S = fit_audit_model(cold_start())

    if verbose:
        bar = "=" * 68
        print(bar)
        print("  RESET — you are back to a clean step 9 starting point.")
        print(bar)
        print(f"  {len(S.df):,} adults · source={S.source} · real={S.is_real}")
        print(f"  test set {len(S.test_idx):,} rows · model AUROC {S.auroc:.4f}")
        print()
        print("  Type this, then re-run your search cell:")
        print()
        print("      S = reset()")
        print()
        print("  Nothing you do in the EDIT HERE cell can damage the published")
        print("  findings — those four lines never write findings.json. The")
        print("  Record cell writes the pinned default search on CPU and")
        print("  ignores your edits. Break the search as often as you like.")
        print(bar)

    return S


def cold_start(data_dir: str | None = None, synthetic: bool = False) -> WorkshopState:
    from data_supply_chain.data import load_cohort

    df, source = load_cohort(data_dir=data_dir, synthetic=synthetic)

    rng_split  = np.random.default_rng(_SPLIT_SEED)
    SPLIT_IDX  = rng_split.permutation(len(df))
    CUT        = int(0.7 * len(df))
    TRAIN_IDX, TEST_IDX = SPLIT_IDX[:CUT], SPLIT_IDX[CUT:]

    return WorkshopState(
        df=df,
        train_idx=TRAIN_IDX,
        test_idx=TEST_IDX,
        source=source,
        is_real=(source != "synthetic"),
    )
