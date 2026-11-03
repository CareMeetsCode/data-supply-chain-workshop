from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from data_supply_chain.audit import subgroup_audit
from data_supply_chain.bootstrap import WorkshopState
from data_supply_chain.preprocess import build_feature_matrix, impute_train_test


def foil_mortality_model(
    df: pd.DataFrame,
    egfr_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
):
    X = df[["age", "female", egfr_col]].astype(float)
    y = df.death.values
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=7),
    )
    clf.fit(X.iloc[train_idx], y[train_idx])
    p = clf.predict_proba(X.iloc[test_idx])[:, 1]
    sub = df.iloc[test_idx].reset_index(drop=True)
    return roc_auc_score(y[test_idx], p), sub, y[test_idx], p


@dataclass
class TrainResult:
    auroc: float
    brier: float
    tab: pd.DataFrame
    summ: dict
    p: np.ndarray
    y: np.ndarray
    sub: pd.DataFrame


def train_eval(
    df: pd.DataFrame,
    egfr_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    k: int = 5,
    tau: float = 0.5,
    calibrate: bool = False,
    reweight: bool = False,
    use_gpu: bool = False,
) -> TrainResult:
    X_df, cols = build_feature_matrix(df, egfr_col)
    X = X_df.values
    y = df["death"].values

    X_train, X_test = impute_train_test(
        X, train_idx, test_idx, k=k, tau=tau, use_gpu=use_gpu
    )
    y_train, y_test = y[train_idx], y[test_idx]

    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight=("balanced" if reweight else None),
            random_state=7,
        ),
    )
    if calibrate:
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    else:
        model = base
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    sub = df.iloc[test_idx].reset_index(drop=True)
    g = roc_auc_score(y_test, p)
    tab, summ = subgroup_audit(sub, y_test, p, "race", n_boot=500)
    return TrainResult(
        auroc=g,
        brier=brier_score_loss(y_test, p),
        tab=tab,
        summ=summ,
        p=p,
        y=y_test,
        sub=sub,
    )


def fit_audit_model(S: WorkshopState, egfr_col: str = "egfr_2021") -> WorkshopState:
    from dataclasses import replace

    from data_supply_chain.egfr import add_egfr_columns

    df = add_egfr_columns(S.df)
    auroc, sub, y_test, p = foil_mortality_model(df, egfr_col, S.train_idx, S.test_idx)

    return replace(
        S,
        df=df,
        predictions=p,
        y_test=y_test,
        audit_sub=sub,
        auroc=float(auroc),
        egfr_col=egfr_col,
    )


def pareto_sweep(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ks: list[int] | None = None,
    taus: list[float] | None = None,
    calibrate_opts: list[bool] | None = None,
    use_gpu: bool = False,
) -> pd.DataFrame:
    from data_supply_chain.audit import subgroup_audit

    ks = ks or [3, 5, 7, 11]
    taus = taus or [0.3, 0.5, 1.0]
    calibrate_opts = calibrate_opts or [False, True]
    rows = []
    for k in ks:
        for tau in taus:
            for cal in calibrate_opts:
                r = train_eval(
                    df,
                    "egfr_2021",
                    train_idx,
                    test_idx,
                    k=k,
                    tau=tau,
                    calibrate=cal,
                    reweight=True,
                    use_gpu=use_gpu,
                )
                _, race_summ = subgroup_audit(r.sub, r.y, r.p, "race", n_boot=100)
                _, ins_summ = subgroup_audit(r.sub, r.y, r.p, "insurance", n_boot=100)
                rows.append(
                    dict(
                        k=k,
                        tau=tau,
                        calibrate=cal,
                        global_auroc=r.auroc,
                        worst_race_auroc=race_summ["worst_auroc"],
                        worst_ins_auroc=ins_summ["worst_auroc"],
                        worst_race_ece=race_summ["worst_ece"],
                        worst_ins_ece=ins_summ["worst_ece"],
                    )
                )
    return pd.DataFrame(rows)
