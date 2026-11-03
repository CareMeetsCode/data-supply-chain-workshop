from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from data_supply_chain.audit import EmptyAuditError, ece, recovery_guidance

DEFAULT_MAX_DEPTH = 2

_CI_PERCENTILES = (2.5, 97.5)


_LAST_BACKEND: str | None = None


def last_backend() -> str | None:
    return _LAST_BACKEND


class _CpuBackend:
    name = "cpu"

    @staticmethod
    def auroc(y: np.ndarray, p: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        n_pos = float((y == 1).sum())
        n_neg = float(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        ranks = rankdata(np.asarray(p, dtype=float))
        return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    @staticmethod
    def auroc_many(y: np.ndarray, p: np.ndarray, draws: np.ndarray) -> np.ndarray:
        draws = np.asarray(draws, dtype=int)
        if draws.size == 0:
            return np.empty(0, dtype=float)
        yd = y[draws]
        pd_ = p[draws]
        n = yd.shape[1]
        n_pos = (yd == 1).sum(axis=1).astype(float)
        n_neg = n - n_pos
        keep = (n_pos > 0) & (n_neg > 0)
        if not keep.any():
            return np.empty(0, dtype=float)
        ranks = rankdata(pd_[keep], axis=1)
        yk = yd[keep]
        pos = n_pos[keep]
        neg = n_neg[keep]
        sums = (ranks * (yk == 1)).sum(axis=1)
        return (sums - pos * (pos + 1) / 2) / (pos * neg)


def _midranks_2d(xp, a):
    m, n = a.shape
    order = xp.argsort(a, axis=1, kind="stable")
    s = xp.take_along_axis(a, order, axis=1)

    is_new = xp.ones((m, n), dtype=bool)
    if n > 1:
        is_new[:, 1:] = s[:, 1:] != s[:, :-1]

    g = xp.cumsum(is_new, axis=1) - 1
    row_offset = (xp.arange(m) * n).reshape(m, 1)
    flat = (g + row_offset).ravel()

    ordinal = xp.broadcast_to(xp.arange(1, n + 1, dtype=xp.float64), (m, n)).ravel()
    counts = xp.bincount(flat, minlength=m * n).astype(xp.float64)
    sums = xp.bincount(flat, weights=ordinal, minlength=m * n)

    mean_rank = sums / xp.where(counts == 0, 1.0, counts)
    mid_sorted = mean_rank[flat].reshape(m, n)

    out = xp.empty((m, n), dtype=xp.float64)
    out[xp.arange(m).reshape(m, 1), order] = mid_sorted
    return out


class _GpuBackend:
    name = "gpu"

    @staticmethod
    def _xp():
        import cupy as cp

        return cp

    @classmethod
    def auroc(cls, y, p) -> float:
        cp = cls._xp()
        y = cp.asarray(y, dtype=cp.float64)
        p = cp.asarray(p, dtype=cp.float64)
        n_pos = float((y == 1).sum())
        n_neg = float(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        ranks = _midranks_2d(cp, p.reshape(1, -1))[0]
        return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    @classmethod
    def auroc_many(cls, y, p, draws):
        cp = cls._xp()
        draws = cp.asarray(draws)
        if draws.size == 0:
            return np.empty(0, dtype=float)
        y = cp.asarray(y, dtype=cp.float64)
        p = cp.asarray(p, dtype=cp.float64)
        yd = y[draws]
        pd_ = p[draws]
        n = yd.shape[1]
        n_pos = (yd == 1).sum(axis=1).astype(cp.float64)
        n_neg = n - n_pos
        keep = (n_pos > 0) & (n_neg > 0)
        if not bool(keep.any()):
            return np.empty(0, dtype=float)
        ranks = _midranks_2d(cp, pd_[keep])
        yk = yd[keep]
        pos = n_pos[keep]
        neg = n_neg[keep]
        sums = (ranks * (yk == 1)).sum(axis=1)
        return cp.asnumpy((sums - pos * (pos + 1) / 2) / (pos * neg))


def gpu_backend_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _get_backend(use_gpu: bool):
    global _LAST_BACKEND
    if use_gpu:
        if not gpu_backend_available():
            raise RuntimeError(
                "use_gpu=True but no CUDA device is available (cupy did not "
                "import, or reports zero devices).\n"
                "This raises rather than falling back to CPU so a GPU run cannot "
                "be reported as green when it never happened.\n"
                "The CPU path gives the same results — call "
                "with use_gpu=False."
            )
        _LAST_BACKEND = _GpuBackend.name
        return _GpuBackend
    _LAST_BACKEND = _CpuBackend.name
    return _CpuBackend


def _cell_rng(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int.from_bytes(digest, "big")])
    )


def _resample_indices(rng: np.random.Generator, n: int, n_draws: int) -> np.ndarray:
    return rng.integers(0, n, size=(n_draws, n))


def _cell_label(combo: tuple[str, ...], values: tuple) -> str:
    return " & ".join(f"{a}={v}" for a, v in zip(combo, values))


def _enumerate_cells(
    sub: pd.DataFrame, axes: list[str], max_depth: int
) -> list[tuple[tuple[str, ...], tuple, np.ndarray]]:
    cells = []
    for depth in range(1, max_depth + 1):
        for combo in itertools.combinations(axes, depth):
            grouped = sub.groupby(list(combo), observed=True, dropna=True)
            for values, positions in grouped.indices.items():
                if not isinstance(values, tuple):
                    values = (values,)
                cells.append((combo, values, np.asarray(positions, dtype=int)))
    return cells


def _suppression_reason(y_cell: np.ndarray, min_n: int) -> str | None:
    n = len(y_cell)
    if n < min_n:
        return f"n < min_n ({min_n})"
    if y_cell.min() == y_cell.max():
        return "single outcome class"
    return None


def _composition_matched_reference(
    y: np.ndarray,
    p: np.ndarray,
    positions: np.ndarray,
    rng: np.random.Generator,
    backend,
    n_ref: int,
) -> tuple[float, float, float]:
    mask = np.zeros(len(y), dtype=bool)
    mask[positions] = True
    comp_pos = np.flatnonzero(~mask & (y == 1))
    comp_neg = np.flatnonzero(~mask & (y == 0))

    y_cell = y[positions]
    n_pos = int((y_cell == 1).sum())
    n_neg = int((y_cell == 0).sum())

    if n_ref <= 0:
        return (np.nan, np.nan, np.nan)
    if len(comp_pos) == 0 or len(comp_neg) == 0 or n_pos == 0 or n_neg == 0:
        return (np.nan, np.nan, np.nan)

    draws = np.empty((n_ref, n_pos + n_neg), dtype=int)
    for i in range(n_ref):
        draws[i, :n_pos] = rng.choice(comp_pos, size=n_pos, replace=len(comp_pos) < n_pos)
        draws[i, n_pos:] = rng.choice(comp_neg, size=n_neg, replace=len(comp_neg) < n_neg)

    arr = backend.auroc_many(y, p, draws)
    lo, hi = np.percentile(arr, _CI_PERCENTILES)
    return (float(arr.mean()), float(lo), float(hi))


def slice_lattice(
    df: pd.DataFrame,
    axes: list[str],
    y: np.ndarray,
    p: np.ndarray,
    min_n: int = 30,
    n_boot: int = 200,
    max_depth: int = DEFAULT_MAX_DEPTH,
    n_ref: int = 200,
    seed: int = 1,
    use_gpu: bool = False,
) -> pd.DataFrame:
    backend = _get_backend(use_gpu)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    if len(df) != len(y) or len(y) != len(p):
        raise ValueError(
            f"df, y and p must be aligned: got len(df)={len(df)}, "
            f"len(y)={len(y)}, len(p)={len(p)}. `df` is the test-set frame with a "
            "reset index (WorkshopState.audit_sub), not the full cohort."
        )
    missing = [a for a in axes if a not in df.columns]
    if missing:
        raise KeyError(
            f"axes not present in the frame: {missing}. Available columns "
            f"include: {sorted(df.columns)[:12]}..."
        )

    rows = []
    covered: dict[tuple[str, ...], int] = {}
    for combo, values, positions in _enumerate_cells(df, axes, max_depth):
        label = _cell_label(combo, values)
        covered[combo] = covered.get(combo, 0) + len(positions)
        y_cell, p_cell = y[positions], p[positions]
        reason = _suppression_reason(y_cell, min_n)

        row = dict(
            depth=len(combo),
            axes=combo,
            values=values,
            cell=label,
            n=len(positions),
            n_pos=int((y_cell == 1).sum()),
            auroc=np.nan,
            lo=np.nan,
            hi=np.nan,
            ece=np.nan,
            ref_auroc=np.nan,
            ref_lo=np.nan,
            ref_hi=np.nan,
            excess=np.nan,
            suppressed=reason,
        )
        if reason is None:
            rng = _cell_rng(seed, label)
            boots = backend.auroc_many(
                y_cell, p_cell, _resample_indices(rng, len(y_cell), n_boot)
            )
            lo, hi = (
                np.percentile(boots, _CI_PERCENTILES) if len(boots) else (np.nan, np.nan)
            )
            ref_mean, ref_lo, ref_hi = _composition_matched_reference(
                y, p, positions, rng, backend, n_ref
            )
            row.update(
                auroc=backend.auroc(y_cell, p_cell),
                lo=lo,
                hi=hi,
                ece=ece(y_cell, p_cell),
                ref_auroc=ref_mean,
                ref_lo=ref_lo,
                ref_hi=ref_hi,
            )
            row["excess"] = row["auroc"] - ref_mean
        rows.append(row)

    out = (
        pd.DataFrame(rows)
        .sort_values("auroc", na_position="last")
        .reset_index(drop=True)
    )
    out.attrs.update(
        backend=backend.name,
        seed=seed,
        min_n=min_n,
        n_boot=n_boot,
        n_ref=n_ref,
        max_depth=max_depth,
        n_cells_rows={" & ".join(k): v for k, v in covered.items()},
    )
    return out


def permutation_null(
    df: pd.DataFrame,
    axes: list[str],
    y: np.ndarray,
    p: np.ndarray,
    n_perm: int = 500,
    min_n: int = 30,
    max_depth: int = DEFAULT_MAX_DEPTH,
    alpha: float = 0.05,
    seed: int = 1,
    use_gpu: bool = False,
) -> pd.DataFrame:
    backend = _get_backend(use_gpu)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    observed = slice_lattice(
        df, axes, y, p,
        min_n=min_n, n_boot=0, max_depth=max_depth, n_ref=0,
        seed=seed, use_gpu=use_gpu,
    )
    family = observed[observed.suppressed.isna()].copy()
    if family.empty:
        raise EmptyAuditError(
            "\n".join(
                [
                    f"slice_lattice: all {len(observed)} cells were suppressed at "
                    f"min_n={min_n}, so there is no family to correct over.",
                    "",
                    "This is a lattice too thin to test, not a model that passed.",
                    "Nothing survived to be corrected, which is not the same as "
                    "nothing being wrong.",
                    "",
                    *recovery_guidance(min_n),
                    "  * or lower max_depth — depth 2 intersects every pair of "
                    "axes, and",
                    "    each intersection is smaller than either axis alone",
                ]
            )
        )

    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int.from_bytes(b"permutation", "big")])
    )

    cells = [positions for _c, _v, positions in _enumerate_cells(df, axes, max_depth)]
    n_rows = len(df)
    null_minima = []
    for _ in range(n_perm):
        perm = rng.permutation(n_rows)
        minimum = np.inf
        for positions in cells:
            shuffled = perm[positions]
            y_cell = y[shuffled]
            if _suppression_reason(y_cell, min_n) is not None:
                continue
            minimum = min(minimum, backend.auroc(y_cell, p[shuffled]))
        if np.isfinite(minimum):
            null_minima.append(minimum)

    null = np.asarray(null_minima, dtype=float)
    n_valid = len(null)
    if n_valid == 0:
        raise ValueError(
            "permutation_null: no permutation produced a single auditable cell, so "
            f"the null is empty after {n_perm} attempts. The lattice is too thin "
            f"at min_n={min_n} for this test to mean anything."
        )

    family["p_fwer"] = [
        (1 + int((null <= a).sum())) / (1 + n_valid) for a in family.auroc
    ]
    family["survives"] = family.p_fwer < alpha

    family = family.drop(
        columns=["lo", "hi", "ece", "ref_auroc", "ref_lo", "ref_hi", "excess",
                 "suppressed"],
        errors="ignore",
    )
    out = family.sort_values("auroc").reset_index(drop=True)
    for inherited in ("n_boot", "n_ref"):
        out.attrs.pop(inherited, None)
    out.attrs.update(
        backend=backend.name,
        seed=seed,
        alpha=alpha,
        n_perm=n_perm,
        n_perm_valid=n_valid,
        n_family=len(family),
        null_min=float(null.min()),
        null_median=float(np.median(null)),
        null_max=float(null.max()),
    )
    return out
