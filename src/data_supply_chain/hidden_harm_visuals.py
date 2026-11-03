from __future__ import annotations

import textwrap
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

_ESTIMATE = "#B45309"
_REFERENCE = "#12506B"
_MUTED = "#94a3b8"
_INK = "#010203"

SURVIVES = "Survives correction"
DOES_NOT = "Does not survive"
NOT_EVALUATED = "Not evaluated"


def _ready(frame: pd.DataFrame | None, columns: Sequence[str], name: str) -> bool:
    if frame is None:
        print(f"{name}: not available. Run the analysis cell above first. If you did, "
              "the correction was skipped or had no auditable family.")
        return False
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: the table is missing columns {sorted(missing)}.")
    if frame.empty:
        print(f"{name}: no rows to show. That is not evidence of safety.")
        return False
    return True


def _alpha(frame: pd.DataFrame, alpha: float | None) -> float:
    stored = frame.attrs.get("alpha")
    if alpha is None:
        if stored is None:
            raise ValueError("This table does not record its alpha; pass the one the search used.")
        return float(stored)
    if stored is not None and not np.isclose(alpha, stored):
        raise ValueError(f"alpha={alpha} but this search was corrected at alpha={stored}. "
                         "Its survival decisions are only meaningful against that value.")
    return float(alpha)


def _status(value: object) -> str:
    if pd.isna(value):
        return NOT_EVALUATED
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("The 'survives' column must hold booleans.")
    return SURVIVES if value else DOES_NOT


def _labels(frame: pd.DataFrame) -> list[str]:
    out = []
    for _, row in frame.iterrows():
        label = textwrap.fill(str(row["cell"]), width=40, break_long_words=False)
        if "n" in frame and pd.notna(row["n"]):
            label += f"\nn = {int(row['n']):,}"
            if "n_pos" in frame and pd.notna(row["n_pos"]):
                label += f" · deaths = {int(row['n_pos']):,}"
        out.append(label)
    return out


def _figure(labels: list[str], title: str, xlabel: str, note: str, right: float = 0.76):
    height = max(4.0, 0.75 * len(labels) + 2.6)
    fig, ax = plt.subplots(figsize=(12, height))
    fig.subplots_adjust(left=0.30, right=right, top=1 - 1.0 / height, bottom=1.45 / height)
    fig.suptitle(title, x=0.02, y=0.98, ha="left", fontsize=15, color=_INK)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.02, 0.02, textwrap.fill(note, 120), fontsize=10, va="bottom", color=_INK)
    return fig, ax


def _side(ax, row: int, text: str) -> None:
    ax.text(1.02, row, text, transform=ax.get_yaxis_transform(), va="center", fontsize=10)


def _show(ax, legend: bool = True) -> None:
    if legend:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0,
                  frameon=False, fontsize=10, ncol=3)
    plt.show()


def _auditable(frame: pd.DataFrame, needed: Sequence[str]) -> pd.DataFrame:
    view = frame
    if "suppressed" in view:
        suppressed = view["suppressed"].notna()
        if suppressed.any():
            print(f"{int(suppressed.sum())} suppressed group(s) are left off this chart, "
                  "not drawn at zero. Their reasons are in the table above.")
        view = view.loc[~suppressed]
    gaps = view[list(needed)].isna().any(axis=1)
    if gaps.any():
        print(f"{int(gaps.sum())} row(s) have no value to draw and are left off, not drawn at zero.")
    return view.loc[~gaps].copy()


def plot_audit_counts(runs: Sequence[tuple[str, pd.DataFrame | None, pd.DataFrame | None]],
                      title: str = "Count what made it through") -> None:
    rows = []
    for name, lattice, corrected in runs:
        if lattice is None:
            print(f"{name}: no subgroup table yet; not shown.")
            continue
        survived = None
        if corrected is not None and corrected["survives"].notna().all():
            survived = int(corrected["survives"].sum())
        auditable = int(lattice["suppressed"].isna().sum())
        rows.append((name, [len(lattice), auditable, len(lattice) - auditable, survived]))
    if not rows:
        print("No completed searches to show.")
        return

    stages = ["Considered", "Auditable", "Suppressed", SURVIVES]
    note = ("These are stages of one search, not separate groups: do not add them. "
            f"'{NOT_EVALUATED}' means the correction did not run, which is not zero. "
            "Zero surviving signals is not proof of safety.")
    fig, ax = _figure(stages, title, "Number of subgroups", note, right=0.95)
    y = np.arange(len(stages))
    height = 0.7 / len(rows)
    top = max([1] + [v for _, values in rows for v in values if v is not None])
    for k, (name, values) in enumerate(rows):
        offset = (k - (len(rows) - 1) / 2) * height
        known = [i for i, v in enumerate(values) if v is not None]
        bars = ax.barh(y[known] + offset, [values[i] for i in known], height=height * 0.9,
                       label=name, color=(_REFERENCE, _ESTIMATE, _MUTED)[k % 3],
                       hatch=("", "//", "xx")[k % 3], edgecolor="white")
        ax.bar_label(bars, padding=4, fontsize=10)
        for i, v in enumerate(values):
            if v is None:
                ax.text(0.01 * top, y[i] + offset, NOT_EVALUATED, va="center", fontsize=10)
    ax.set_xlim(0, top * 1.15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _show(ax, legend=len(rows) > 1)


def plot_group_intervals(frame: pd.DataFrame | None, overall_auroc: float,
                         title: str = "Look inside the groups", *,
                         sort_by: str | None = "auroc") -> None:
    if not _ready(frame, ["cell", "n", "n_pos", "auroc", "lo", "hi"], title):
        return
    view = _auditable(frame, ["auroc", "lo", "hi"])
    if view.empty:
        print("No auditable groups to draw.")
        return
    if sort_by is not None:
        view = view.sort_values(sort_by, kind="stable")
    if (view["lo"] > view["hi"]).any():
        raise ValueError("A stored interval has lo > hi; inspect the table, not the chart.")

    note = ("Dot = the group's AUROC. Line = its 95% bootstrap interval. Dashed line = the "
            "overall held-out AUROC, for orientation only: a small group's AUROC is not "
            "directly comparable to it, because size and death count move AUROC on their "
            "own. The matched-peer comparison is the fair one.")
    fig, ax = _figure(_labels(view), title, "AUROC (0.5 = chance, 1.0 = perfect ranking)", note)
    y = np.arange(len(view))
    ax.hlines(y, view["lo"], view["hi"], color=_MUTED, linewidth=2.5,
              label="95% bootstrap interval")
    ax.plot(view["auroc"], y, "o", color=_ESTIMATE, markersize=7, label="Group AUROC")
    ax.axvline(overall_auroc, color=_REFERENCE, linestyle="--", linewidth=1.4,
               label=f"Overall held-out AUROC {overall_auroc:.4f}")
    ax.set_xlim(min(0.5, np.floor(view["lo"].min() * 20) / 20), 1.0)
    for j, (_, row) in enumerate(view.iterrows()):
        _side(ax, j, f"{row.auroc:.4f}  [{row.lo:.4f}, {row.hi:.4f}]")
    _show(ax)


def plot_peer_difference(frame: pd.DataFrame | None,
                         title: str = "Compare with matched peers") -> None:
    if not _ready(frame, ["cell", "auroc", "excess"], title):
        return
    view = _auditable(frame, ["excess"]).sort_values("auroc", kind="stable")
    if view.empty:
        print("No matched-peer differences to draw.")
        return

    y = np.arange(len(view))
    note = ("Bar = excess: the group's AUROC minus the average AUROC of reference groups "
            "matched on size and death count. Left of zero = below its matched peers. "
            "Grey band = where 95% of those reference groups themselves land; a bar inside "
            "it is within ordinary variation for a group of that shape. Descriptive and "
            "uncorrected: the next section is the corrected test, and it tests raw AUROC.")
    fig, ax = _figure(_labels(view), title, "excess = group AUROC − matched-reference AUROC", note)
    extent = float(view["excess"].abs().max())
    if {"ref_auroc", "ref_lo", "ref_hi"} <= set(view.columns):
        band = view[["ref_lo", "ref_hi"]].sub(view["ref_auroc"], axis=0)
        ok = band.notna().all(axis=1).to_numpy()
        lo, hi = band["ref_lo"].to_numpy(), band["ref_hi"].to_numpy()
        ax.barh(y[ok], (hi - lo)[ok], left=lo[ok], height=0.7,
                color=_MUTED, alpha=0.35, label="Matched-reference 95% range")
        extent = max(extent, float(band.abs().max().max()))
    ax.barh(y, view["excess"], height=0.4, color=_ESTIMATE, label="excess")
    ax.axvline(0, color=_INK, linewidth=1)
    extent = max(extent * 1.15, 0.01)
    ax.set_xlim(-extent, extent)
    for j, value in enumerate(view["excess"]):
        _side(ax, j, f"{value:+.4f}")
    _show(ax)


def plot_corrected_pvalues(frame: pd.DataFrame | None, alpha: float | None = None,
                           title: str = "See what survives") -> None:
    if not _ready(frame, ["cell", "auroc", "p_fwer", "survives"], title):
        return
    alpha = _alpha(frame, alpha)
    view = frame.sort_values("auroc", kind="stable")
    gaps = view["p_fwer"].isna()
    if gaps.any():
        print(f"{int(gaps.sum())} row(s) have no p-value and are left off, not drawn at zero.")
    view = view.loc[~gaps]
    if view.empty:
        return

    statuses = view["survives"].map(_status).to_numpy()
    note = ("Each mark is a group's stored p_fwer: its raw AUROC compared with the worst AUROC "
            f"in each permuted copy of this whole search. '{SURVIVES}' = stored p_fwer below "
            f"this search's alpha. '{DOES_NOT}' does not mean safe.")
    fig, ax = _figure(_labels(view), title, "Corrected p-value (p_fwer)", note)
    y = np.arange(len(view))
    for status, marker, colour in [(SURVIVES, "o", _ESTIMATE), (DOES_NOT, "X", _MUTED),
                                   (NOT_EVALUATED, "s", _MUTED)]:
        keep = statuses == status
        if keep.any():
            ax.plot(view["p_fwer"].to_numpy()[keep], y[keep], linestyle="none",
                    marker=marker, color=colour, markersize=9, label=status)
    ax.axvline(alpha, color=_REFERENCE, linestyle="--", label=f"alpha = {alpha:g}")
    ax.set_xlim(0, 1.02)
    for j, (_, row) in enumerate(view.iterrows()):
        _side(ax, j, f"{row.p_fwer:.4f} · {_status(row.survives)}")
    _show(ax)


def plot_pvalue_change(before: pd.DataFrame | None, after: pd.DataFrame | None,
                       before_name: str = "Baseline", after_name: str = "Comparison",
                       title: str = "Same groups, different search") -> None:
    columns = ["cell", "auroc", "p_fwer", "survives"]
    if not _ready(before, columns, before_name) or not _ready(after, columns, after_name):
        return
    a_before, a_after = _alpha(before, None), _alpha(after, None)
    view = before[columns].merge(after[columns], on="cell", how="inner",
                                 suffixes=("_before", "_after"), validate="one_to_one")
    for name, frame, other in [(before_name, before, after_name), (after_name, after, before_name)]:
        if len(frame) > len(view):
            print(f"{len(frame) - len(view)} group(s) tested only in {name} are not shown. "
                  f"{other} did not test them, which is neither a pass nor a fail.")
    gaps = view[["p_fwer_before", "p_fwer_after"]].isna().any(axis=1)
    if gaps.any():
        print(f"{int(gaps.sum())} shared group(s) lack a p-value and are left off, not drawn at zero.")
    view = view.loc[~gaps].sort_values("auroc_before", kind="stable")
    if view.empty:
        print("No shared groups to compare.")
        return

    labels = [textwrap.fill(str(r.cell), width=40, break_long_words=False)
              + f"\nAUROC {r.auroc_before:.4f} → {r.auroc_after:.4f}" for r in view.itertuples()]
    note = (f"Each row is one group tested in both runs ({before_name}; {after_name}). "
            "Same patients, same predictions, so its AUROC does not move; what moves "
            "is p_fwer, because the search it is corrected against changed. These are two "
            "exploratory searches: nothing here corrects for having run both.")
    fig, ax = _figure(labels, title, "Corrected p-value (p_fwer)", note, right=0.66)
    y = np.arange(len(view))
    ax.hlines(y, view[["p_fwer_before", "p_fwer_after"]].min(axis=1),
              view[["p_fwer_before", "p_fwer_after"]].max(axis=1), color=_MUTED, linewidth=1.6)
    ax.plot(view["p_fwer_before"], y, "o", color=_REFERENCE, markersize=8, label=before_name)
    ax.plot(view["p_fwer_after"], y, "s", color=_ESTIMATE, fillstyle="none", markersize=10,
            markeredgewidth=2, label=after_name)
    if np.isclose(a_before, a_after):
        ax.axvline(a_before, color=_INK, linestyle="--", linewidth=1, label=f"alpha = {a_before:g}")
    else:
        ax.axvline(a_before, color=_REFERENCE, linestyle="--", label=f"{before_name} alpha = {a_before:g}")
        ax.axvline(a_after, color=_ESTIMATE, linestyle=":", label=f"{after_name} alpha = {a_after:g}")
    ax.set_xlim(0, 1.02)
    for j, r in enumerate(view.itertuples()):
        _side(ax, j, f"p {r.p_fwer_before:.4f} → {r.p_fwer_after:.4f}\n"
                     f"{_status(r.survives_before)} → {_status(r.survives_after)}")
    _show(ax)


def search_details(corrected: pd.DataFrame | None, settings: dict,
                   alpha: float | None = None) -> pd.DataFrame:
    if corrected is not None:
        alpha = _alpha(corrected, alpha)
    records = [
        ("Characteristics searched", ", ".join(settings.get("axes", [])) or "Not recorded"),
        ("Deepest intersection", settings.get("max_depth", "Not recorded")),
        ("Minimum group size", settings.get("min_n", "Not recorded")),
        ("Alpha", alpha if alpha is not None else "Not recorded"),
    ]
    if corrected is None:
        records.append(("Correction", NOT_EVALUATED))
    else:
        attrs = corrected.attrs
        records += [
            ("Groups in the corrected family", attrs.get("n_family", "Not recorded")),
            ("Valid permutations", f"{attrs['n_perm_valid']} of {attrs['n_perm']}"
             if {"n_perm_valid", "n_perm"} <= set(attrs) else "Not recorded"),
            ("Lowest simulated minimum (not a cutoff)",
             f"{attrs['null_min']:.4f}" if "null_min" in attrs else "Not recorded"),
        ]
    return pd.DataFrame(records, columns=["Search detail", "Value"]).set_index("Search detail")
