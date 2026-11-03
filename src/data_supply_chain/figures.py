from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

matplotlib.rcParams["svg.hashsalt"] = "data-supply-chain"

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

_INK = "#010203"

_C_2009 = "#3b82f6"
_C_2021 = "#ef4444"
_C_GATE = "#94a3b8"


def _write_svg(fig: plt.Figure, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight", transparent=True,
                metadata={"Date": None})
    plt.close(fig)

    svg = out.read_text()
    if _INK not in svg:
        raise RuntimeError(
            f"{out.name}: nothing was drawn in {_INK}. Text and rules must use "
            "_INK so they follow the page theme; a figure without it is "
            "invisible in either light or dark mode."
        )
    out.write_text(svg.replace(_INK, "currentColor"))


def _png_bytes(fig: plt.Figure) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(fig)
    fig.set_facecolor("white")
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", bbox_inches="tight", dpi=150,
        facecolor="white", edgecolor="none", metadata={"Date": None},
    )
    plt.close(fig)
    plt.close("all")
    return buf.getvalue()


def build_egfr_gates(egfr_2009: float, egfr_2021: float) -> plt.Figure:
    fig = Figure(figsize=(7.2, 2.7))
    ax = fig.add_subplot(111)

    lo, hi = min(egfr_2009, egfr_2021), max(egfr_2009, egfr_2021)
    x_min, x_max = lo - 30, hi + 34

    for x, label in ((15, "G5│G4"), (30, "G4│G3b"), (45, "G3b│G3a"),
                     (60, "G3a│G2"), (90, "G2│G1")):
        if not x_min <= x <= x_max:
            continue
        crossed = lo < x <= hi
        ax.axvline(
            x, color=_C_GATE, zorder=1,
            linewidth=2.2 if crossed else 1.0,
            linestyle="-" if crossed else (0, (4, 3)),
        )
        ax.annotate(
            label, xy=(x, 1.04), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=7.5, color=_INK,
            fontweight="bold" if crossed else "normal",
            alpha=1.0 if crossed else 0.5,
        )

    ax.annotate(
        "", xy=(egfr_2021, 0), xytext=(egfr_2009, 0),
        arrowprops=dict(arrowstyle="-|>", color=_INK, linewidth=1.5,
                        shrinkA=11, shrinkB=11),
    )

    for value, colour, caption in (
        (egfr_2009, _C_2009, "CKD-EPI 2009\nwith race coefficient"),
        (egfr_2021, _C_2021, "CKD-EPI 2021\nrace-free"),
    ):
        outward = "left" if value == hi else "right"
        nudge = 10 if value == hi else -10
        ax.scatter([value], [0], s=160, color=colour, zorder=3, clip_on=False)
        ax.annotate(f"{value:.1f}", xy=(value, 0), xytext=(nudge, 16),
                    textcoords="offset points", ha=outward,
                    fontsize=11.5, fontweight="bold", color=colour)
        ax.annotate(caption, xy=(value, 0), xytext=(nudge, -38),
                    textcoords="offset points", ha=outward,
                    fontsize=8, color=colour, linespacing=1.45)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("eGFR  (mL/min/1.73m²)", fontsize=9, color=_INK)
    ax.tick_params(axis="x", colors=_INK, labelsize=8.5)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_INK)
    ax.spines["bottom"].set_alpha(0.35)

    return fig


def plot_egfr_gates(egfr_2009: float, egfr_2021: float, out: Path) -> None:
    _write_svg(build_egfr_gates(egfr_2009, egfr_2021), out)


def egfr_gates_png(egfr_2009: float, egfr_2021: float) -> bytes:
    return _png_bytes(build_egfr_gates(egfr_2009, egfr_2021))


def build_reclass(rates: dict[str, dict[str, float]]) -> plt.Figure:
    strategies = list(rates)
    fig = Figure(figsize=(7.6, 2.9))
    axes = fig.subplots(1, len(strategies), sharex=True, sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    limit = max(
        max(abs(v) for v in r.values()) for r in rates.values()
    ) * 1.35 or 1.0

    for ax, name in zip(axes, strategies):
        r = rates[name]
        for y, grp in ((1, "black"), (0, "nonblack")):
            pct_into, pct_out = r[f"{grp}_into"], r[f"{grp}_out"]
            ax.barh(y, pct_into, height=0.46, color=_C_2021, zorder=3)
            ax.barh(y, -pct_out, height=0.46, color=_C_2009, zorder=3)
            for value, x, ha in ((pct_into, pct_into, "left"), (pct_out, -pct_out, "right")):
                if value > 0.001:
                    ax.annotate(
                        f"{value:.2f}%",
                        xy=(x, y), xytext=(4 if ha == "left" else -4, 0),
                        textcoords="offset points", va="center", ha=ha,
                        fontsize=8, color=_INK,
                    )
            if pct_into < 0.001 and pct_out < 0.001:
                ax.annotate("no movement", xy=(0, y), xytext=(6, 0),
                            textcoords="offset points", va="center",
                            fontsize=8, style="italic", color=_INK, alpha=0.6)

        ax.axvline(0, color=_INK, linewidth=1, alpha=0.45, zorder=2)
        ax.set_title(name, fontsize=9.5, color=_INK)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Non-Black", "Black"], fontsize=8.5)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-0.6, 1.6)
        ax.tick_params(axis="x", colors=_INK, labelsize=8)
        ax.tick_params(axis="y", colors=_INK)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(_INK)
        ax.spines["bottom"].set_alpha(0.3)
        ax.set_xlabel("← out of CKD          into CKD →", fontsize=8.5, color=_INK)

    return fig


def plot_reclass(rates: dict[str, dict[str, float]], out: Path) -> None:
    _write_svg(build_reclass(rates), out)


def reclass_png(rates: dict[str, dict[str, float]]) -> bytes:
    return _png_bytes(build_reclass(rates))


def _gate_default(name: str) -> float:
    import inspect

    from data_supply_chain.audit import ship_gate

    return inspect.signature(ship_gate).parameters[name].default


def plot_audit_bar(tab: pd.DataFrame, global_auroc: float, title: str, out: Path) -> None:
    _write_svg(build_audit_bar(tab, global_auroc, title), out)


def audit_bar_png(tab: pd.DataFrame, global_auroc: float, title: str) -> bytes:
    return _png_bytes(build_audit_bar(tab, global_auroc, title))


def build_audit_bar(tab: pd.DataFrame, global_auroc: float, title: str) -> plt.Figure:
    floor = _gate_default("min_worst_auroc")
    max_drop = _gate_default("max_drop")

    scored = tab[tab["auroc"].notna()]
    suppressed = tab[tab["auroc"].isna()]
    order = list(scored["group"].astype(str)) + list(suppressed["group"].astype(str))

    fig = Figure(figsize=(5.6, 3.4))
    ax = fig.add_subplot(111)

    x = range(len(scored))
    ax.bar(x, scored["auroc"], width=0.5, color=_C_2021, zorder=3)
    ax.errorbar(
        x, scored["auroc"],
        yerr=[scored["auroc"] - scored["lo"], scored["hi"] - scored["auroc"]],
        fmt="none", ecolor=_INK, elinewidth=1.2, capsize=4, zorder=4,
    )
    for i, row in enumerate(scored.itertuples()):
        ax.annotate(f"{row.auroc:.3f}", xy=(i, row.hi), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=_INK, zorder=5)
        ax.annotate(f"n={row.n:,}", xy=(i, 0), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=_INK, alpha=0.75, zorder=5)

    for j, row in enumerate(suppressed.itertuples(), start=len(scored)):
        ax.annotate(f"not evaluated\nn={row.n:,}", xy=(j, 0.06),
                    ha="center", va="bottom", fontsize=7.5, style="italic",
                    color=_INK, alpha=0.65, zorder=5)

    ax.axhline(global_auroc, color=_INK, linestyle=(0, (1, 2.5)), linewidth=1.3,
               zorder=2, label=f"global {global_auroc:.3f}")
    ax.axhline(floor, color=_C_GATE, linestyle="--", linewidth=1.6, zorder=2,
               label=f"worst-group floor {floor:.2f}")

    subtitle = ""
    if len(scored):
        worst = scored.loc[scored["auroc"].idxmin()]
        gap = global_auroc - worst["auroc"]
        if gap > max_drop:
            i = order.index(str(worst["group"]))
            ax.annotate(
                "", xy=(i + 0.42, global_auroc), xytext=(i + 0.42, worst["auroc"]),
                arrowprops=dict(arrowstyle="<->", color=_INK, linewidth=1.1),
            )
            subtitle = (
                f"worst group is {gap:.3f} below global — the gate trips over {max_drop:.2f}"
            )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8.5)

    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("AUROC", fontsize=9, color=_INK)
    ax.set_title(title, fontsize=9.5, color=_INK, pad=30 if subtitle else 14)
    if subtitle:
        ax.annotate(subtitle, xy=(0.5, 1.012), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=8, color=_INK, alpha=0.85)
    ax.tick_params(colors=_INK, labelsize=8)

    legend = ax.legend(fontsize=7.5, frameon=False, loc="upper left",
                       ncol=2, columnspacing=1.6, handlelength=2.4)
    for text in legend.get_texts():
        text.set_color(_INK)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK)
        ax.spines[side].set_alpha(0.3)

    return fig


def plot_metformin_masking(rates: dict[str, float], counts: dict[str, int], out: Path) -> None:
    _write_svg(build_metformin_masking(rates, counts), out)


def metformin_masking_png(rates: dict[str, float], counts: dict[str, int]) -> bytes:
    return _png_bytes(build_metformin_masking(rates, counts))


def build_metformin_masking(rates: dict[str, float], counts: dict[str, int]) -> plt.Figure:
    fig = Figure(figsize=(6.4, 2.6))
    ax = fig.add_subplot(111)

    limit = max(max(abs(v) for v in rates.values()), 0.5) * 1.9

    for y, grp, label in ((1, "black", "Black"), (0, "nonblack", "Non-Black")):
        masked, reverse = rates[f"{grp}_masked"], rates[f"{grp}_reverse"]
        ax.barh(y, reverse, height=0.42, color=_C_2021, zorder=3)
        ax.barh(y, -masked, height=0.42, color=_C_2009, zorder=3)

        for value, key, x, ha in ((masked, f"{grp}_masked", -masked, "right"),
                                  (reverse, f"{grp}_reverse", reverse, "left")):
            text = f"{value:.2f}%  (n={counts[key]})" if value > 0.001 else "0.00%"
            ax.annotate(
                text,
                xy=(x, y), xytext=(5 if ha == "left" else -5, 0),
                textcoords="offset points", va="center", ha=ha,
                fontsize=8, color=_INK, alpha=1.0 if value > 0.001 else 0.55,
            )

    ax.axvline(0, color=_INK, linewidth=1, alpha=0.45, zorder=2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Non-Black", "Black"], fontsize=8.5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-0.6, 1.6)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _pos: f"{abs(v):.0f}%")
    )
    ax.set_xlabel(
        "← tier MASKED by the race equation          tier RAISED without it →",
        fontsize=8.5, color=_INK,
    )
    ax.set_title("Metformin eligibility, both directions", fontsize=9.5, color=_INK)
    ax.tick_params(colors=_INK, labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_INK)
    ax.spines["bottom"].set_alpha(0.3)

    return fig


def plot_bundle_sweep(sweep: pd.DataFrame, out: Path) -> None:
    floor = _gate_default("min_worst_auroc")
    ece_cap = _gate_default("max_worst_ece")

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    cal_style = {
        False: ("#B45309", "o", "uncalibrated"),
        True: ("#12506B", "D", "isotonic"),
    }
    ks = sorted(sweep["k"].unique())
    size = {k: 26 + 26 * (i / max(len(ks) - 1, 1)) for i, k in enumerate(ks)}

    for cal, (colour, marker, _label) in cal_style.items():
        part = sweep[sweep["calibrate"] == cal]
        if part.empty:
            continue
        ax.scatter(
            part["worst_race_ece"], part["global_auroc"],
            c=colour, marker=marker, s=[size[k] for k in part["k"]],
            edgecolors=_INK, linewidths=0.6, zorder=3, alpha=0.9,
        )

    ax.axvline(ece_cap, ls="--", color="crimson", linewidth=1.0, alpha=0.8, zorder=2)
    ax.text(ece_cap, 1.0, f"  ship-gate ECE cap ({ece_cap:g})", fontsize=7,
            color="crimson", va="top", ha="left")

    ax.set_ylim(floor, 1.0)
    ax.set_xlim(0, max(0.13, float(sweep["worst_race_ece"].max()) * 1.08))

    ax.set_xlabel("Worst-group ECE (race) — lower is better", fontsize=9, color=_INK)
    ax.set_ylabel(f"Global AUROC  (axis from the ship-gate floor, {floor:g})",
                  fontsize=9, color=_INK)
    ax.tick_params(colors=_INK, labelsize=8)

    from matplotlib.lines import Line2D

    cal_handles = [
        Line2D([], [], marker=m, linestyle="none", markersize=7,
               markerfacecolor=c, markeredgecolor=_INK, markeredgewidth=0.6,
               label=lab)
        for _cal, (c, m, lab) in cal_style.items()
    ]
    k_handles = [
        Line2D([], [], marker="o", linestyle="none",
               markersize=3.2 + 2.2 * (i / max(len(ks) - 1, 1)),
               markerfacecolor="none", markeredgecolor=_INK,
               markeredgewidth=0.9, label=f"k = {k}")
        for i, k in enumerate(ks)
    ]
    first = ax.legend(handles=cal_handles, fontsize=7.5, frameon=False,
                      loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      handletextpad=0.4, labelspacing=0.5)
    ax.add_artist(first)
    second = ax.legend(handles=k_handles, fontsize=7.5, frameon=False,
                       loc="upper left", bbox_to_anchor=(1.02, 0.60),
                       handletextpad=0.4, labelspacing=0.5)
    for legend in (first, second):
        for text in legend.get_texts():
            text.set_color(_INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK)
        ax.spines[side].set_alpha(0.3)

    _write_svg(fig, out)
