"""Plotting utilities for evaporation results.

All plotting functions assume the results dicts have the shape produced
by `evap_cool.evaporation` (`create_result_dict` / `create_mb_result_dict`).
Quantum runs that halted early may have shorter arrays than the MB run on
the same Q-schedule; `align_results` pads them to the MB length so that
positional indexing into Q matches across statistics.

`None` and NaN values are tolerated: matplotlib's scatter will simply
skip non-finite points, so partial runs render correctly without manual
masking.
"""

from __future__ import annotations

import math
from typing import Optional

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PLOT_COLORS = {
    "bosons":   "tab:green",
    "fermions": "tab:red",
    "mb":       "tab:blue",
}

PLOT_LABELS = {
    "bosons":   "Bose-Einstein",
    "fermions": "Fermi-Dirac",
    "mb":       "Maxwell-Boltzmann",
}


# ---------------------------------------------------------------------------
# Alignment for partial runs
# ---------------------------------------------------------------------------
def align_results(
    results_quantum: dict,
    results_mb: dict,
) -> dict:
    """Pad a partial quantum results dict to the MB run's full length.

    Quantum runs may halt early (BEC proximity, NR failure). For plotting
    against a complete MB run on the same Q-schedule, the quantum arrays
    need to be padded with NaN so positional indexing into Q is consistent
    across statistics.

    Parameters
    ----------
    results_quantum : dict
        Possibly-partial quantum results from a halted run.
    results_mb : dict
        Reference (complete) MB results dict on the same schedule.

    Returns
    -------
    dict
        Copy of `results_quantum` with all per-step arrays NaN-padded to
        the MB length. The Q schedule is left unmodified (it's pre-built
        at full length before the run starts).
    """
    aligned: dict = {}
    target_default = len(results_mb["N"])

    for key, vals in results_quantum.items():
        if not isinstance(vals, list):
            aligned[key] = vals
            continue

        if key == "Q":
            # Pre-built at full length; never pad.
            aligned[key] = list(vals)
            continue

        # Match length against MB if MB has the key, else fall back to N length.
        target = len(results_mb[key]) if key in results_mb else target_default

        clean = [(_safe_float(v)) for v in vals]
        if len(clean) < target:
            clean = clean + [math.nan] * (target - len(clean))

        aligned[key] = clean

    return aligned


def _safe_float(x):
    """Cast to float, mapping None / unconvertible to NaN."""
    if x is None:
        return math.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_combined_overview(
    results_b: dict,
    results_f: dict,
    results_mb: dict,
    trap_name: str,
    n_b: Optional[int] = None,
    n_f: Optional[int] = None,
    n_mb: Optional[int] = None,
    figsize: tuple = (24, 7),
):
    """1×3 overview plot: T/T0 vs Q, N/N0 vs Q, N/N0 vs T/T0.

    All three statistics overlaid on each panel.
    """
    n_b  = n_b  or len(results_b["Tf"])
    n_f  = n_f  or len(results_f["Tf"])
    n_mb = n_mb or len(results_mb["Tf"])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    panels = [
        ("Q",  "Tf", r"Cut-off temperature $Q$ [K]", r"$T_i / T_0$",
         f"{trap_name}: Temperature vs. Cut-off"),
        ("Q",  "Nf", r"Cut-off temperature $Q$ [K]", r"$N_i / N_0$",
         f"{trap_name}: Particle fraction vs. Cut-off"),
        ("Tf", "Nf", r"$T_i / T_0$",                 r"$N_i / N_0$",
         f"{trap_name}: Particle fraction vs. Temperature"),
    ]

    datasets = [
        (results_mb, n_mb, "mb"),
        (results_b,  n_b,  "bosons"),
        (results_f,  n_f,  "fermions"),
    ]

    for ax, (xkey, ykey, xlabel, ylabel, title) in zip(axes, panels):
        for data, n_pts, key in datasets:
            ax.scatter(
                data[xkey][:n_pts], data[ykey][:n_pts],
                c=PLOT_COLORS[key], s=2, label=PLOT_LABELS[key],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=15)
        ax.legend(fontsize=12)
        ax.tick_params(axis="both", labelsize=12)

    fig.tight_layout()
    return fig


def plot_individual_panels(
    results_b: dict,
    results_f: dict,
    results_mb: dict,
    trap_name: str,
    n_b: Optional[int] = None,
    n_f: Optional[int] = None,
    n_mb: Optional[int] = None,
    figsize: tuple = (20, 18),
    log_scale: bool = False,
):
    """3×3 grid: rows are (T vs Q, N vs Q, N vs T); columns are (MB, B, F).

    Plots dimensional T and N (not normalized).

    Parameters
    ----------
    log_scale : bool
        If True, set both axes to log on every panel. Defaults to False
        because partial runs and very small dimensional values can break
        log scaling.
    """
    n_b  = n_b  or len(results_b.get("Tf", results_b["T"]))
    n_f  = n_f  or len(results_f.get("Tf", results_f["T"]))
    n_mb = n_mb or len(results_mb.get("Tf", results_mb["T"]))

    datasets = [
        ("Maxwell-Boltzmann", results_mb, n_mb, PLOT_COLORS["mb"]),
        ("Bose-Einstein",     results_b,  n_b,  PLOT_COLORS["bosons"]),
        ("Fermi-Dirac",       results_f,  n_f,  PLOT_COLORS["fermions"]),
    ]

    rows = [
        ("Q", "T", "Cut-off temperature [K]", "Sample temperature [K]", "T vs Q"),
        ("Q", "N", "Cut-off temperature [K]", "Number of atoms",         "N vs Q"),
        ("T", "N", "Sample temperature [K]",  "Number of atoms",         "N vs T"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=figsize)

    for col, (label, data, n_pts, color) in enumerate(datasets):
        for row, (xkey, ykey, xlabel, ylabel, row_title) in enumerate(rows):
            ax = axes[row, col]
            ax.scatter(data[xkey][:n_pts], data[ykey][:n_pts], c=color, s=2)
            ax.set_title(f"{label} — {trap_name}: {row_title}", fontsize=11)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if log_scale:
                ax.set_xscale("log")
                ax.set_yscale("log")

    fig.tight_layout()
    return fig