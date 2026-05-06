"""Plotting utilities for evaporation results.

All plotting functions assume the results dicts have the shape produced
by `evap_cool.evaporation` (`create_result_dict` / `create_mb_result_dict`).
Quantum runs that halted early may have shorter arrays than the MB run on
the same Q-schedule; `align_results` pads them to the MB length so that
positional indexing into Q matches across statistics.

`None` and NaN values are tolerated: matplotlib's scatter will simply
skip non-finite points, so partial runs render correctly without manual
masking.

Equilibrium thermodynamics
--------------------------
`plot_state_functions` and `plot_thermal_coefficients` plot the
post-processed quantities from `evap_cool.post_processing` against the
sample temperature. They consume the per-step thermo arrays alongside
the source run's T array.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

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


def _abs_or_nan(values, take_abs):
    """Return [|v| if take_abs else v, ...] with None / non-finite -> NaN."""
    floats = [_safe_float(v) for v in values]
    if take_abs:
        return [abs(v) if not math.isnan(v) else math.nan for v in floats]
    return floats


# ---------------------------------------------------------------------------
# Plots: evaporation overview
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


# ---------------------------------------------------------------------------
# Plots: equilibrium thermodynamics
# ---------------------------------------------------------------------------
def _plot_thermo_panel(
    ax,
    T_b, q_b,
    T_f, q_f,
    *,
    ylabel: str,
    title: str,
    log_x: bool,
    log_y: bool,
):
    """Scatter one thermodynamic quantity (B + F overlaid) on a single axis.

    When `log_y=True`, absolute values are plotted to handle
    sign-changing quantities like Omega (always negative) or F, G
    (sign depends on regime).
    """
    q_b_plot = _abs_or_nan(q_b, take_abs=log_y)
    q_f_plot = _abs_or_nan(q_f, take_abs=log_y)

    ax.scatter(T_b, q_b_plot, c=PLOT_COLORS["bosons"], s=2,
               label=PLOT_LABELS["bosons"])
    ax.scatter(T_f, q_f_plot, c=PLOT_COLORS["fermions"], s=2,
               label=PLOT_LABELS["fermions"])

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"Sample temperature $T$ [K]", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(axis="both", labelsize=10)


def plot_state_functions(
    T_b: Sequence[float],
    thermo_b: dict,
    T_f: Sequence[float],
    thermo_f: dict,
    trap_name: str,
    *,
    n_b: Optional[int] = None,
    n_f: Optional[int] = None,
    figsize: tuple = (24, 14),
    log_x: bool = True,
    log_y: bool = True,
    units_energy: str = "J",
):
    """2×3 grid of equilibrium state functions vs sample temperature.

    Panels: Ω, S, P, H, F, G. Bosons and fermions overlaid on each.

    Parameters
    ----------
    T_b, T_f : sequence of float
        Sample temperature at each step, from the source evap run's
        `results['T']` (bosons and fermions runs respectively).
    thermo_b, thermo_f : dict
        Per-step thermodynamics from `load_thermodynamics(...)['results']`.
        Must contain keys `Omega`, `S`, `P`, `H`, `F`, `G`.
    trap_name : str
        Trap label used in subplot titles, e.g. "Quadrupole".
    n_b, n_f : int, optional
        Trim each statistics' arrays to this length. Useful if a run
        halted early. Defaults to the full array length.
    figsize : tuple
        Forwarded to `plt.subplots`. Default (24, 14) gives 8×7-inch
        panels, matching the rest of the plotting suite.
    log_x, log_y : bool
        Axis scaling. With `log_y=True`, |Ω|, |F|, |G| are plotted to
        handle quantities that can be negative (Ω = -PV always, and
        F = μN - PV, G = μN can flip sign with regime).
    units_energy : str
        Energy unit label, used in y-axis labels. Default "J"; pass
        "eV" for traps in the eV unit system.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_b = n_b or len(T_b)
    n_f = n_f or len(T_f)

    # (key, axis-title symbol, full y-axis label)
    panels = [
        ("Omega",
         r"$|\Omega|$" if log_y else r"$\Omega$",
         rf"$|\Omega|$ [{units_energy}]" if log_y
         else rf"$\Omega$ [{units_energy}]"),
        ("S",
         r"$S$",
         rf"Entropy $S$ [{units_energy}/K]"),
        ("P",
         r"$P$",
         rf"Pressure $P$ [{units_energy}/m$^3$]"),
        ("H",
         r"$H$",
         rf"Enthalpy $H$ [{units_energy}]"),
        ("F",
         r"$|F|$" if log_y else r"$F$",
         rf"Helmholtz energy $|F|$ [{units_energy}]" if log_y
         else rf"Helmholtz energy $F$ [{units_energy}]"),
        ("G",
         r"$|G|$" if log_y else r"$G$",
         rf"Gibbs energy $|G|$ [{units_energy}]" if log_y
         else rf"Gibbs energy $G$ [{units_energy}]"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    for ax, (key, sym, ylabel) in zip(axes.flat, panels):
        _plot_thermo_panel(
            ax,
            T_b[:n_b], thermo_b[key][:n_b],
            T_f[:n_f], thermo_f[key][:n_f],
            ylabel=ylabel,
            title=f"{trap_name}: {sym} vs $T$",
            log_x=log_x, log_y=log_y,
        )

    fig.suptitle(
        f"{trap_name} trap — Equilibrium state functions",
        fontsize=15, y=1.0,
    )
    fig.tight_layout()
    return fig


def plot_thermal_coefficients(
    T_b: Sequence[float],
    thermo_b: dict,
    T_f: Sequence[float],
    thermo_f: dict,
    trap_name: str,
    *,
    n_b: Optional[int] = None,
    n_f: Optional[int] = None,
    figsize: tuple = (16, 14),
    log_x: bool = True,
    log_y: bool = True,
    units_energy: str = "J",
):
    """2×2 grid of thermal coefficients vs sample temperature.

    Panels: C_V, C_P, κ_T, B_P. Bosons and fermions overlaid on each.

    Parameters
    ----------
    T_b, T_f, thermo_b, thermo_f, trap_name, n_b, n_f, figsize,
    log_x, log_y, units_energy
        Same conventions as `plot_state_functions`. The thermo dicts
        must contain keys `CV`, `CP`, `kappa_T`, `B_P`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_b = n_b or len(T_b)
    n_f = n_f or len(T_f)

    panels = [
        ("CV",
         r"$C_V$",
         rf"Heat capacity $C_V$ [{units_energy}/K]"),
        ("CP",
         r"$C_P$",
         rf"Heat capacity $C_P$ [{units_energy}/K]"),
        ("kappa_T",
         r"$\kappa_T$",
         rf"Compressibility $\kappa_T$ [m$^3$/{units_energy}]"),
        ("B_P",
         r"$B_P$",
         r"Thermal expansion $B_P$ [1/K]"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, (key, sym, ylabel) in zip(axes.flat, panels):
        _plot_thermo_panel(
            ax,
            T_b[:n_b], thermo_b[key][:n_b],
            T_f[:n_f], thermo_f[key][:n_f],
            ylabel=ylabel,
            title=f"{trap_name}: {sym} vs $T$",
            log_x=log_x, log_y=log_y,
        )

    fig.suptitle(
        f"{trap_name} trap — Thermal coefficients",
        fontsize=15, y=1.0,
    )
    fig.tight_layout()
    return fig