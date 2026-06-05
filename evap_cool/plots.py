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
the source run's T array. The Maxwell-Boltzmann reference (output of
`compute_mb_run_thermodynamics`) can be overlaid by passing
`T_mb=...` and `thermo_mb=...`; an immediate read on where quantum
statistics depart from the classical limit.

Zoom inset
~~~~~~~~~~
When MB is supplied, each panel also gets a small inset showing the
cold-end window where BE or FD departs from MB by more than
`zoom_threshold` (default 1 %). The inset axes share the parent's log-
or-linear scaling, so the visible departure in the inset corresponds
exactly to that in the main panel. Set `zoom=False` to suppress.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


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
    figsize: tuple = (30, 10),
    *,
    zoom: bool = True,
    zoom_threshold: float = 0.01,
    zoom_position: Optional[tuple] = None,
):
    """1×3 overview plot: T/T0 vs Q, N/N0 vs Q, N/N0 vs T/T0.

    All three statistics overlaid on each panel. With `zoom=True`
    (default) each panel gets a cold-end inset showing only the steps
    where BE or FD departs from MB by more than `zoom_threshold` --
    the regime where the cooling curves visibly split. Set
    `zoom=False` to suppress.
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
                c=PLOT_COLORS[key], s=3.5, label=PLOT_LABELS[key],
                zorder=(0 if key == "mb" else 1),
                alpha=(0.7 if key == "mb" else 1.0),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=15)
        ax.legend(fontsize=12, loc="lower right")
        ax.tick_params(axis="both", labelsize=12)

        # Cold-end inset, same simple detection as the thermo plots.
        if zoom:
            y_b  = results_b[ykey][:n_b]
            y_f  = results_f[ykey][:n_f]
            y_mb = results_mb[ykey][:n_mb]
            x_b = results_mb[xkey][:n_b]
            i_x = _classical_collapse_index(y_b, y_f, y_mb,
                                            threshold=zoom_threshold)
            if i_x is not None:
                _add_zoom_inset(
                    ax,
                    x_b, y_b,
                    results_b[xkey][:n_b], y_b,
                    results_f[xkey][:n_f], y_f,
                    i_x,
                    log_x=True, log_y=True, position=zoom_position,
                )

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
def _classical_collapse_index(q_b, q_f, q_mb, threshold=0.01):
    """Walk steps from warmest to coldest; return the first index `i` where
    BE or FD differs from MB by more than `threshold` (relative).

    Returns None if MB isn't available, lengths don't match up, or the
    quantum series never separates from MB within the dataset.
    """
    if q_b is None or q_f is None or q_mb is None:
        return None
    n = min(len(q_b), len(q_f), len(q_mb))
    for i in range(n):
        m = q_mb[i]; b = q_b[i]; f = q_f[i]
        if m is None or b is None or f is None:
            continue
        try:
            m_abs = abs(float(m))
            if m_abs == 0 or not math.isfinite(m_abs):
                continue
            db = abs(abs(float(b)) - m_abs) / m_abs
            df = abs(abs(float(f)) - m_abs) / m_abs
            if (math.isfinite(db) and db > threshold) or \
               (math.isfinite(df) and df > threshold):
                return i
        except (TypeError, ValueError):
            continue
    return None


def _auto_inset_position(q_mb, log_y, size=(0.40, 0.38), margin=0.06):
    """Pick an empty corner for the inset based on the slope of `q_mb`.

    Returns (x, y, w, h) in axes-fraction coords.

    * ↗ data (value drops as we cool): upper-LEFT is empty.
    * ↘ data (value grows as we cool, e.g. κ_T, B_P): upper-RIGHT is empty.
    """
    w, h = size
    upper_left  = (margin,         1 - h - margin, w, h)
    upper_right = (1 - w - margin, 1 - h - margin, w, h)
    if q_mb is None:
        return upper_left
    v_first = v_last = None
    for v in q_mb:
        if v is None: continue
        try:
            x = float(v)
            if math.isfinite(x):
                v_first = x; break
        except (TypeError, ValueError):
            continue
    for v in reversed(q_mb):
        if v is None: continue
        try:
            x = float(v)
            if math.isfinite(x):
                v_last = x; break
        except (TypeError, ValueError):
            continue
    if v_first is None or v_last is None:
        return upper_left
    if log_y:
        v_first, v_last = abs(v_first), abs(v_last)
    # q_mb[0] is at warmest T (right of plot); q_mb[-1] at coldest (left).
    # If v_last > v_first the value grows as T drops → data goes ↘ → upper-right empty.
    return upper_right if v_last > v_first else upper_left


def _add_zoom_inset(
    ax, T_mb, q_mb, T_b, q_b, T_f, q_f, i_x,
    *, log_x, log_y, position=None,
):
    """Add a small inset to `ax` showing only steps with index >= i_x.

    The inset replots all three series, clipped to `T <= T_mb[i_x]`.
    If `position` is None, the corner is auto-picked so the inset
    lands away from the data line.
    """
    try:
        T_max = float(T_mb[i_x])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(T_max):
        return None

    if position is None:
        position = _auto_inset_position(q_mb, log_y=log_y)

    axins = ax.inset_axes(position)

    def _clip_and_plot(T, q, color, alpha=1.0, zorder=2):
        T_z, q_z = [], []
        for t, v in zip(T, q):
            try:
                tv = float(t)
                if math.isfinite(tv) and tv <= T_max:
                    T_z.append(tv); q_z.append(v)
            except (TypeError, ValueError):
                continue
        if T_z:
            axins.scatter(T_z, _abs_or_nan(q_z, take_abs=log_y),
                          c=color, s=2.5, alpha=alpha, zorder=zorder)

    _clip_and_plot(T_mb, q_mb, PLOT_COLORS["mb"],       alpha=0.7, zorder=0)
    _clip_and_plot(T_b,  q_b,  PLOT_COLORS["bosons"])
    _clip_and_plot(T_f,  q_f,  PLOT_COLORS["fermions"])

    if log_x: axins.set_xscale("log")
    if log_y: axins.set_yscale("log")
    axins.tick_params(axis="both", labelsize=7)
    axins.set_title("low-$T$ zoom", fontsize=8, pad=2)
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.35")
    return axins


def _plot_thermo_panel(
    ax,
    T_b, q_b,
    T_f, q_f,
    *,
    ylabel: str,
    title: str,
    log_x: bool,
    log_y: bool,
    T_mb=None, q_mb=None,
    zoom: bool = True,
    zoom_threshold: float = 0.01,
    zoom_position: Optional[tuple] = None,
):
    """Scatter one thermodynamic quantity on a single axis.

    Bosons (green) and fermions (red) are always overlaid. If `T_mb` and
    `q_mb` are supplied, Maxwell-Boltzmann (blue) is drawn underneath
    them and -- if `zoom=True` -- a small inset is added showing the
    cold-end window where BE or FD differs from MB by more than
    `zoom_threshold`.

    When `log_y=True`, absolute values are plotted (handles sign-
    changing quantities like Ω, F, G).
    """
    # MB underlay, then quantum series on top.
    if T_mb is not None and q_mb is not None:
        ax.scatter(T_mb, _abs_or_nan(q_mb, take_abs=log_y),
                   c=PLOT_COLORS["mb"], s=2,
                   label=PLOT_LABELS["mb"], zorder=0, alpha=0.7)

    ax.scatter(T_b, _abs_or_nan(q_b, take_abs=log_y),
               c=PLOT_COLORS["bosons"], s=2, label=PLOT_LABELS["bosons"])
    ax.scatter(T_f, _abs_or_nan(q_f, take_abs=log_y),
               c=PLOT_COLORS["fermions"], s=2, label=PLOT_LABELS["fermions"])

    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.set_xlabel(r"Sample temperature $T$ [K]", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    # Legend goes to lower-right to leave upper-left free for the zoom inset.
    ax.legend(fontsize=10, loc="lower right")
    ax.tick_params(axis="both", labelsize=10)

    # Cold-end zoom inset (only meaningful when MB is supplied).
    if zoom and T_mb is not None and q_mb is not None:
        i_x = _classical_collapse_index(q_b, q_f, q_mb, threshold=zoom_threshold)
        if i_x is not None:
            _add_zoom_inset(
                ax, T_mb, q_mb, T_b, q_b, T_f, q_f, i_x,
                log_x=log_x, log_y=log_y, position=zoom_position,
            )


def plot_state_functions(
    T_b: Sequence[float],
    thermo_b: dict,
    T_f: Sequence[float],
    thermo_f: dict,
    trap_name: str,
    *,
    T_mb: Optional[Sequence[float]] = None,
    thermo_mb: Optional[dict] = None,
    n_b: Optional[int] = None,
    n_f: Optional[int] = None,
    n_mb: Optional[int] = None,
    figsize: tuple = (24, 14),
    log_x: bool = True,
    log_y: bool = True,
    units_energy: str = "J",
    zoom: bool = True,
    zoom_threshold: float = 0.01,
    zoom_position: Optional[tuple] = None,
):
    """2×3 grid of equilibrium state functions vs sample temperature.

    Panels: Ω, S, P, H, F, G. Bosons (green), fermions (red), and --
    when MB data is supplied -- Maxwell-Boltzmann (blue) overlaid on
    each panel. With `zoom=True` (default) each panel gets a small
    inset showing only the cold-end region where BE or FD departs
    from MB by more than `zoom_threshold` -- i.e. the quantum-
    degeneracy window. The inset is skipped if no such departure is
    detected, or if MB data isn't supplied.

    Parameters
    ----------
    T_b, T_f : sequence of float
        Sample temperature at each step (from `results['T']`).
    thermo_b, thermo_f : dict
        Per-step thermodynamics from `load_thermodynamics(...)['results']`.
        Must contain keys `Omega`, `S`, `P`, `H`, `F`, `G`.
    trap_name : str
        Trap label for subplot titles.
    T_mb : sequence of float, optional
        Sample temperature of the MB run.
    thermo_mb : dict, optional
        Per-step MB thermodynamics. Must contain the same six state-
        function keys. If either `T_mb` or `thermo_mb` is None, the
        MB overlay and the zoom inset are both skipped.
    n_b, n_f, n_mb : int, optional
        Trim each statistics' arrays to this length.
    figsize : tuple
        Default (24, 14) for 8×7-inch panels.
    log_x, log_y : bool
        Axis scaling. With `log_y=True`, |Ω|, |F|, |G| are plotted.
    units_energy : str
        Energy unit label for y-axis labels. Default "J"; use "eV"
        for traps in the eV unit system.
    zoom : bool
        Whether to add the cold-end inset. Default True.
    zoom_threshold : float
        Relative |Δ|/|MB| above which the system is considered
        "still quantum". Default 0.01 (1 %).
    zoom_position : tuple
        Inset placement (x, y, w, h) in axes-fraction coords of the
        parent. Default upper-left (0.06, 0.55, 0.40, 0.38).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_b = n_b or len(T_b)
    n_f = n_f or len(T_f)
    has_mb = T_mb is not None and thermo_mb is not None
    n_mb = n_mb or (len(T_mb) if has_mb else 0)

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
            T_mb=(T_mb[:n_mb] if has_mb else None),
            q_mb=(thermo_mb[key][:n_mb] if has_mb else None),
            zoom=zoom, zoom_threshold=zoom_threshold,
            zoom_position=zoom_position,
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
    T_mb: Optional[Sequence[float]] = None,
    thermo_mb: Optional[dict] = None,
    n_b: Optional[int] = None,
    n_f: Optional[int] = None,
    n_mb: Optional[int] = None,
    figsize: tuple = (16, 14),
    log_x: bool = True,
    log_y: bool = True,
    units_energy: str = "J",
    zoom: bool = True,
    zoom_threshold: float = 0.01,
    zoom_position: Optional[tuple] = None,
):
    """2×2 grid of thermal coefficients vs sample temperature.

    Panels: C_V, C_P, κ_T, B_P. Same overlay (BE/FD/MB) and zoom
    conventions as `plot_state_functions`. In the MB limit these are
    closed-form algebraic functions of (N, T) -- C_V = sNk_B,
    C_P = (s+1)Nk_B, κ_T = V_g/(Nk_BT), B_P = 1/T (see
    `mb_limit_based.pdf` §5) -- so the cold-end inset isolates the
    regime where degeneracy-induced curvature appears in BE / FD.

    Parameters
    ----------
    Same conventions as `plot_state_functions`. The thermo dicts must
    contain keys `CV`, `CP`, `kappa_T`, `B_P`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_b = n_b or len(T_b)
    n_f = n_f or len(T_f)
    has_mb = T_mb is not None and thermo_mb is not None
    n_mb = n_mb or (len(T_mb) if has_mb else 0)

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
            T_mb=(T_mb[:n_mb] if has_mb else None),
            q_mb=(thermo_mb[key][:n_mb] if has_mb else None),
            zoom=zoom, zoom_threshold=zoom_threshold,
            zoom_position=zoom_position,
        )

    fig.suptitle(
        f"{trap_name} trap — Thermal coefficients",
        fontsize=15, y=1.0,
    )
    fig.tight_layout()
    return fig

"""Additions to evap_cool/plots.py for multi-trap dimensionless overview.

Paste the contents of this file at the bottom of `plots.py` (or insert
the named blocks in the indicated locations). The new code adds:

  * TRAP_COLORS, TRAP_LABELS, DIM_STAT_STYLES   — module-level style dicts,
    placed next to the existing PLOT_COLORS / PLOT_LABELS.
  * _DIM_REGISTRY, _PANEL_LAYOUTS               — internal lookup tables.
  * _safe_float_array, _normalize_thermo        — internal helpers.
  * plot_dimensionless_overview                 — the public entry point.

Nothing here mutates existing functions, so the diff is purely additive
and safe to land alongside the existing per-trap plotters.

To make the new function importable, append `plot_dimensionless_overview`
(and optionally the new style dicts) to the `from .plots import (...)`
block in `evap_cool/__init__.py`.

API summary
-----------
    plot_dimensionless_overview(
        traps,                   # list[dict], one entry per trap
        *,
        panels="headline",       # "headline" (2x2) | "full" (2x4)
        figsize=None,            # default per-layout
        log_x=True, log_y=True,
        show_mb_asymptotes=True,
    ) -> matplotlib.figure.Figure

    Each entry of `traps` has the shape

        {
            "name":  "Box",                      # display name
            "s":     1.5,                        # density-of-states exponent
            "V_g":   1e-11,                      # generalized volume
            "kB":    1.380649e-23,               # Boltzmann constant in
                                                 #   the trap's unit system
            "color": "tab:blue",                 # optional override
            "bosons":   {"T": [...], "N": [...], "thermo": {...}},
            "fermions": {"T": [...], "N": [...], "thermo": {...}},
            "mb":       {"T": [...], "N": [...], "thermo": {...}},
        }

    Missing statistics are skipped silently. Thermo dicts use the same
    key names produced by `post_processing.compute_run_thermodynamics`
    (`Omega`, `S`, `P`, `H`, `F`, `G`, `CV`, `CP`, `kappa_T`, `B_P`).
"""
TRAP_COLORS = {
    "box":        "tab:blue",
    "oscillator": "tab:green",
    "quadrupole": "tab:red",
}

TRAP_LABELS = {
    "box":        "Box",
    "oscillator": "Oscillator",
    "quadrupole": "Quadrupole",
}

DIM_STAT_STYLES = {
    "mb":       dict(linestyle="-",  linewidth=1.8, alpha=0.90),
    "bosons":   dict(linestyle="--", linewidth=1.8, alpha=0.90),
    "fermions": dict(linestyle=":",  linewidth=2.6, alpha=0.95),
}


# =============================================================================
# Dimensionless quantity registry
# =============================================================================
# Each entry says how to build the dimensionless array from the per-step
# arrays (T, N) and the thermo dict, plus the panel cosmetics and the MB
# high-T limit (a function of s — None if no clean asymptote exists).
#
# Normalization rules:
#   div_NkT          y = q / (N kB T)
#   div_NkB          y = q / (N kB)
#   mul_T            y = q * T
#   mul_Vg_div_NkT   y = q * V_g / (N kB T)
#   mul_NkT_div_Vg   y = q * N kB T / V_g
_DIM_REGISTRY = {
    "Omega_red": dict(
        dim_key="Omega", norm="div_NkT",
        title=r"$|\Omega| / (N k_B T)$",
        ylabel=r"$|\Omega| / (N k_B T)$",
        mb=lambda s: 1.0, signed=True,
    ),
    "PVg_red": dict(
        dim_key="P", norm="mul_Vg_div_NkT",
        title=r"$P V_g / (N k_B T)$",
        ylabel=r"$P V_g / (N k_B T)$",
        mb=lambda s: 1.0, signed=False,
    ),
    "S_red": dict(
        dim_key="S", norm="div_NkB",
        title=r"$S / (N k_B)$",
        ylabel=r"$S / (N k_B)$",
        mb=None, signed=False,
    ),
    "H_red": dict(
        dim_key="H", norm="div_NkT",
        title=r"$H / (N k_B T)$",
        ylabel=r"$H / (N k_B T)$",
        mb=lambda s: s + 1, signed=False,
    ),
    "F_red": dict(
        dim_key="F", norm="div_NkT",
        title=r"$|F| / (N k_B T)$",
        ylabel=r"$|F| / (N k_B T)$",
        mb=None, signed=True,
    ),
    "G_red": dict(
        dim_key="G", norm="div_NkT",
        title=r"$|G| / (N k_B T)$",
        ylabel=r"$|G| / (N k_B T)$",
        mb=None, signed=True,
    ),
    "CV_red": dict(
        dim_key="CV", norm="div_NkB",
        title=r"$C_V / (N k_B)$",
        ylabel=r"$C_V / (N k_B)$",
        mb=lambda s: s, signed=False,
    ),
    "CP_red": dict(
        dim_key="CP", norm="div_NkB",
        title=r"$C_P / (N k_B)$",
        ylabel=r"$C_P / (N k_B)$",
        mb=lambda s: s + 1, signed=False,
    ),
    "kappa_red": dict(
        dim_key="kappa_T", norm="mul_NkT_div_Vg",
        title=r"$\kappa_T \, N k_B T / V_g$",
        ylabel=r"$\kappa_T \, N k_B T / V_g$",
        mb=lambda s: 1.0, signed=False,
    ),
    "BPT": dict(
        dim_key="B_P", norm="mul_T",
        title=r"$B_P \cdot T$",
        ylabel=r"$B_P \cdot T$",
        mb=lambda s: 1.0, signed=False,
    ),
}

# Two pre-defined layouts. The "headline" version is the one to use in
# the paper; "full" exists for diagnostics / supplementary material.
_PANEL_LAYOUTS = {
    "headline": dict(
        keys=["CV_red", "Omega_red", "kappa_red", "BPT"],
        shape=(2, 2),
        figsize=(13, 9),
    ),
    "full": dict(
        keys=["Omega_red", "S_red",     "H_red",     "PVg_red",
              "CV_red",    "CP_red",    "kappa_red", "BPT"],
        shape=(2, 4),
        figsize=(20, 10),
    ),
}


# =============================================================================
# Internal helpers
# =============================================================================
def _safe_float_array(seq) -> Optional[np.ndarray]:
    """Cast a sequence to a float64 array, mapping None / non-numeric -> NaN.

    Returns None if `seq` is None. NaN propagation through the
    normalization is what lets partial / halted runs render correctly
    without manual masking.
    """
    if seq is None:
        return None
    return np.array([_safe_float(v) for v in seq], dtype=float)


def _normalize_thermo(
    T: Sequence[float],
    N: Sequence[float],
    thermo: dict,
    V_g: float,
    kB: float,
) -> dict:
    """Build dimensionless arrays from per-step (T, N) and the thermo dict.

    Returns a dict keyed by the entries in `_DIM_REGISTRY` plus `'T'`.
    Missing dimensional inputs propagate as NaN; missing thermo keys
    map to `None`.

    Parameters
    ----------
    T, N : sequence of float
        Per-step sample temperature and particle number from the source
        evap run (`results['T']`, `results['N']`).
    thermo : dict
        Per-step thermodynamics from
        `load_thermodynamics(...)['results']`. May contain a subset of
        `_DIM_REGISTRY[*]['dim_key']`; absent keys are skipped.
    V_g : float
        Trap global volume (constant per run).
    kB : float
        Boltzmann constant in the trap's unit system. Cancels against
        the energy unit of the thermo arrays so the dimensionless
        ratio is a pure number regardless of whether the source file
        is in SI or eV.
    """
    T_arr = _safe_float_array(T)
    N_arr = _safe_float_array(N)
    NkT = N_arr * kB * T_arr
    NkB = N_arr * kB

    out = {"T": T_arr}
    for red_key, spec in _DIM_REGISTRY.items():
        dim_key = spec["dim_key"]
        if dim_key not in thermo:
            out[red_key] = None
            continue
        q = _safe_float_array(thermo[dim_key])
        norm = spec["norm"]
        if norm == "div_NkT":
            out[red_key] = q / NkT
        elif norm == "div_NkB":
            out[red_key] = q / NkB
        elif norm == "mul_T":
            out[red_key] = q * T_arr
        elif norm == "mul_Vg_div_NkT":
            out[red_key] = q * V_g / NkT
        elif norm == "mul_NkT_div_Vg":
            out[red_key] = q * NkT / V_g
        else:                                                  # pragma: no cover
            raise ValueError(f"Unknown normalization rule: {norm!r}")
    return out


def _format_s(s: float) -> str:
    """Render `s` as `'3'` or `'9/2'` for legend titles."""
    if abs(s - round(s)) < 1e-9:
        return f"{int(round(s))}"
    if abs(2 * s - round(2 * s)) < 1e-9:
        return f"{int(round(2 * s))}/2"
    return f"{s:g}"


# =============================================================================
# Public: multi-trap dimensionless overview
# =============================================================================
def plot_dimensionless_overview(
    traps: Sequence[dict],
    *,
    panels: str = "headline",
    figsize: Optional[tuple] = None,
    log_x: bool = True,
    log_y: bool = True,
    show_mb_asymptotes: bool = True,
):
    """Single-figure overlay of dimensionless thermodynamics across traps.

    Each panel shows one dimensionless quantity vs sample temperature `T`,
    with up to nine curves overlaid: three traps (color) times three
    statistics (line style). Replaces the six per-trap dimensional
    figures with one figure that captures the universal classical-vs-
    quantum story.

    Normalization (per-step):
        |Omega|/(N kB T),  S/(N kB),  H/(N kB T),  P V_g/(N kB T),
        C_V/(N kB),  C_P/(N kB),  kappa_T N kB T / V_g,  B_P T

    Each is a pure number whose MB high-`T` limit depends only on the
    density-of-states exponent `s` (1, s, s+1 as appropriate). The
    paper text in Sec. II.C and Sec. IV can refer to these limits
    directly without unit caveats.

    Parameters
    ----------
    traps : sequence of dict
        One dict per trap. Required keys:
            `name`     str.   Display name. If `name.lower()` is in
                              `TRAP_COLORS` the color is auto-assigned.
            `s`        float. Density-of-states exponent.
            `V_g`      float. Generalized volume (constant).
            `kB`       float. Boltzmann constant in the trap's unit
                              system (J/K for SI, eV/K for eV).
        Optional keys:
            `color`    str.   Matplotlib color, overrides TRAP_COLORS.
            `bosons`   dict.  Bose-Einstein run, shape
                              `{"T": ..., "N": ..., "thermo": ...}`.
                              The thermo dict uses the same keys as
                              `load_thermodynamics(...)['results']`.
            `fermions` dict.  Fermi-Dirac run, same shape.
            `mb`       dict.  Maxwell-Boltzmann run, same shape.
        Missing statistics are skipped silently; missing thermo keys
        suppress only the corresponding panel for that trap/statistic.

    panels : str
        Layout selector.
            `"headline"` -> 2x2 panels: C_V, |Omega|, kappa_T, B_P*T.
                            Recommended for the main paper figure.
            `"full"`     -> 2x4 panels: adds S, H, C_P, P*V_g.
                            Useful as supplementary material.

    figsize : tuple, optional
        Figure size. Defaults to the layout's built-in choice
        ((13, 9) for "headline", (20, 10) for "full").

    log_x, log_y : bool
        Axis scaling. With `log_y=True`, sign-changing quantities (Omega,
        F, G) are plotted in absolute value, indicated by `|.|` in the
        panel title.

    show_mb_asymptotes : bool
        Draw a faint horizontal line at the MB high-`T` limit for each
        trap on panels where the limit is a constant function of `s`
        (C_V, C_P, H, |Omega|, kappa_T, B_P*T). The MB curves
        themselves (drawn as solid lines if `mb` data is supplied)
        already do this implicitly; the asymptote lines act as a
        legend-free reminder of which `s` each color corresponds to.

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    Build the input dicts from the per-trap source + thermo JSON pair
    saved by `evap_cool.post_processing`::

        from evap_cool import (
            load_run, load_thermodynamics, BoxTrap,
            QuadrupoleTrap, OscillatorTrap,
            plot_dimensionless_overview,
        )

        def trap_dict(name, s, V_g, kB, session):
            entries = {"name": name, "s": s, "V_g": V_g, "kB": kB}
            for stat_key, stat_label in (("bosons",   "bosons"),
                                         ("fermions", "fermions"),
                                         ("mb",       "mb")):
                src_path = session / f"{name.lower()}_{stat_label}.json"
                thr_path = session / f"{name.lower()}_{stat_label}_thermo.json"
                if not (src_path.exists() and thr_path.exists()):
                    continue
                src = load_run(src_path)["results"]
                thr = load_thermodynamics(thr_path)["results"]
                entries[stat_key] = {"T": src["T"], "N": src["N"], "thermo": thr}
            return entries

        box  = BoxTrap(V=1e-11)
        quad = QuadrupoleTrap(A_bar=1e-15)
        osc  = OscillatorTrap(omega=2 * 3.141592653589793 * 100)

        traps = [
            trap_dict("Box",        box.s,  box.volume_global,  box.kB,  session),
            trap_dict("Oscillator", osc.s,  osc.volume_global,  osc.kB,  session),
            trap_dict("Quadrupole", quad.s, quad.volume_global, quad.kB, session),
        ]
        fig = plot_dimensionless_overview(traps, panels="headline")
        fig.savefig("paper_fig3_dimensionless.png", dpi=300, bbox_inches="tight")
    """
    if panels not in _PANEL_LAYOUTS:
        raise ValueError(
            f"panels={panels!r} not in {list(_PANEL_LAYOUTS)}"
        )
    layout = _PANEL_LAYOUTS[panels]
    nrows, ncols = layout["shape"]
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or layout["figsize"])
    axes_flat = list(axes.flat) if (nrows * ncols > 1) else [axes]

    # ----- Pre-build dimensionless arrays for every (trap, stat) pair.
    cache = []
    for tdict in traps:
        name = tdict["name"]
        color = tdict.get("color") or TRAP_COLORS.get(name.lower(), "0.3")
        V_g, kB, s_val = tdict["V_g"], tdict["kB"], tdict["s"]
        for stat_key in ("mb", "bosons", "fermions"):
            if stat_key not in tdict:
                continue
            src = tdict[stat_key]
            dimless = _normalize_thermo(
                T=src["T"], N=src["N"], thermo=src["thermo"],
                V_g=V_g, kB=kB,
            )
            cache.append(dict(
                color=color, stat=stat_key, name=name, s=s_val, data=dimless,
            ))

    # ----- Draw each panel.
    for ax, red_key in zip(axes_flat, layout["keys"]):
        spec = _DIM_REGISTRY[red_key]

        for c in cache:
            y = c["data"].get(red_key)
            if y is None:
                continue
            T = c["data"]["T"]
            y_plot = np.abs(y) if (log_y and spec["signed"]) else y
            ax.plot(T, y_plot, color=c["color"], **DIM_STAT_STYLES[c["stat"]])

        # Faint MB asymptotes — one per trap, on panels that admit one.
        if show_mb_asymptotes and spec["mb"] is not None:
            seen = set()
            for c in cache:
                if c["name"] in seen:
                    continue
                seen.add(c["name"])
                ax.axhline(spec["mb"](c["s"]),
                           color=c["color"], linestyle="-",
                           linewidth=0.6, alpha=0.25)

        if log_x: ax.set_xscale("log")
        if log_y: ax.set_yscale("log")
        ax.set_xlabel(r"Sample temperature $T$ [K]", fontsize=11)
        ax.set_ylabel(spec["ylabel"], fontsize=11)
        ax.set_title(spec["title"], fontsize=12)
        ax.grid(True, which="both", alpha=0.15)
        ax.tick_params(axis="both", labelsize=9)

    # ----- Two-block legend below the grid: traps (color) | stats (style).
    trap_handles = []
    seen_names = set()
    for c in cache:
        if c["name"] in seen_names:
            continue
        seen_names.add(c["name"])
        trap_handles.append(plt.Line2D(
            [], [], color=c["color"], linewidth=2.0,
            label=f"{c['name']} ($s = {_format_s(c['s'])}$)",
        ))

    stat_label_full = {"mb":       "Maxwell-Boltzmann",
                       "bosons":   "Bose-Einstein",
                       "fermions": "Fermi-Dirac"}
    seen_stats = {c["stat"] for c in cache}
    stat_handles = [
        plt.Line2D([], [], color="0.3", label=stat_label_full[s_key],
                   **DIM_STAT_STYLES[s_key])
        for s_key in ("mb", "bosons", "fermions") if s_key in seen_stats
    ]

    leg1 = fig.legend(
        handles=trap_handles, loc="upper center",
        bbox_to_anchor=(0.30, 0.02), ncol=len(trap_handles),
        fontsize=10, title="Trap (color)", title_fontsize=10, frameon=True,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=stat_handles, loc="upper center",
        bbox_to_anchor=(0.72, 0.02), ncol=len(stat_handles),
        fontsize=10, title="Statistic (line style)",
        title_fontsize=10, frameon=True,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    return fig