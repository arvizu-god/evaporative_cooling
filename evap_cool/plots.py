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