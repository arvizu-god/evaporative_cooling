"""Multi-trap overlay of self-normalized thermodynamics.

Reads the ``*_norm.json`` files produced by `evap_cool.normalization` and
draws every quantity as the dimensionless ratio ``X_i / X_0`` against the
normalized temperature ``T_i / T_0``. All traps and statistics are
overlaid on each panel of a single figure.

Encoding
--------
* **Trap -> one fixed (pastel) colour.** Each trap/potential has a single
  colour (see `TRAP_COLORS`): box = red, quadrupole = green,
  oscillator = navy, ``box2d_osc1d`` = orange, ``osc2d_box1d`` = violet.
* **Statistic -> marker shape.** Maxwell-Boltzmann is a circle,
  Bose-Einstein a square, Fermi-Dirac a triangle, drawn as outlines only
  (unfilled) so the shape — not a fill — carries the statistic.
* **Reference lines.** Faint dashed lines at ``y = 1`` and ``x = 1`` mark
  the common initial state where every curve begins.

Handling divergences
---------------------
Several coefficients diverge at the cold end (small ``T_i / T_0``). Rather
than discarding those points (the divergence is physical, near the
BEC/degeneracy edge) the view is adapted per panel:

* ``kappa_T`` and ``B_P`` are drawn on a **log y-axis** (they are positive
  and span many decades);
* the bounded panels (``Omega``, ``C_V``, ``C_P``, ``C_P - C_V``) use
  **robust y-limits** — the axis is clipped to a percentile of the data
  (default 99th) so a few spikes do not flatten the bulk. Every point is
  still plotted; only the axis range changes.

A ``trim_tail=N`` knob optionally drops the last ``N`` steps for a cleaner
presentation figure (off by default).

Quick usage
-----------
    from evap_cool import (
        normalize_session, build_normalized_traps,
        plot_dimensionless_overview, plot_cp_minus_cv,
    )

    normalize_session(session)                     # writes *_norm.json
    traps = build_normalized_traps(session)        # auto-discovers traps
    fig = plot_dimensionless_overview(traps)
    fig.savefig("normalized_overview.png", dpi=300, bbox_inches="tight")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.scale as mscale
import numpy as np

from .normalization import load_normalized


# =============================================================================
# Style
# =============================================================================
# One fixed colour per trap/potential, drawn from the colourblind-safe
# Okabe-Ito palette. The three statistics of a trap are shades of this base
# (see _stat_color): base for MB, lighter for BE, darker for FD.
TRAP_COLORS = {
    "box":         "#E69F00",   # orange (peach)
    "quadrupole":  "#009E73",   # bluish green
    "oscillator":  "#0072B2",   # blue (navy)
    "box2d_osc1d": "#D55E00",   # vermilion   (the "Box2d" potential)
    "osc2d_box1d": "#CC79A7",   # reddish purple (the "Box1d" potential)
}
_FALLBACK_COLOR = "#9e9e9e"

TRAP_LABELS = {
    "box":         "Box",
    "quadrupole":  "Quadrupole",
    "oscillator":  "Oscillator",
    "box2d_osc1d": "Box2d_osc1d",
    "osc2d_box1d": "Osc2d_box1d",
}

# Statistic identity -> marker shape (consistent across all traps).
STAT_MARKERS = {
    "mb":       "o",   # circle
    "bosons":   "s",   # square
    "fermions": "^",   # triangle
}

STAT_LABELS = {
    "mb":       "Maxwell-Boltzmann",
    "bosons":   "Bose-Einstein",
    "fermions": "Fermi-Dirac",
}

_STAT_ORDER = ("mb", "bosons", "fermions")

# Panels: (results key, panel title). x-axis is always T_i / T_0.
_PANELS = [
    ("Omega",       r"$\Omega_i / \Omega_0$"),
    ("CV",          r"$C_{V,i} / C_{V,0}$"),
    ("CP",          r"$C_{P,i} / C_{P,0}$"),
    ("kappa_T",     r"$\kappa_{T,i} / \kappa_{T,0}$"),
    ("B_P",         r"$B_{P,i} / B_{P,0}$"),
    ("CP_minus_CV", r"$(C_P - C_V)_i / (C_P - C_V)_0$"),
]

# Panels drawn on a log y-axis by default (positive, many-decade range).
_DEFAULT_LOG_PANELS = ("kappa_T", "B_P")

_XLABEL = r"$T_i / T_0$"


# =============================================================================
# Internal helpers
# =============================================================================
_VALID_SCALES = set(mscale.get_scale_names())


def _resolve_scale(scale: Optional[str], log_flag: bool) -> str:
    """Resolve an axis scale.

    An explicit `scale` string (any matplotlib scale, e.g. "linear", "log",
    "symlog", "asinh", "logit") wins; otherwise fall back to "log" if
    `log_flag` else "linear". Raises ValueError on an unknown scale name.
    """
    s = scale if scale is not None else ("log" if log_flag else "linear")
    if s not in _VALID_SCALES:
        raise ValueError(
            f"unknown axis scale {s!r}; valid options: {sorted(_VALID_SCALES)}"
        )
    return s


def _to_float_array(seq) -> Optional[np.ndarray]:
    """Cast a sequence to float64, mapping None / non-numeric -> NaN."""
    if seq is None:
        return None
    out = np.empty(len(seq), dtype=float)
    for i, v in enumerate(seq):
        try:
            out[i] = float(v) if v is not None else math.nan
        except (TypeError, ValueError):
            out[i] = math.nan
    return out


def _trap_key(tdict: dict) -> str:
    return (tdict.get("key") or tdict.get("name") or "").lower()


def _trap_color(key: str):
    """One colour per trap, with substring fallbacks for combo potentials."""
    key = (key or "").lower()
    if key in TRAP_COLORS:
        return TRAP_COLORS[key]
    if "box2d" in key:
        return TRAP_COLORS["box2d_osc1d"]
    if "box1d" in key:
        return TRAP_COLORS["osc2d_box1d"]
    if "quad" in key:
        return TRAP_COLORS["quadrupole"]
    if "osc" in key:
        return TRAP_COLORS["oscillator"]
    if "box" in key:
        return TRAP_COLORS["box"]
    return _FALLBACK_COLOR


def _trap_display(tdict: dict) -> str:
    name = tdict.get("name")
    if name:
        return name
    return TRAP_LABELS.get(_trap_key(tdict), _trap_key(tdict))


def _series_xy(res: dict, key: str, *, trim_tail: int, stride: int):
    """Return (x, y) float arrays for one quantity, after trim + stride."""
    x = _to_float_array(res.get("T"))
    y = _to_float_array(res.get(key))
    if x is None or y is None:
        return None, None
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    if trim_tail > 0 and n > trim_tail:
        x, y = x[:-trim_tail], y[:-trim_tail]
    if stride > 1:
        x, y = x[::stride], y[::stride]
    return x, y


def _robust_ylim(ax, ys: list, pct: float, *, ref: float = 1.0):
    """Clip the y-axis to [low_pct, high_pct] so spikes don't flatten the bulk.

    The reference value `ref` (=1, the initial state) and 0 are kept in
    view. All points remain plotted; only the visible range changes.
    """
    finite = [y[np.isfinite(y)] for y in ys if y is not None]
    finite = [a for a in finite if a.size]
    if not finite:
        return
    allv = np.concatenate(finite)
    lo = float(np.nanpercentile(allv, 100.0 - pct))
    hi = float(np.nanpercentile(allv, pct))
    lo = min(lo, 0.0, ref)
    hi = max(hi, ref)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return
    pad = 0.05 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def _scatter_panel(
    ax,
    traps: Sequence[dict],
    key: str,
    *,
    marker_size: float,
    stride: int,
    trim_tail: int,
    reference_lines: bool,
    xscale: str,
    yscale: str,
    robust_ylim: bool,
    ylim_pct: float,
):
    """Scatter one normalized quantity for every (trap, statistic) on `ax`."""
    ys_for_limits = []
    for tdict in traps:
        color = _trap_color(_trap_key(tdict))
        for stat in _STAT_ORDER:
            res = tdict.get(stat)
            if not res:
                continue
            x, y = _series_xy(res, key, trim_tail=trim_tail, stride=stride)
            if x is None or y is None:
                continue
            ax.scatter(
                x, y,
                s=marker_size,
                marker=STAT_MARKERS.get(stat, "o"),
                facecolors="none",                 # outline only
                edgecolors=[color],
                linewidths=0.7,
                alpha=0.85,
                zorder=2,
            )
            ys_for_limits.append(y)

    if reference_lines:
        ax.axhline(1.0, color="0.5", lw=0.8, ls="--", alpha=0.6, zorder=0)
        ax.axvline(1.0, color="0.5", lw=0.8, ls="--", alpha=0.6, zorder=0)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if yscale == "linear" and robust_ylim:
        _robust_ylim(ax, ys_for_limits, ylim_pct)

    ax.set_xlabel(_XLABEL, fontsize=11)
    ax.grid(True, which="both", alpha=0.15)
    ax.tick_params(axis="both", labelsize=9)


def _present_stats(traps: Sequence[dict]) -> list:
    seen = {stat for t in traps for stat in _STAT_ORDER if t.get(stat)}
    return [s for s in _STAT_ORDER if s in seen]


def _trap_handles(traps: Sequence[dict]) -> list:
    handles = []
    for t in traps:
        name = _trap_display(t)
        if not name:
            continue
        c = _trap_color(_trap_key(t))
        handles.append(plt.Line2D(
            [], [], linestyle="none", marker="s", markersize=8,
            markerfacecolor=c, markeredgecolor=c, label=name,
        ))
    return handles


def _stat_handles(traps: Sequence[dict]) -> list:
    return [
        plt.Line2D(
            [], [], linestyle="none", marker=STAT_MARKERS[s], markersize=8,
            markerfacecolor="none", markeredgecolor="0.25", markeredgewidth=1.1,
            label=STAT_LABELS[s],
        )
        for s in _present_stats(traps)
    ]


def _add_legends(fig, traps: Sequence[dict], *, stat_ax, stat_loc="upper left"):
    """Statistic legend INSIDE `stat_ax`; trap legend OUTSIDE (figure bottom)."""
    sh = _stat_handles(traps)
    if sh and stat_ax is not None:
        stat_ax.legend(
            handles=sh, loc=stat_loc, fontsize=8,
            title="Statistic", title_fontsize=8,
            framealpha=0.9, borderpad=0.4, handletextpad=0.4, labelspacing=0.3,
        )
    th = _trap_handles(traps)
    if th:
        fig.legend(
            handles=th, loc="lower center", bbox_to_anchor=(0.5, 0.0),
            ncol=min(len(th), 5), fontsize=10,
            title="Trap (potential)", title_fontsize=10, frameon=True,
        )


# =============================================================================
# Public: loader
# =============================================================================
def _derive_trap_names(session_dir: Path) -> list:
    """Unique trap base names parsed from <trap>_<stat>_norm.json filenames."""
    names = []
    for p in sorted(session_dir.glob("*_norm.json")):
        stem = p.stem[:-5] if p.stem.endswith("_norm") else p.stem  # drop _norm
        trap = stem.rsplit("_", 1)[0]                               # drop _<stat>
        if trap and trap not in names:
            names.append(trap)
    return names


def build_normalized_traps(
    session_dir: Union[str, Path],
    traps: Optional[Sequence[str]] = None,
    *,
    stat_files: Optional[dict] = None,
) -> list:
    """Load ``*_norm.json`` files from a session into the plotting structure.

    Parameters
    ----------
    session_dir : str or Path
        Folder containing the normalized files (e.g. the session dir).
    traps : sequence of str, optional
        Trap base names to look for. If omitted, they are auto-derived
        from the ``*_norm.json`` filenames present.
    stat_files : dict, optional
        Override the statistic-key -> filename-label map. Default
        ``{"mb": "mb", "bosons": "bosons", "fermions": "fermions"}``.

    Returns
    -------
    list of dict
        One entry per trap with at least one file, shaped
        ``{"name": <display>, "key": <base>, "mb": {...}, ...}`` where each
        statistic value is a normalized `results` dict. The ``key`` field
        drives the colour lookup.
    """
    session_dir = Path(session_dir)
    if traps is None:
        traps = _derive_trap_names(session_dir)
    labels = stat_files or {"mb": "mb", "bosons": "bosons", "fermions": "fermions"}

    out = []
    for trap in traps:
        key = trap.lower()
        entry = {"name": TRAP_LABELS.get(key, trap), "key": key}
        found = False
        for stat_key, label in labels.items():
            path = session_dir / f"{key}_{label}_norm.json"
            if not path.exists():
                continue
            entry[stat_key] = load_normalized(path)["results"]
            found = True
        if found:
            out.append(entry)
    return out


# =============================================================================
# Public: figures
# =============================================================================
def plot_dimensionless_overview(
    traps: Sequence[dict],
    *,
    figsize: Optional[tuple] = (16, 10),
    log_x: bool = True,
    log_panels: Sequence[str] = _DEFAULT_LOG_PANELS,
    robust_ylim: bool = True,
    ylim_pct: float = 99.0,
    marker_size: float = 9.0,
    stride: int = 1,
    trim_tail: int = 0,
    reference_lines: bool = True,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
):
    """2x3 overlay of self-normalized thermodynamics across traps.

    Panels: ``Omega``, ``C_V``, ``C_P``, ``kappa_T``, ``B_P`` and
    ``(C_P - C_V)`` — each as ``X_i / X_0`` vs ``T_i / T_0``. Trap is read
    from the colour, statistic from the marker shape.

    Parameters
    ----------
    traps : sequence of dict
        One entry per trap (from `build_normalized_traps`).
    figsize : tuple, optional
        Figure size. Default ``(16, 10)``.
    log_x : bool
        Log-scale the x-axis. Ignored if `xscale` is given. Default True.
    log_panels : sequence of str
        Panel keys drawn on a log y-axis (the per-panel default). Ignored if
        `yscale` is given. Default ``("kappa_T", "B_P")``.
    robust_ylim : bool
        On linear-y panels, clip the y-axis to `ylim_pct` of the data so
        cold-end spikes do not flatten the bulk. Default True.
    ylim_pct : float
        Upper percentile for the robust clip (lower is ``100 - ylim_pct``).
    marker_size : float
        Scatter marker area in points^2. Default 9.
    stride : int
        Plot every `stride`-th step.
    trim_tail : int
        Drop the last `trim_tail` steps before plotting (0 keeps all).
    reference_lines : bool
        Draw the faint ``x = 1`` / ``y = 1`` initial-state guides.
    xscale, yscale : str, optional
        Explicit matplotlib axis scales ("linear", "log", "symlog", "asinh",
        "logit", ...). `xscale` overrides `log_x`; `yscale`, when given,
        forces *all* panels to that y-scale (overriding `log_panels`).

    Returns
    -------
    matplotlib.figure.Figure
    """
    xs = _resolve_scale(xscale, log_x)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes_flat = list(axes.flat)

    for ax, (key, title) in zip(axes_flat, _PANELS):
        panel_y = _resolve_scale(yscale, key in log_panels)
        _scatter_panel(
            ax, traps, key,
            marker_size=marker_size, stride=stride, trim_tail=trim_tail,
            reference_lines=reference_lines, xscale=xs,
            yscale=panel_y,
            robust_ylim=robust_ylim, ylim_pct=ylim_pct,
        )
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12)

    for ax in axes_flat[len(_PANELS):]:
        ax.set_visible(False)

    # Statistic legend inside the first panel; trap legend below the figure.
    _add_legends(fig, traps, stat_ax=axes_flat[0], stat_loc="upper left")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    return fig


def plot_cp_minus_cv(
    traps: Sequence[dict],
    *,
    figsize: Optional[tuple] = (8, 6),
    log_x: bool = True,
    log_y: bool = False,
    robust_ylim: bool = True,
    ylim_pct: float = 99.0,
    marker_size: float = 14.0,
    stride: int = 1,
    trim_tail: int = 0,
    reference_lines: bool = True,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
):
    """Standalone figure of the normalized heat-capacity difference.

    Plots ``(C_P - C_V)_i / (C_P - C_V)_0`` (the normalized *raw*
    difference, stored under ``CP_minus_CV``) vs ``T_i / T_0`` for every
    trap and statistic, using the same colour / marker encoding as
    `plot_dimensionless_overview`.

    `xscale` / `yscale` accept any matplotlib scale and override
    `log_x` / `log_y`. The robust y-clip applies only on a linear y-axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    xs = _resolve_scale(xscale, log_x)
    ys = _resolve_scale(yscale, log_y)
    fig, ax = plt.subplots(figsize=figsize)
    _scatter_panel(
        ax, traps, "CP_minus_CV",
        marker_size=marker_size, stride=stride, trim_tail=trim_tail,
        reference_lines=reference_lines, xscale=xs,
        yscale=ys,
        robust_ylim=(robust_ylim and ys == "linear"), ylim_pct=ylim_pct,
    )
    title = r"$(C_P - C_V)_i / (C_P - C_V)_0$"
    ax.set_ylabel(title, fontsize=12)
    ax.set_title(title, fontsize=13)
    _add_legends(fig, traps, stat_ax=ax, stat_loc="upper left")
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    return fig


# =============================================================================
# Line plots (split figures) — for all traps and statistics
# =============================================================================
# Statistic -> line style (the primary statistic cue; colour shade is a
# redundant secondary cue so the figures stay colourblind-safe).
STAT_LINESTYLES = {
    "mb":       "-",    # solid
    "bosons":   "--",   # dashed
    "fermions": ":",    # dotted
}

# Shade derivation off each trap's base colour.
STAT_LIGHTEN = 0.55   # BE: blend toward white (clearly lighter; preferred)
STAT_DARKEN  = 0.38   # FD: blend toward black


def _lighten(color, amount: float):
    r, g, b = mcolors.to_rgb(color)
    return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount)


def _darken(color, amount: float):
    r, g, b = mcolors.to_rgb(color)
    return (r * (1.0 - amount), g * (1.0 - amount), b * (1.0 - amount))


def _stat_color(base_color, stat: str):
    """Base for MB, a clearly lighter tone for BE, a darker tone for FD."""
    if stat == "bosons":
        return _lighten(base_color, STAT_LIGHTEN)
    if stat == "fermions":
        return _darken(base_color, STAT_DARKEN)
    return mcolors.to_rgb(base_color)


def _stat_line_handles(traps: Sequence[dict]) -> list:
    """Neutral line-style key (MB solid / BE dashed / FD dotted)."""
    return [
        plt.Line2D([], [], color="0.2", linestyle=STAT_LINESTYLES[s], lw=1.8,
                   label=STAT_LABELS[s])
        for s in _present_stats(traps)
    ]


def _trap_figlegend(fig, traps: Sequence[dict], *, y: float = 0.0):
    th = _trap_handles(traps)
    if th:
        fig.legend(
            handles=th, loc="lower center", bbox_to_anchor=(0.5, y),
            ncol=min(len(th), 5), fontsize=10,
            title="Trap (potential)", title_fontsize=10, frameon=True,
        )


def _line_panel(
    ax, traps: Sequence[dict], key: str, title: str, *,
    abs_value: bool, xscale: str, yscale: str,
    reference_lines: bool, lw: float,
    stat_legend: bool, stat_legend_loc: str,
):
    """Line plot of one quantity for every (trap, statistic) on `ax`.

    Trap -> colour, statistic -> line style + shade. No markers.
    """
    for tdict in traps:
        base = _trap_color(_trap_key(tdict))
        for stat in _STAT_ORDER:
            res = tdict.get(stat)
            if not res:
                continue
            x = _to_float_array(res.get("T"))
            y = _to_float_array(res.get(key))
            if x is None or y is None:
                continue
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
            if abs_value:
                y = np.abs(y)
            ax.plot(
                x, y,
                color=_stat_color(base, stat),
                linestyle=STAT_LINESTYLES.get(stat, "-"),
                lw=lw, alpha=0.9, solid_capstyle="round", zorder=2,
            )

    if reference_lines:
        ax.axhline(1.0, color="0.6", lw=0.8, ls="--", alpha=0.55, zorder=0)
        ax.axvline(1.0, color="0.6", lw=0.8, ls="--", alpha=0.55, zorder=0)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(_XLABEL, fontsize=11)
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, which="both", alpha=0.15)
    ax.tick_params(axis="both", labelsize=9)

    if stat_legend:
        sh = _stat_line_handles(traps)
        if sh:
            ax.legend(
                handles=sh, loc=stat_legend_loc, fontsize=8,
                title="Statistic", title_fontsize=8, framealpha=0.9,
                handlelength=2.4, labelspacing=0.3, borderpad=0.4,
            )


# ---------------------------------------------------------------------------
# BE / FD region divider (for the kappa_T and B_P panels)
# ---------------------------------------------------------------------------
def _interp_loglog(x, y, xs):
    """Interpolate (x, y) onto xs as straight lines in log-log; NaN outside."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 2:
        return np.full(xs.shape, np.nan)
    lx, ly = np.log(x[m]), np.log(y[m])
    order = np.argsort(lx)
    lx, ly = lx[order], ly[order]
    return np.exp(np.interp(np.log(xs), lx, ly, left=np.nan, right=np.nan))


def _bundle(traps, key, stat, xs, abs_value):
    """Stack of every curve for one statistic, interpolated onto xs."""
    rows = []
    for t in traps:
        res = t.get(stat)
        if not res:
            continue
        x = _to_float_array(res.get("T"))
        y = _to_float_array(res.get(key))
        if x is None or y is None:
            continue
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        if abs_value:
            y = np.abs(y)
        rows.append(_interp_loglog(x, y, xs))
    return np.vstack(rows) if rows else None


def _add_be_fd_divider(ax, traps, key, *, abs_value=True, n_grid=240, label=True):
    """Draw a separatrix between the BE and FD bundles, where they diverge.

    The two statistics fan out into an upper and a lower bundle at the cold
    end (with MB between). This draws the geometric-mean midline between the
    upper bundle's lower edge and the lower bundle's upper edge, over the
    range where they are clearly separated, and labels the two regions.
    """
    tvals = []
    for t in traps:
        for stat in ("bosons", "fermions"):
            res = t.get(stat)
            if res and res.get("T"):
                a = _to_float_array(res["T"])
                a = a[np.isfinite(a) & (a > 0)]
                if a.size:
                    tvals.append(a)
    if not tvals:
        return
    allt = np.concatenate(tvals)
    xs = np.logspace(np.log10(allt.min()), np.log10(allt.max()), n_grid)

    be = _bundle(traps, key, "bosons", xs, abs_value)
    fd = _bundle(traps, key, "fermions", xs, abs_value)
    if be is None or fd is None:
        return

    cold = slice(0, max(1, n_grid // 3))   # small-T end, where they separate
    be_above = np.nanmedian(np.nanmedian(be, axis=0)[cold]) >= \
        np.nanmedian(np.nanmedian(fd, axis=0)[cold])
    if be_above:
        upper_lo, lower_hi = np.nanmin(be, axis=0), np.nanmax(fd, axis=0)
        up_name, lo_name = "Bose-Einstein", "Fermi-Dirac"
    else:
        upper_lo, lower_hi = np.nanmin(fd, axis=0), np.nanmax(be, axis=0)
        up_name, lo_name = "Fermi-Dirac", "Bose-Einstein"

    with np.errstate(invalid="ignore"):
        sep = np.sqrt(upper_lo * lower_hi)
        sep = np.where(upper_lo > lower_hi, sep, np.nan)   # only where separated
    if np.isfinite(sep).sum() < 2:
        return

    ax.plot(xs, sep, color="0.35", lw=1.2, ls="-.", alpha=0.85, zorder=1)
    if label:
        idx = np.where(np.isfinite(sep))[0]
        i = idx[max(0, len(idx) // 6)]
        ax.text(xs[i], sep[i] * 3.0, up_name, color="0.3", fontsize=8,
                style="italic", ha="left", va="bottom", zorder=3)
        ax.text(xs[i], sep[i] / 3.0, lo_name, color="0.3", fontsize=8,
                style="italic", ha="left", va="top", zorder=3)


# ---------------------------------------------------------------------------
# Public: the four split figures (each "for all traps and statistics")
# ---------------------------------------------------------------------------
def plot_energies_per_particle(
    traps: Sequence[dict], *,
    figsize: Optional[tuple] = (16, 5),
    log_x: bool = True, log_y: bool = True,
    lw: float = 1.6, reference_lines: bool = True,
    stat_legend_loc: str = "best",
    xscale: Optional[str] = None, yscale: Optional[str] = None,
):
    """Figure 1: |Omega/N|, |F/N|, |G/N| (per-particle, normalized), log-log.

    Each quantity is ``|(X/N)_i / (X/N)_0|`` built from the self-normalized
    lists. Trap -> colour, statistic -> line style; a statistic key sits in
    every subplot, the trap key below the figure. `xscale`/`yscale` accept
    any matplotlib scale and override `log_x`/`log_y`.
    """
    xs = _resolve_scale(xscale, log_x)
    ys = _resolve_scale(yscale, log_y)
    panels = [
        ("Omega_over_N", r"$|\Omega/N|$"),
        ("F_over_N",     r"$|F/N|$"),
        ("G_over_N",     r"$|G/N|$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (key, title) in zip(axes, panels):
        _line_panel(ax, traps, key, title, abs_value=True,
                    xscale=xs, yscale=ys, reference_lines=reference_lines,
                    lw=lw, stat_legend=True, stat_legend_loc=stat_legend_loc)
    fig.suptitle(r"Energies per particle, normalized to initial:  "
                 r"$|(X/N)_i / (X/N)_0|$", fontsize=12)
    _trap_figlegend(fig, traps)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig


def plot_compressibility(
    traps: Sequence[dict], *,
    figsize: Optional[tuple] = (12, 5),
    log_x: bool = True, log_y: bool = True,
    lw: float = 1.6, reference_lines: bool = True,
    divide_be_fd: bool = True,
    stat_legend_loc: str = "best",
    xscale: Optional[str] = None, yscale: Optional[str] = None,
):
    """Figure 2: kappa_T and B_P (normalized X_i/X_0), log-log.

    BE and FD fan out into separate trajectories at the cold end; with
    `divide_be_fd` a separatrix splits the plane into the BE and FD regions.
    `xscale`/`yscale` accept any matplotlib scale and override `log_x`/`log_y`.
    """
    xs = _resolve_scale(xscale, log_x)
    ys = _resolve_scale(yscale, log_y)
    panels = [
        ("kappa_T", r"$\kappa_{T,i} / \kappa_{T,0}$"),
        ("B_P",     r"$B_{P,i} / B_{P,0}$"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (key, title) in zip(axes, panels):
        _line_panel(ax, traps, key, title, abs_value=True,
                    xscale=xs, yscale=ys, reference_lines=reference_lines,
                    lw=lw, stat_legend=True, stat_legend_loc=stat_legend_loc)
        if divide_be_fd:
            _add_be_fd_divider(ax, traps, key, abs_value=True)
    fig.suptitle(r"Isothermal compressibility & thermal expansion "
                 r"(normalized $X_i / X_0$)", fontsize=12)
    _trap_figlegend(fig, traps)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig


def plot_heat_capacities(
    traps: Sequence[dict], *,
    figsize: Optional[tuple] = (16, 5),
    log_x: bool = True, log_y: bool = True,
    lw: float = 1.6, reference_lines: bool = True,
    stat_legend_loc: str = "best",
    xscale: Optional[str] = None, yscale: Optional[str] = None,
):
    """Figure 3: |C_V/N|, |C_P/N|, |(C_P-C_V)/N| (per-particle), log-log.

    `xscale`/`yscale` accept any matplotlib scale and override `log_x`/`log_y`
    (e.g. pass ``yscale="linear"`` for a semilog-x view).
    """
    xs = _resolve_scale(xscale, log_x)
    ys = _resolve_scale(yscale, log_y)
    panels = [
        ("CV_over_N",          r"$|C_V/N|$"),
        ("CP_over_N",          r"$|C_P/N|$"),
        ("CP_minus_CV_over_N", r"$|(C_P-C_V)/N|$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (key, title) in zip(axes, panels):
        _line_panel(ax, traps, key, title, abs_value=True,
                    xscale=xs, yscale=ys, reference_lines=reference_lines,
                    lw=lw, stat_legend=True, stat_legend_loc=stat_legend_loc)
    fig.suptitle(r"Heat capacities per particle, normalized to initial:  "
                 r"$|(X/N)_i / (X/N)_0|$", fontsize=12)
    _trap_figlegend(fig, traps)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig


def plot_n_vs_t(
    traps: Sequence[dict], *,
    figsize: Optional[tuple] = (7.5, 6),
    log_x: bool = True, log_y: bool = True,
    lw: float = 1.6, reference_lines: bool = True,
    stat_legend_loc: str = "best",
    xscale: Optional[str] = None, yscale: Optional[str] = None,
):
    """Figure 4: N_i/N_0 vs T_i/T_0, log-log.

    `xscale`/`yscale` accept any matplotlib scale and override `log_x`/`log_y`.
    """
    xs = _resolve_scale(xscale, log_x)
    ys = _resolve_scale(yscale, log_y)
    fig, ax = plt.subplots(figsize=figsize)
    _line_panel(ax, traps, "N", r"$N_i / N_0$", abs_value=False,
                xscale=xs, yscale=ys, reference_lines=reference_lines,
                lw=lw, stat_legend=True, stat_legend_loc=stat_legend_loc)
    ax.set_title(r"Population vs temperature:  $N_i/N_0$ vs $T_i/T_0$",
                 fontsize=12)
    _trap_figlegend(fig, traps, y=-0.02)
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    return fig