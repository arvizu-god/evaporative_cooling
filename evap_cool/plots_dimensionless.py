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
    "box":       "#1B3A4B",   
    "quadrupole":"#A0372A",   # bluish green
    "oscillator": "#C46D3B",   # blue (navy)
    "box2d_osc1d": "#4FA8B8",   # vermilion   (the "Box2d" potential)
    "osc2d_box1d": "#E8A87C",   # reddish purple (the "Box1d" potential)
}
_FALLBACK_COLOR = "#9e9e9e"

TRAP_LABELS = {
    "box":         "Box",
    "quadrupole":  "Quadrupole",
    "oscillator":  "Harmonic Oscillator",
    "box2d_osc1d": "2D Box + 1D Oscillator",
    "osc2d_box1d": "2D Oscillator + 1D Box",
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

# Canonical order (descending C_trap) + labels for the four main split figures.
_MAIN_TRAP_ORDER = ["box", "box2d_osc1d", "osc2d_box1d", "oscillator", "quadrupole"]
_MAIN_TRAP_LABELS = {
    "box":         "Box",
    "box2d_osc1d": "2D Box + 1D harmonic",
    "osc2d_box1d": "2D Harmonic + 1D Box",
    "oscillator":  "Harmonic",
    "quadrupole":  "Quadrupolar",
}


def _ordered_relabeled(traps: Sequence[dict]) -> list:
    """Reorder `traps` to _MAIN_TRAP_ORDER and relabel per _MAIN_TRAP_LABELS.

    Returns shallow copies with only 'name' overridden (keys/data untouched, so
    colours stay pinned). Unknown keys keep their order and land after the
    known ones. Does not mutate the caller's list or dicts.
    """
    rank = {k: i for i, k in enumerate(_MAIN_TRAP_ORDER)}
    out = []
    for t in sorted(traps, key=lambda t: rank.get(_trap_key(t), len(rank))):
        t2 = dict(t)
        lbl = _MAIN_TRAP_LABELS.get(_trap_key(t))
        if lbl:
            t2["name"] = lbl
        out.append(t2)
    return out


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
            ncol=min(len(th), 5), fontsize=10, frameon=False,
        )


# ---------------------------------------------------------------------------
# Labeling: 'old' (two separate, frameless legends) vs 'new' (combined block).
# Both modes are box-less. `_draw_bottom_legend` is the final layout step of
# every line figure; it both places the bottom legend and runs tight_layout.
# ---------------------------------------------------------------------------
def _resolve_labeling(labeling: Optional[str]) -> str:
    l = (labeling or "old").lower()
    if l not in ("old", "new"):
        raise ValueError(f"labeling must be 'old' or 'new', got {labeling!r}")
    return l


def _draw_bottom_legend(fig, traps: Sequence[dict], *, labeling: str,
                        top: float = 0.97):
    """Place the bottom legend per `labeling` and apply ``tight_layout``.

    ``'old'`` -> frameless bottom "Trap (potential)" legend (the per-panel
    "Statistic" boxes are drawn inside `_line_panel`). ``'new'`` -> the
    frameless combined Trap x Statistic block. Must be the last layout call.
    """
    if labeling == "new":
        fig.tight_layout(rect=[0, _LEGEND_RECT_BOTTOM, 1, top])
        _combined_legend(fig, traps)
    else:
        _trap_figlegend(fig, traps)
        fig.tight_layout(rect=[0, 0.08, 1, top])


def _line_panel(
    ax, traps: Sequence[dict], key: str, title: str, *,
    abs_value: bool, xscale: str, yscale: str,
    reference_lines: bool, lw: float,
    stat_legend: bool, stat_legend_loc: str,
    grid: bool = True,
):
    """Line plot of one quantity for every (trap, statistic) on `ax`.

    Trap -> colour, statistic -> line style + shade. No markers. `grid`
    toggles the background grid (the ``y=1`` / ``x=1`` reference lines are
    drawn independently via `reference_lines`).
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
    if grid:
        ax.grid(True, which="both", alpha=0.15)
    ax.tick_params(axis="both", labelsize=9)

    if stat_legend:
        sh = _stat_line_handles(traps)
        if sh:
            ax.legend(
                handles=sh, loc=stat_legend_loc, fontsize=8,
                title="Statistic", title_fontsize=8, frameon=False,
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
        traps = _ordered_relabeled(traps)
        ax.text(xs[i], sep[i] / 3.0, lo_name, color="0.3", fontsize=8,
                style="italic", ha="left", va="top", zorder=3)


# ---------------------------------------------------------------------------
# Public: the four split figures (each "for all traps and statistics")
# ---------------------------------------------------------------------------
def plot_energies_per_particle(
    traps: Sequence[dict], *,
    figsize: Optional[tuple] = (12, 9),
    log_x: bool = True, log_y: bool = True,
    lw: float = 1.6, reference_lines: bool = True,
    stat_legend_loc: str = "best",
    labeling: str = "old", grid: bool = False,
    xscale: Optional[str] = None, yscale: Optional[str] = None,
):
    """Figure 1: |S/N|, |F/N|, |G/N|, |Omega/N| (per-particle, normalized).

    Each quantity is ``|(X/N)_i / (X/N)_0|`` built from the self-normalized
    lists. The four panels are laid out 2x2 as::

        S      F
        G      Omega

    Trap -> colour, statistic -> line style. `labeling` selects the legend
    layout (``'old'`` = per-panel Statistic box + bottom Trap legend;
    ``'new'`` = combined block); both are box-less. `grid` toggles the
    background grid. `xscale`/`yscale` accept any matplotlib scale and
    override `log_x`/`log_y`.
    """
    labeling = _resolve_labeling(labeling)
    xs = _resolve_scale(xscale, log_x)
    ys = _resolve_scale(yscale, log_y)
    traps = _ordered_relabeled(traps)
    panels = [
        ("S_over_N",     r"$|S/N|$"),
        ("F_over_N",     r"$|F/N|$"),
        ("G_over_N",     r"$|G/N|$"),
        ("Omega_over_N", r"$|\Omega/N|$"),
    ]
    fig, axd = plt.subplot_mosaic(
        [["S_over_N", "F_over_N"],
         ["G_over_N", "Omega_over_N"]],
        figsize=figsize,
    )
    for key, title in panels:
        ax = axd[key]
        _line_panel(ax, traps, key, title, abs_value=True,
                    xscale=xs, yscale=ys, reference_lines=reference_lines,
                    lw=lw, stat_legend=(labeling == "old"),
                    stat_legend_loc=stat_legend_loc, grid=grid)
        #if key == "Omega_over_N":
            #ax.set_xlim(1e-3, 1e-1)   # Omega panel — its own x-range
            #ax.set_ylim(1e-3, 1e-1)   # Omega panel — its own y-range
        #elif key == "S_over_N":
            #ax.set_xlim(5e-4, 1e-1)   # Omega panel — its own x-range
            #ax.set_ylim(1e-3, 1e-0)   # Omega panel — its own y-range
        #else:
            #ax.set_xlim(3e-4, 1e-1)   # S/N, F/N and G/N panels
            #ax.set_ylim(1e-6, 1e-1)
    _draw_bottom_legend(fig, traps, labeling=labeling, top=0.97)
    return fig


def plot_compressibility(
    traps: Sequence[dict], *,
    figsize: Optional[tuple] = (12, 5),
    log_x: bool = True, log_y: bool = True,
    lw: float = 1.6, reference_lines: bool = True,
    divide_be_fd: bool = True,
    stat_legend_loc: str = "best",
    labeling: str = "old", grid: bool = False,
    xscale: Optional[str] = None, yscale: Optional[str] = None,
):
    """Figure 2: kappa_T and B_P (normalized X_i/X_0), log-log.

    BE and FD fan out into separate trajectories at the cold end; with
    `divide_be_fd` a separatrix line splits the plane into the BE and FD
    regions. `labeling` selects the legend layout, `grid` toggles the
    background grid. `xscale`/`yscale` accept any matplotlib scale and
    override `log_x`/`log_y`.
    """
    labeling = _resolve_labeling(labeling)
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
                    lw=lw, stat_legend=(labeling == "old"),
                    stat_legend_loc=stat_legend_loc, grid=grid)
        traps = _ordered_relabeled(traps)
        if divide_be_fd:
            _add_be_fd_divider(ax, traps, key, abs_value=True)
            if key == "kappa_T":
                ax.set_xlim(3e-4, 1e-1)
                ax.set_ylim(1e2, 1e6)
            elif key == "B_P":
                ax.set_xlim(9e-4, 5e-2)
                ax.set_ylim(2e1, 1e3)
    
    #fig.suptitle(r"Isothermal compressibility & thermal expansion "
                 #r"(normalized $X_i / X_0$)", fontsize=12)
    _draw_bottom_legend(fig, traps, labeling=labeling, top=0.95)
    return fig


def plot_heat_capacities(
    traps: Sequence[dict], *,
    figsize: Optional[tuple] = (16, 9),
    log_x: bool = True, log_y: bool = True,
    lw: float = 1.6, reference_lines: bool = True,
    stat_legend_loc: str = "best",
    labeling: str = "old", grid: bool = False,
    xscale: Optional[str] = None, yscale: Optional[str] = None,
):
    """Figure 3: |C_V/N|, |C_P/N|, |(C_P-C_V)/N| (per-particle), log-log.

    C_V and C_P share the top row; (C_P-C_V) spans the row below them.
    `labeling` selects the legend layout, `grid` toggles the background grid.
    `xscale`/`yscale` accept any matplotlib scale and override `log_x`/`log_y`
    (e.g. pass ``yscale="linear"`` for a semilog-x view).
    """
    labeling = _resolve_labeling(labeling)
    xs = _resolve_scale(xscale, log_x)
    ys = _resolve_scale(yscale, log_y)
    panels = [
        ("CV_over_N",          r"$|C_V/N|$"),
        ("CP_over_N",          r"$|C_P/N|$"),
        #("CP_minus_CV_over_N", r"$|(C_P-C_V)/N|$"),
    ]
    traps = _ordered_relabeled(traps)
    fig, axd = plt.subplot_mosaic(
        [["CV_over_N", "CV_over_N", "CP_over_N", "CP_over_N"]],
         #[".", "CP_minus_CV_over_N", "CP_minus_CV_over_N", "."]],
        figsize=figsize,
    )
    for key, title in panels:
        ax = axd[key]
        _line_panel(ax, traps, key, title, abs_value=True,
                    xscale=xs, yscale=ys, reference_lines=reference_lines,
                    lw=lw, stat_legend=(labeling == "old"),
                    stat_legend_loc=stat_legend_loc, grid=grid)
        ax.set_xlim(5e-4, 1e-1)
    #fig.suptitle(r"Heat capacities per particle, normalized to initial:  "
                 #r"$|(X/N)_i / (X/N)_0|$", fontsize=12)
    _draw_bottom_legend(fig, traps, labeling=labeling, top=0.95)
    return fig


def plot_n_vs_t(
    traps: Sequence[dict], *,
    figsize: Optional[tuple] = (7.5, 6),
    log_x: bool = True, log_y: bool = True,
    lw: float = 1.6, reference_lines: bool = True,
    stat_legend_loc: str = "best",
    labeling: str = "old", grid: bool = False,
    xscale: Optional[str] = None, yscale: Optional[str] = None,
):
    """Figure 4: N_i/N_0 vs T_i/T_0, log-log.

    `labeling` selects the legend layout, `grid` toggles the background grid.
    `xscale`/`yscale` accept any matplotlib scale and override `log_x`/`log_y`.
    """
    labeling = _resolve_labeling(labeling)
    xs = _resolve_scale(xscale, log_x)
    ys = _resolve_scale(yscale, log_y)
    traps = _ordered_relabeled(traps)
    fig, ax = plt.subplots(figsize=figsize)
    _line_panel(ax, traps, "N", r"$N_i / N_0$", abs_value=False,
                xscale=xs, yscale=ys, reference_lines=reference_lines,
                lw=lw, stat_legend=(labeling == "old"),
                stat_legend_loc=stat_legend_loc, grid=grid)
    _draw_bottom_legend(fig, traps, labeling=labeling, top=0.97)
    return fig

# =============================================================================
# Region-shading primitives (used by the *_regions figures)
# =============================================================================
REGION_BE_MARKER = "."          # Bose-Einstein -> dots
REGION_FD_MARKER = "^"          # Fermi-Dirac   -> triangles
_REGION_DOT_COLOR = "0.45"
_REGION_TRI_COLOR = "0.45"


def _ffill_bfill(a: np.ndarray) -> np.ndarray:
    """Fill NaNs by carrying the nearest finite value (flat extension)."""
    a = np.asarray(a, dtype=float).copy()
    last = math.nan
    for i in range(a.size):                       # forward
        if math.isfinite(a[i]):
            last = a[i]
        elif math.isfinite(last):
            a[i] = last
    nxt = math.nan
    for i in range(a.size - 1, -1, -1):           # backward
        if math.isfinite(a[i]):
            nxt = a[i]
        elif math.isfinite(nxt):
            a[i] = nxt
    return a


def _boundary_on_grid(xs, boundary, gx):
    """Interpolate the (positive) boundary onto `gx` in log-log, flat outside."""
    b = np.asarray(boundary, float)
    m = np.isfinite(b) & (b > 0) & np.isfinite(xs) & (xs > 0)
    if m.sum() < 2:
        return None
    lx, ly = np.log(np.asarray(xs, float)[m]), np.log(b[m])
    order = np.argsort(lx)
    lx, ly = lx[order], ly[order]
    gx_safe = np.clip(np.asarray(gx, float), np.finfo(float).tiny, None)
    return np.exp(np.interp(np.log(gx_safe), lx, ly))   # clamped (flat) outside


def _shade_be_fd_regions(
    ax, xs, boundary, be_is_upper, *,
    xscale, yscale,
    nx=46, ny=28,
    dot_color=_REGION_DOT_COLOR, tri_color=_REGION_TRI_COLOR,
    dot_size=11.0, tri_size=14.0,         # BE dots enlarged
    dot_alpha=0.40, tri_alpha=0.22,
    zorder=0.6,
):
    """Stipple the BE half-plane with dots and the FD half-plane with triangles.

    A regular grid is laid over the current axes box (spaced to match each
    axis scale), split by `boundary`, and each half scattered with its marker.
    Axis limits are preserved.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    tiny = np.finfo(float).tiny

    if xscale == "log":
        gx = np.logspace(np.log10(max(x0, tiny)), np.log10(max(x1, tiny)), nx)
    else:
        gx = np.linspace(x0, x1, nx)
    if yscale == "log":
        gy = np.logspace(np.log10(max(y0, tiny)), np.log10(max(y1, tiny)), ny)
    else:
        gy = np.linspace(y0, y1, ny)

    bnd = _boundary_on_grid(xs, boundary, gx)
    if bnd is None:
        return

    XX, YY = np.meshgrid(gx, gy)
    BND = np.tile(bnd, (ny, 1))
    upper = YY > BND
    lower = ~upper
    be_mask, fd_mask = (upper, lower) if be_is_upper else (lower, upper)

    ax.scatter(XX[be_mask], YY[be_mask], marker=REGION_BE_MARKER, s=dot_size,
               c=dot_color, alpha=dot_alpha, linewidths=0.0, zorder=zorder)
    ax.scatter(XX[fd_mask], YY[fd_mask], marker=REGION_FD_MARKER, s=tri_size,
               c=tri_color, alpha=tri_alpha, linewidths=0.0, zorder=zorder)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def _region_corner_labels(
    ax, up_name, lo_name, *,
    color="0.20", fontsize=10, x_frac=0.035,
):
    """Place the two region labels in the top/bottom corners, off the curves."""
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
    ax.text(x_frac, 0.965, up_name, transform=ax.transAxes, color=color,
            fontsize=fontsize, style="italic", ha="left", va="top",
            bbox=box, zorder=4)
    ax.text(x_frac, 0.035, lo_name, transform=ax.transAxes, color=color,
            fontsize=fontsize, style="italic", ha="left", va="bottom",
            bbox=box, zorder=4)


# =============================================================================
# Combined Trap x Statistic legend (shared by all four split figures)
# =============================================================================
# A single legend block drawn OUTSIDE the subplots: one column per potential
# (trap name as the column header), one row per statistic (MB / BE / FD), and
# each cell a short line sample in that trap's true colour-shade and the
# statistic's true line style. Reading down a column gives one trap's three
# lines; reading across compares a statistic across traps. This replaces both
# the inner "Statistic" legend and the old bottom "Trap (potential)" legend.

# Fraction of the figure height reserved at the bottom for the combined legend.
_LEGEND_RECT_BOTTOM = 0.24

# Short row labels for the statistics.
STAT_SHORT = {"mb": "MB", "bosons": "BE", "fermions": "FD"}


def _combined_legend(
    fig, traps: Sequence[dict], *,
    y0: float = 0.01, height: float = 0.21,
    title: Optional[str] = "Potential / Statistic",
):
    """Draw the combined Trap x Statistic legend in a strip at the figure bottom.

    One block per potential (trap name header), and within each block the three
    statistics drawn in that trap's true colour-shade + line style, each tagged
    MB / BE / FD. Drawn box-less; the title is centred on the figure, just
    under the panels' x-axis. Call AFTER
    ``fig.tight_layout(rect=[0, _LEGEND_RECT_BOTTOM, 1, top])``.
    """
    import textwrap

    stats = _present_stats(traps)
    cols = [t for t in traps if _trap_display(t)]
    n_cols, n_rows = len(cols), len(stats)
    if n_cols == 0 or n_rows == 0:
        return None

    lax = fig.add_axes([0.0, y0, 1.0, height])
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axis("off")

    left, right = 0.06, 0.98
    span = right - left
    col_w = span / n_cols
    head_fs = 10 if n_cols <= 4 else 9
    label_gray = "0.40"          # MB/BE/FD tags: lighter but readable

    # Vertical layout within the strip.
    title_y = 0.94
    head_y = 0.68
    row_ys = (np.linspace(0.46, 0.12, n_rows) if n_rows > 1
              else [0.5 * (0.46 + 0.12)])

    # Title: normal weight, centred on the figure middle, alongside the x-axis.
    if title:
        lax.text(0.5, title_y, title, ha="center", va="center", fontsize=11)

    for i, t in enumerate(cols):
        cell_left = left + i * col_w
        cx = cell_left + col_w / 2.0
        base = _trap_color(_trap_key(t))

        header = "\n".join(textwrap.wrap(_trap_display(t), width=13)) \
            or _trap_display(t)
        lax.text(cx, head_y, header, ha="center", va="center",
                 fontsize=head_fs, linespacing=0.95)

        lbl_x = cell_left + col_w * 0.06
        s_x0 = cell_left + col_w * 0.34
        s_x1 = cell_left + col_w * 0.94
        for ry, stat in zip(row_ys, stats):
            lax.text(lbl_x, ry, STAT_SHORT.get(stat, stat),
                     ha="left", va="center", fontsize=8,
                     color=label_gray)
            lax.plot([s_x0, s_x1], [ry, ry],
                     color=_stat_color(base, stat),
                     linestyle=STAT_LINESTYLES.get(stat, "-"),
                     lw=2.0, solid_capstyle="round")
    return lax


# =============================================================================
# MB-curve frontier + BE / FD region shading
# =============================================================================
# The boundary between the BE and FD halves is the Maxwell-Boltzmann curve
# (the per-x median of all traps' MB curves -- they sit almost on top of one
# another, so this is a sharp line). BE is textured on the side where the BE
# bundle sits relative to MB at the cold end; FD on the other. The MB solid
# curves already drawn are the visible frontier; no extra divider is added.

def _mb_frontier(traps, key, *, abs_value=True, n_grid=240):
    """Return ``(xs, mb_curve, be_is_upper, up_name, lo_name)`` or None.

    ``mb_curve`` is the per-x median of the traps' MB curves, flat-extended
    across the x-range; ``be_is_upper`` says whether the BE bundle sits above
    MB at the cold end.
    """
    tvals = []
    for t in traps:
        res = t.get("mb")
        if res and res.get("T"):
            a = _to_float_array(res["T"])
            a = a[np.isfinite(a) & (a > 0)]
            if a.size:
                tvals.append(a)
    if not tvals:
        return None
    allt = np.concatenate(tvals)
    xs = np.logspace(np.log10(allt.min()), np.log10(allt.max()), n_grid)

    mb = _bundle(traps, key, "mb", xs, abs_value)
    if mb is None:
        return None
    mb_curve = np.nanmedian(mb, axis=0)
    if np.isfinite(mb_curve).sum() < 2:
        return None

    cold = slice(0, max(1, n_grid // 3))
    be = _bundle(traps, key, "bosons", xs, abs_value)
    be_above = True
    if be is not None:
        be_med = np.nanmedian(be, axis=0)
        be_above = (np.nanmedian(be_med[cold]) >= np.nanmedian(mb_curve[cold]))
    up_name, lo_name = (("Bose-Einstein", "Fermi-Dirac") if be_above
                        else ("Fermi-Dirac", "Bose-Einstein"))
    return xs, _ffill_bfill(mb_curve), be_above, up_name, lo_name


def plot_compressibility_regions(
    traps: Sequence[dict], *,
    figsize: Optional[tuple] = (12, 5),
    log_x: bool = True, log_y: bool = True,
    lw: float = 1.6, reference_lines: bool = True,
    shade: bool = True,
    stat_legend_loc: str = "best",
    labeling: str = "old", grid: bool = False,
    xscale: Optional[str] = None, yscale: Optional[str] = None,
    region_kwargs: Optional[dict] = None,
):
    """Compressibility figure with MB-bounded BE / FD shading on B_P only.

    Two panels: ``kappa_T`` (curves only -- the BE/FD bundles overlap around
    MB there, so no clean frontier) and ``B_P`` (BE stippled with dots above
    the MB curve, FD with translucent triangles below it). `labeling` selects
    the legend layout, `grid` toggles the background grid.
    """
    labeling = _resolve_labeling(labeling)
    xs_ = _resolve_scale(xscale, log_x)
    ys_ = _resolve_scale(yscale, log_y)
    panels = [
        ("kappa_T", r"$\kappa_{T,i} / \kappa_{T,0}$", False),
        ("B_P",     r"$B_{P,i} / B_{P,0}$",           True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (key, title, do_shade) in zip(axes, panels):
        _line_panel(ax, traps, key, title, abs_value=True,
                    xscale=xs_, yscale=ys_, reference_lines=reference_lines,
                    lw=lw, stat_legend=(labeling == "old"),
                    stat_legend_loc=stat_legend_loc, grid=grid)
        if shade and do_shade:
            fr = _mb_frontier(traps, key, abs_value=True)
            if fr is not None:
                xsg, mbc, be_up, up_name, lo_name = fr
                _shade_be_fd_regions(ax, xsg, mbc, be_up,
                                     xscale=xs_, yscale=ys_,
                                     **(region_kwargs or {}))
                _region_corner_labels(ax, up_name, lo_name)
    _draw_bottom_legend(fig, traps, labeling=labeling, top=0.97)
    return fig


def plot_heat_capacities_regions(
    traps: Sequence[dict], *,
    figsize: Optional[tuple] = (16, 5),
    log_x: bool = True, log_y: bool = True,
    lw: float = 1.6, reference_lines: bool = True,
    shade: bool = True,
    stat_legend_loc: str = "best",
    labeling: str = "old", grid: bool = False,
    xscale: Optional[str] = None, yscale: Optional[str] = None,
    region_kwargs: Optional[dict] = None,
):
    """Heat-capacity figure with MB-bounded BE / FD shading on all panels.

    Three per-particle panels (|C_V/N|, |C_P/N|, |(C_P-C_V)/N|): BE stippled
    with dots above the MB curve, FD with translucent triangles below it.
    `labeling` selects the legend layout, `grid` toggles the background grid.
    """
    labeling = _resolve_labeling(labeling)
    xs_ = _resolve_scale(xscale, log_x)
    ys_ = _resolve_scale(yscale, log_y)
    panels = [
        ("CV_over_N",          r"$|C_V/N|$"),
        ("CP_over_N",          r"$|C_P/N|$"),
        ("CP_minus_CV_over_N", r"$|(C_P-C_V)/N|$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (key, title) in zip(axes, panels):
        _line_panel(ax, traps, key, title, abs_value=True,
                    xscale=xs_, yscale=ys_, reference_lines=reference_lines,
                    lw=lw, stat_legend=(labeling == "old"),
                    stat_legend_loc=stat_legend_loc, grid=grid)
        if shade:
            fr = _mb_frontier(traps, key, abs_value=True)
            if fr is not None:
                xsg, mbc, be_up, up_name, lo_name = fr
                _shade_be_fd_regions(ax, xsg, mbc, be_up,
                                     xscale=xs_, yscale=ys_,
                                     **(region_kwargs or {}))
                _region_corner_labels(ax, up_name, lo_name)
    _draw_bottom_legend(fig, traps, labeling=labeling, top=0.97)
    return fig