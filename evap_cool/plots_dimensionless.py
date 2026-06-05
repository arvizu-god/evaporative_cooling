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

# =============================================================================
# Required imports — already present in plots.py except numpy.
# =============================================================================
from __future__ import annotations

import math
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np   # NEW: add to existing import block at top of plots.py


# =============================================================================
# Style — add alongside the existing PLOT_COLORS / PLOT_LABELS block
# =============================================================================
# Color is reserved for trap identity in the dimensionless overview.
# Statistic identity is carried by line style (see DIM_STAT_STYLES).
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