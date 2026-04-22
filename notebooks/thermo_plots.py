"""
thermo_plots.py
===============

Plotting helpers for the outputs of ``thermodynamics.py``.

Intended workflow: after running an evaporative-cooling simulation and
post-processing with ``compute_thermodynamics_box*``, feed the augmented
results dicts here to visualize the thermodynamic potentials vs temperature.

Main entry point
----------------
``plot_thermodynamics_overview(results_b, results_f, results_mb, ...)``
    2×4 grid of panels: Ω, E, F, G, H, S, P, α = μ/(k_B T).
    Each panel overlays all three statistics (BE, FD, MB).

Design notes
------------
* Matches the color / label conventions of ``evap_cool_utils.py``:
  bosons = green, fermions = red, MB = blue, tab colors, s=8 scatter.
* Every quantity is plotted vs temperature on log-x.
* Energies (Ω, E, F, G, H) and other positive-definite quantities (S, P)
  use log-y.  Ω, F, G can be negative — they are plotted as |Ω|, |F|, |G|
  so log scale remains valid; the absolute-value bars appear in the labels.
* α = μ/(k_B T) uses linear-y (small dimensionless number, not log-spread).
* Only the first panel shows a legend, to reduce visual clutter.

Results dict requirements
-------------------------
Each non-None results dict must contain 'T' and at least some of the keys
{'Omega', 'E_thermo', 'F', 'G', 'H', 'S', 'P', 'Mu'}.  Missing quantities
are simply skipped without raising.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Style (matches evap_cool_utils.PLOT_COLORS / PLOT_LABELS exactly)
# ---------------------------------------------------------------------------
PLOT_COLORS = {
    'bosons':   'tab:green',
    'fermions': 'tab:red',
    'mb':       'tab:blue',
}

PLOT_LABELS = {
    'bosons':   'Bose-Einstein',
    'fermions': 'Fermi-Dirac',
    'mb':       'Maxwell-Boltzmann',
}


# ---------------------------------------------------------------------------
# Quantity metadata table
#
# Each entry: (results_key, display_label, unit, use_abs, yscale)
#   - use_abs = True:  plot |y| (quantity is always negative or can change sign)
#   - yscale:  'log' or 'linear'
# ---------------------------------------------------------------------------
_QUANTITIES = [
    ('Omega',    r'$|\Omega|$',              'J',    True,  'log'),
    ('E_thermo', r'$E$',                     'J',    False, 'log'),
    ('F',        r'$|F|$',                   'J',    True,  'log'),
    ('G',        r'$|G|$',                   'J',    True,  'log'),
    ('H',        r'$H$',                     'J',    False, 'log'),
    ('S',        r'$S$',                     'J/K',  False, 'log'),
    ('P',        r'$P$',                     'Pa',   False, 'log'),
    ('Mu',       r'$\alpha = \mu / (k_B T)$', '',    False, 'linear'),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _scatter_one(ax, T, y, color, label, use_abs, marker_size=8):
    """Scatter a single (T, y) series on axis ``ax``, handling |y| if needed."""
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)
    y_plot = np.abs(y) if use_abs else y
    ax.scatter(T, y_plot, c=color, s=marker_size, label=label, alpha=0.85,
               edgecolors='none')


def _sources(results_b, results_f, results_mb):
    """Return iterable of (results_dict, color, label) for non-None inputs."""
    out = []
    if results_b  is not None:
        out.append((results_b,  PLOT_COLORS['bosons'],   PLOT_LABELS['bosons']))
    if results_f  is not None:
        out.append((results_f,  PLOT_COLORS['fermions'], PLOT_LABELS['fermions']))
    if results_mb is not None:
        out.append((results_mb, PLOT_COLORS['mb'],       PLOT_LABELS['mb']))
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def plot_thermodynamics_overview(results_b=None, results_f=None, results_mb=None,
                                 trap_name='3D Box',
                                 figsize=(20, 9),
                                 marker_size=8,
                                 save_path=None):
    """
    Produce a 2×4 grid of thermodynamic quantities versus temperature.

    Panels (row-major):
        [ Ω,  E,  F,  G ]
        [ H,  S,  P,  α ]

    All three statistics (bosons, fermions, MB) are overlaid on every panel
    that has a corresponding key in the results dicts.  Pass ``None`` for
    any statistic you wish to omit.

    Parameters
    ----------
    results_b : dict or None
        Boson results, augmented by ``compute_thermodynamics_box(·, sign=+1)``.
    results_f : dict or None
        Fermion results, augmented by ``compute_thermodynamics_box(·, sign=-1)``.
    results_mb : dict or None
        MB results, augmented by ``compute_thermodynamics_box_mb``.
    trap_name : str, optional
        Used in the figure suptitle.  Default '3D Box'.
    figsize : tuple, optional
        Matplotlib figure size in inches.  Default (20, 9).
    marker_size : int, optional
        Scatter marker size.  Default 8 (matches evap_cool_utils convention).
    save_path : str or None, optional
        If given, save the figure to this path (extension determines format).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : 2D numpy array of matplotlib.axes.Axes, shape (2, 4)
    """
    if results_b is None and results_f is None and results_mb is None:
        raise ValueError("Must pass at least one of results_b, results_f, results_mb.")

    fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
    axes_flat = axes.flatten()
    sources = _sources(results_b, results_f, results_mb)

    for i, (key, label, unit, use_abs, yscale) in enumerate(_QUANTITIES):
        ax = axes_flat[i]

        any_data = False
        for results, color, stat_label in sources:
            if key not in results:
                continue
            _scatter_one(ax, results['T'], results[key],
                         color, stat_label, use_abs, marker_size)
            any_data = True

        ax.set_xscale('log')
        if any_data:
            ax.set_yscale(yscale)

        ax.set_xlabel(r'$T$ [K]', fontsize=11)
        ylab = f'{label} [{unit}]' if unit else label
        ax.set_ylabel(ylab, fontsize=12)
        ax.grid(True, which='both', ls='--', lw=0.4, alpha=0.5)
        ax.tick_params(labelsize=10)

        if not any_data:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', color='gray', fontsize=11)

    # Single legend on the first panel
    handles, lbls = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[0].legend(handles, lbls, fontsize=9, loc='best',
                            framealpha=0.9)

    fig.suptitle(
        f'Thermodynamic quantities vs. temperature — {trap_name}',
        fontsize=15, fontweight='bold',
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


# ---------------------------------------------------------------------------
# Per-quantity helper (for custom layouts)
# ---------------------------------------------------------------------------
def plot_single_quantity(ax, results_b=None, results_f=None, results_mb=None,
                         key='Omega', use_abs=None, marker_size=8):
    """
    Draw one thermodynamic quantity on a user-provided axis.

    Useful for building custom figure layouts (e.g. side-by-side comparison
    with another diagnostic, or embedding in a larger report figure).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    results_b, results_f, results_mb : dict or None
        Results dicts; any may be None.
    key : str
        Which quantity to plot (e.g. 'Omega', 'S', 'E_thermo', 'F', 'G',
        'H', 'P', 'Mu').
    use_abs : bool or None
        If True, plot ``|y|``.  If None, auto-detect from the internal
        metadata table (same behaviour as the overview function).
    marker_size : int
        Scatter marker size.

    Returns
    -------
    ax : the same axis (for chaining)
    """
    meta = {k: (lbl, u, abs_, ys)
            for (k, lbl, u, abs_, ys) in _QUANTITIES}
    if key not in meta:
        raise KeyError(f"Unknown quantity '{key}'.  "
                       f"Valid keys: {list(meta)}")
    label, unit, default_abs, yscale = meta[key]
    if use_abs is None:
        use_abs = default_abs

    for results, color, stat_label in _sources(results_b, results_f, results_mb):
        if key not in results:
            continue
        _scatter_one(ax, results['T'], results[key],
                     color, stat_label, use_abs, marker_size)

    ax.set_xscale('log')
    ax.set_yscale(yscale)
    ax.set_xlabel(r'$T$ [K]', fontsize=11)
    ax.set_ylabel(f'{label} [{unit}]' if unit else label, fontsize=12)
    ax.grid(True, which='both', ls='--', lw=0.4, alpha=0.5)

    return ax