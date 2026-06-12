#!/usr/bin/env python3
"""Generate every dimensionless (self-normalized) figure from a run.

Terminal equivalent of ``dimensionless_plots.ipynb``. Run it from the repo
root (the folder that contains the ``evap_cool`` package and ``runs/``):

    python make_dimensionless_plots.py

It takes a session under ``runs/`` (the most recent by default), (re)writes
the ``*_norm.json`` files, builds the normalized trap structure, and saves
all figures into ``figures/``:

    fig1_energies_per_particle.png
    fig2_compressibility.png
    fig3_heat_capacities.png
    fig4_N_vs_T.png
    dimensionless_overview.png
    cp_minus_cv.png
    fig2_new_compressibility.png    <- BE/FD region-shaded variant (new)
    fig3_new_heat_capacities.png    <- BE/FD region-shaded variant (new)

Adjust the CONFIG block below to change scales, the session, or the output
folder. No arguments are parsed; edit the constants and rerun.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")          # headless/batch-safe: save to disk, no popups
import matplotlib.pyplot as plt

# Import the package, falling back to the script's own folder (mirrors the
# notebook's behaviour when evap_cool isn't installed on the path).
try:
    import evap_cool  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import evap_cool  # noqa: F401

from evap_cool import (
    list_sessions, list_runs,
    normalize_session, load_normalized, build_normalized_traps,
    plot_dimensionless_overview, plot_cp_minus_cv,
    plot_energies_per_particle, plot_compressibility,
    plot_heat_capacities, plot_n_vs_t,
    plot_compressibility_regions, plot_heat_capacities_regions,
)

# =============================================================================
# CONFIG  -- edit these, then rerun
# =============================================================================
RUNS_DIR = Path("runs")
FIG_DIR  = Path("figures")

# Which session to plot.
#   None  -> auto: the most recent session that actually contains source runs
#            (empty / aborted session folders are skipped). Recommended.
#   int   -> an explicit index into list_sessions() (sorted oldest-first), e.g.
#            -1 for the newest folder regardless of contents, -7 to step back.
SESSION_INDEX = None

# Axis scaling for every figure (any matplotlib scale name: "linear", "log",
# "symlog", "asinh", "logit", ...).
XSCALE = "log"
YSCALE = "log"

# Overview / Cp-Cv knobs (these were referenced but never set in the notebook).
YLIM_PCT  = 99.0
TRIM_TAIL = 0
STRIDE    = 1

# Per-figure DPI (matches the notebook's settings).
DPI_ENERGIES = 500
DPI_DEFAULT  = 500


# =============================================================================
def _has_source_runs(session: Path) -> bool:
    """True if `session` contains at least one source run JSON.

    Source runs exclude the post-processed ``*_thermo.json`` and the
    normalized ``*_norm.json`` siblings.
    """
    return any(
        not p.stem.endswith("_norm")
        for p in list_runs(session, include_thermo=False)
    )


def _select_session(sessions: list[Path]) -> Path:
    """Resolve the session to plot from the CONFIG."""
    if SESSION_INDEX is not None:
        return sessions[SESSION_INDEX]
    # auto: newest-first, first session that actually has source runs
    for s in reversed(sessions):
        if _has_source_runs(s):
            return s
    raise SystemExit(
        f"None of the {len(sessions)} session(s) under {RUNS_DIR.resolve()} "
        "contain source run files (all empty?). Set SESSION_INDEX explicitly."
    )


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    # --- 1. Locate the session -------------------------------------------
    sessions = list_sessions(RUNS_DIR)
    if not sessions:
        raise SystemExit(f"No sessions under {RUNS_DIR.resolve()}; set RUNS_DIR.")
    session = _select_session(sessions)
    print(f"Selected session: {session}")

    source_runs = [p for p in list_runs(session, include_thermo=False)
                   if not p.stem.endswith("_norm")]
    print(f"\nSource run files ({len(source_runs)}):")
    for p in source_runs:
        thermo = p.with_name(p.stem + "_thermo.json")
        tag = "  + thermo" if thermo.exists() else "  (no thermo)"
        print(f"  {p.name}{tag}")

    # --- 2. Generate the normalized data ---------------------------------
    norm_files = normalize_session(session, verbose=True)
    print(f"\nWrote {len(norm_files)} normalized files")

    # --- 3. Sanity check (first norm file should start at 1.0) ------------
    if norm_files:
        res = load_normalized(sorted(norm_files)[0])["results"]
        print("available series:",
              ", ".join(k for k in res if res[k] is not None))
        print("first value (should be 1.0 where defined):")
        for k in ("T", "N", "Omega", "kappa_T", "B_P",
                  "Omega_over_N", "CV_over_N", "CP_minus_CV_over_N"):
            v = res.get(k)
            if v:
                print(f"  {k:>20}: {v[0]}")

    # --- 4. Build the trap structure -------------------------------------
    traps = build_normalized_traps(session)
    if not traps:
        raise SystemExit("No normalized traps found; nothing to plot.")
    print("\nTraps:")
    for t in traps:
        stats = [s for s in ("mb", "bosons", "fermions") if s in t]
        print(f"  {t['name']:<14} (key={t['key']}): {', '.join(stats)}")

    # --- 5. Figures ------------------------------------------------------
    def save(fig, name, dpi=DPI_DEFAULT):
        path = FIG_DIR / name
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")

    print("\nRendering figures:")

    save(plot_energies_per_particle(traps, xscale=XSCALE, yscale=YSCALE),
         "fig1_energies_per_particle.png", dpi=DPI_ENERGIES)

    save(plot_compressibility(traps, xscale=XSCALE, yscale=YSCALE,
                              divide_be_fd=True),
         "fig2_compressibility.png")

    save(plot_heat_capacities(traps, xscale=XSCALE, yscale=YSCALE),
         "fig3_heat_capacities.png")

    save(plot_n_vs_t(traps, xscale=XSCALE, yscale="linear"),
         "fig4_N_vs_T.png")

    save(plot_dimensionless_overview(traps, xscale=XSCALE, yscale=YSCALE,
                                     robust_ylim=True, ylim_pct=YLIM_PCT,
                                     trim_tail=TRIM_TAIL, stride=STRIDE),
         "dimensionless_overview.png")

    save(plot_cp_minus_cv(traps, xscale=XSCALE, yscale=YSCALE,
                          ylim_pct=YLIM_PCT, trim_tail=TRIM_TAIL, stride=STRIDE),
         "cp_minus_cv.png")

    # --- New BE/FD region-shaded variants --------------------------------
    save(plot_compressibility_regions(traps, xscale=XSCALE, yscale=YSCALE),
         "fig2_new_compressibility.png")

    save(plot_heat_capacities_regions(traps, xscale=XSCALE, yscale=YSCALE),
         "fig3_new_heat_capacities.png")

    print(f"\nDone. Figures in {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()