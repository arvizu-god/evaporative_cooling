#!/usr/bin/env python3
"""Stage 1 of the evap_cool pipeline: generate the run + thermo data.

Terminal equivalent of the data-producing half of ``evap_cool_usage.ipynb``.
For every trap in the roster it:

  1. solves the initial (mu0, E0, Omega0) consistent with (N0, T0) for BE / FD,
  2. builds the cut-off schedule and runs BE, FD and MB evaporation (with the
     tested two-stage BEC / Fermi zoom by default),
  3. saves the three runs as ``<key>_bosons.json`` / ``<key>_fermions.json`` /
     ``<key>_mb.json`` into a fresh timestamped session under ``runs/``,
  4. post-processes each run into its ``*_thermo.json`` sibling.

It produces DATA ONLY -- no figures. The final figures (the dimensionless,
self-normalized overlay) are stage 2: ``generate_dimensionless_plots.py``.
``run_pipeline.py`` chains the two and hands this session straight to stage 2.

Usage
-----
    python evap_cool_data.py                 # all traps, prudent defaults
    python evap_cool_data.py --only box      # one trap (or: --only box,quadrupole)
    python evap_cool_data.py --no-zoom        # single coarse pass, faster

Edit the CONFIG block to change the physical initial state, the cut-off
schedule, the zoom depth, or the precision. Defaults are deliberately light so
a full five-trap run finishes quickly; raise ``N_STEPS_COARSE`` (toward ~9999)
and ``DPS_FINE`` (toward ~80) for publication-depth trajectories.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

# Import the package, falling back to the script's own folder (mirrors the
# notebooks' behaviour when evap_cool isn't installed on the path).
try:
    import evap_cool  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import evap_cool  # noqa: F401

from evap_cool import (
    BoxTrap, QuadrupoleTrap, OscillatorTrap, OscBoxTrap, BoxOscTrap,
    create_result_dict, create_mb_result_dict, build_cutoff_schedule,
    initialize_quantum_state, initialize_mb_state,
    run_quantum_evaporation, run_mb_evaporation,
    run_with_bec_zoom, run_with_fermi_zoom,
    make_session_dir, save_run, list_runs,
    process_and_save_run, process_and_save_mb_run,
)

# =============================================================================
# CONFIG  -- edit these, then rerun
# =============================================================================
RUNS_DIR = Path("runs")

# Physical initial state and evaporation schedule.
#
# Defaults are PRUDENT (fast). The two knobs that trade speed for trajectory
# depth are N_STEPS_COARSE (how far the coarse evaporation sweeps) and the
# precision pair DPS_COARSE / DPS_FINE. For the publication runs use roughly
#   N_STEPS_COARSE = 9999, N_STEPS_FINE = 200, DPS_COARSE = 30, DPS_FINE = 80.
#
# NOTE: the coarse schedule bottoms out at Q = 0 after Q0/DQ_COARSE steps
# (= 10000 with the values below). Keep N_STEPS_COARSE <= Q0/DQ_COARSE - 1
# (i.e. <= 9999) to use the whole schedule without a wasted final step; the
# MB run halts gracefully at Q = 0 regardless.
N0 = 1.0e7          # initial particle number
T0 = 5.0e-5         # initial temperature

Q0             = 5e-4      # initial cut-off temperature
DQ_COARSE      = 5e-8      # coarse cut-off decrement
N_STEPS_COARSE = 10000      # coarse steps  (publication: ~9999)
DQ_FINE        = 5e-20     # fine-pass cut-off decrement (zoom)
N_STEPS_FINE   = 200       # fine steps    (publication: ~200)

# Maxwell-Boltzmann reference length -- INDEPENDENT of the quantum runs.
#   N_STEPS_MB : number of MB steps   (None -> match N_STEPS_COARSE).
#   DQ_MB      : MB cut-off decrement (None -> match DQ_COARSE).
# Two regimes when N_STEPS_MB < N_STEPS_COARSE:
#   * DQ_MB = DQ_COARSE (or None): MB stops earlier -> warmer, SHORTER T-reach
#     than the quantum curves (fewer points, same temperatures).
#   * DQ_MB larger: MB spans the SAME Q range with coarser sampling. Keep
#     N_STEPS_MB * DQ_MB ~ N_STEPS_COARSE * DQ_COARSE to match the quantum
#     coarse T-reach with fewer points.
N_STEPS_MB = 9950          # e.g. 2000  (None -> N_STEPS_COARSE)
DQ_MB      = None          # e.g. 2.5e-7 (None -> DQ_COARSE)

ZOOM = True            # two-stage BEC/Fermi zoom; False = single coarse pass

ALPHA_FLOOR_COARSE = -1.05
ALPHA_FLOOR_FINE   = -1.02

DPS_COARSE = 20            # mpmath precision, coarse pass (publication: ~30)
DPS_FINE   = 40            # mpmath precision, fine pass   (publication: ~80)
BACKOFF    = 5            # steps to re-seed before the fine pass

DT_NR  = 1e-20            # Newton-Raphson step in T
DMU_NR = 1e-30           # Newton-Raphson step in mu

# Which traps to run by default (keys). None -> the full roster below.
# The --only CLI flag overrides this.
ONLY = None


# -----------------------------------------------------------------------------
# Trap roster -- the single source of truth for stage 1.
# Keys are the filename stems shared with stage 2 (generate_dimensionless_plots).
# -----------------------------------------------------------------------------
def build_roster() -> list[dict]:
    """Instantiate the five-trap roster. Edit parameters here, in one place."""
    return [
        dict(key="box",         name="Box",
             trap=BoxTrap(V=1e-11)),
        dict(key="box2d_osc1d", name="2D box + 1D HO",
             trap=BoxOscTrap(omega_z=2 * np.pi * 100, Sigma=1e-8)),
        dict(key="osc2d_box1d", name="2D HO + 1D box",
             trap=OscBoxTrap(omega_x=2 * np.pi * 100, omega_y=2 * np.pi * 100, L=1e-4)),
        dict(key="oscillator",  name="Oscillator",
             trap=OscillatorTrap(omega=2 * np.pi * 100)),
        dict(key="quadrupole",  name="Quadrupole",
             trap=QuadrupoleTrap(A_bar=1e-15)),
    ]


# =============================================================================
# Core run logic (ported verbatim from evap_cool_usage.ipynb, minus plotting)
# =============================================================================
def solve_initial_quantum_state(trap, N0, T0, sign, **kw):
    """Return (mu0, E0, Omega0) consistent with (N0, T0) for the given statistics."""
    alpha0 = trap.solve_alpha0(N0, T0, sign=sign, **kw)
    mu0    = alpha0 * trap.kB * T0
    E0     = float(trap.equilibrium_E(T0, mu0, sign).real)
    Omega0 = float(trap.equilibrium_Omega(T0, mu0, sign).real)
    return mu0, E0, Omega0


def run_and_save_trap(trap, name, stem, session) -> dict:
    """Run BE / FD / MB on `trap` and save the three runs into `session`.

    Returns a small summary dict {bosons, fermions, mb} of committed steps.
    Raises on failure (the caller decides whether to continue).
    """
    print(f"\n=== {name}  ({stem}) ===")

    # 1. Initial states for the two quantum statistics.
    mu0_b, E0_b, Omega0_b = solve_initial_quantum_state(trap, N0, T0, sign=+1)
    mu0_f, E0_f, Omega0_f = solve_initial_quantum_state(trap, N0, T0, sign=-1, alpha_hi=20.0)
    print(f"  alpha0 (bosons)   = {mu0_b/(trap.kB*T0):+.4f}")
    print(f"  alpha0 (fermions) = {mu0_f/(trap.kB*T0):+.4f}")

    # 2. Coarse Q-schedule. Used by the no-zoom BE/FD passes and by the MB run.
    #    (In zoom mode the BE/FD passes build their own internal schedules.)
    Q_schedule = build_cutoff_schedule(Q0=Q0, dQ=DQ_COARSE, n_steps=N_STEPS_COARSE)

    # 3a. Bosons.
    if ZOOM:
        res_b, zoom_b = run_with_bec_zoom(
            trap, N0=N0, T0=T0, mu0=mu0_b, E0=E0_b, Omega0=Omega0_b,
            Q0=Q0, dQ_coarse=DQ_COARSE, n_steps_coarse=N_STEPS_COARSE,
            dQ_fine=DQ_FINE, n_steps_fine=N_STEPS_FINE,
            dT=DT_NR, dmu=DMU_NR, sign=+1,
            alpha_floor_coarse=ALPHA_FLOOR_COARSE, alpha_floor_fine=ALPHA_FLOOR_FINE,
            dps_coarse=DPS_COARSE, dps_fine=DPS_FINE, backoff=BACKOFF, verbose=False,
        )
        out_b = zoom_b
    else:
        res_b = create_result_dict(); res_b["Q"] = list(Q_schedule)
        initialize_quantum_state(res_b, N0, T0, mu0_b, E0_b, Omega0_b)
        out_b = run_quantum_evaporation(
            res_b, trap, N0, n_steps=N_STEPS_COARSE, dT=DT_NR, dmu=DMU_NR, sign=+1,
            alpha_floor=ALPHA_FLOOR_COARSE, verbose=False,
        )
        zoom_b = None

    # 3b. Fermions.
    if ZOOM:
        res_f, zoom_f = run_with_fermi_zoom(
            trap, N0=N0, T0=T0, mu0=mu0_f, E0=E0_f, Omega0=Omega0_f,
            Q0=Q0, dQ_coarse=DQ_COARSE, n_steps_coarse=N_STEPS_COARSE,
            dQ_fine=DQ_FINE, n_steps_fine=N_STEPS_FINE,
            dT=DT_NR, dmu=DMU_NR, sign=-1,
            dps_coarse=DPS_COARSE, dps_fine=DPS_FINE, backoff=BACKOFF, verbose=False,
        )
        out_f = zoom_f
    else:
        res_f = create_result_dict(); res_f["Q"] = list(Q_schedule)
        initialize_quantum_state(res_f, N0, T0, mu0_f, E0_f, Omega0_f)
        out_f = run_quantum_evaporation(
            res_f, trap, N0, n_steps=N_STEPS_COARSE, dT=DT_NR, dmu=DMU_NR, sign=-1,
            verbose=False,
        )
        zoom_f = None

    # 3c. Maxwell-Boltzmann reference. Independent of the quantum runs: its
    #     length is set by N_STEPS_MB / DQ_MB (each falling back to the coarse
    #     value), so MB can run shorter than the quantum case. run_mb_evaporation
    #     halts gracefully if Q reaches 0.
    n_steps_mb = N_STEPS_MB if N_STEPS_MB is not None else N_STEPS_COARSE
    dq_mb      = DQ_MB if DQ_MB is not None else DQ_COARSE
    Q_schedule_mb = build_cutoff_schedule(Q0=Q0, dQ=dq_mb, n_steps=n_steps_mb)
    res_mb = create_mb_result_dict(); res_mb["Q"] = list(Q_schedule_mb)
    initialize_mb_state(res_mb, N0, T0)
    out_mb = run_mb_evaporation(res_mb, trap, N0, n_steps=n_steps_mb, verbose=False)

    n_b  = zoom_b.n_total_committed if zoom_b is not None else out_b.n_completed
    n_f  = zoom_f.n_total_committed if zoom_f is not None else out_f.n_completed
    n_mb = out_mb.n_completed
    print(f"  committed : bosons={n_b}  fermions={n_f}  mb={n_mb}")

    # 4. Save all three runs into the session, stems == key.
    common = dict(
        N0=N0, T0=T0, Q0=Q0, dQ=DQ_COARSE, n_steps=N_STEPS_COARSE,
        dQ_fine=DQ_FINE, n_steps_fine=N_STEPS_FINE,
        alpha_floor_coarse=ALPHA_FLOOR_COARSE, alpha_floor_fine=ALPHA_FLOOR_FINE,
        dps_fine=DPS_FINE, backoff=BACKOFF, zoom=ZOOM,
    )
    boson_extra   = {"zoom": zoom_b.to_metadata()} if zoom_b is not None else None
    fermion_extra = {"zoom": zoom_f.to_metadata()} if zoom_f is not None else None
    save_run(res_b, session / f"{stem}_bosons.json", trap=trap,
             parameters={**common, "mu0": mu0_b, "E0": E0_b, "Omega0": Omega0_b, "sign": +1},
             outcome=out_b, extra_metadata=boson_extra)
    save_run(res_f, session / f"{stem}_fermions.json", trap=trap,
             parameters={**common, "mu0": mu0_f, "E0": E0_f, "Omega0": Omega0_f, "sign": -1},
             outcome=out_f, extra_metadata=fermion_extra)
    save_run(res_mb, session / f"{stem}_mb.json", trap=trap,
             parameters={**common, "sign": 0, "n_steps_mb": n_steps_mb, "dQ_mb": dq_mb},
             outcome=out_mb)

    return {"bosons": n_b, "fermions": n_f, "mb": n_mb}


def post_process_trap(trap, stem, session) -> None:
    """Write the three *_thermo.json siblings for one trap's saved runs."""
    process_and_save_run(session / f"{stem}_bosons.json",   trap, sign=+1)
    process_and_save_run(session / f"{stem}_fermions.json", trap, sign=-1)
    process_and_save_mb_run(session / f"{stem}_mb.json",    trap)


# =============================================================================
def _select_roster(only) -> list[dict]:
    """Filter the roster by a set/list of keys (None -> everything)."""
    roster = build_roster()
    if not only:
        return roster
    wanted = {k.strip() for k in only} if not isinstance(only, str) else \
             {k.strip() for k in only.split(",")}
    chosen = [t for t in roster if t["key"] in wanted]
    unknown = wanted - {t["key"] for t in roster}
    if unknown:
        print(f"  warning: unknown trap key(s) ignored: {sorted(unknown)}")
    if not chosen:
        raise SystemExit(f"No matching traps for --only {sorted(wanted)}. "
                         f"Available: {[t['key'] for t in roster]}")
    return chosen


def main(only=None) -> Path:
    """Generate runs + thermo for the selected traps. Returns the session path."""
    roster = _select_roster(only if only is not None else ONLY)

    session = make_session_dir(base=str(RUNS_DIR))
    print(f"Session folder : {session}")
    print(f"Traps to run   : {', '.join(t['key'] for t in roster)}")
    print(f"Zoom           : {ZOOM}   (coarse steps={N_STEPS_COARSE}, "
          f"dps {DPS_COARSE}/{DPS_FINE})")

    # Continue past a failing trap; collect outcomes for the end-of-run summary.
    summary: list[tuple[str, str, str]] = []   # (key, status, detail)
    for t in roster:
        key, name, trap = t["key"], t["name"], t["trap"]
        try:
            counts = run_and_save_trap(trap, name, key, session)
        except Exception as exc:                       # run/save stage failed
            print(f"  [FAIL] {key}: run/save raised {type(exc).__name__}: {exc}")
            traceback.print_exc()
            summary.append((key, "run-failed", f"{type(exc).__name__}: {exc}"))
            continue
        try:
            post_process_trap(trap, key, session)
        except Exception as exc:                       # thermo stage failed
            print(f"  [WARN] {key}: runs saved but thermo failed "
                  f"({type(exc).__name__}: {exc})")
            summary.append((key, "thermo-failed", f"{type(exc).__name__}: {exc}"))
            continue
        summary.append((key, "ok",
                        f"bosons={counts['bosons']} fermions={counts['fermions']} "
                        f"mb={counts['mb']}"))
        print(f"  thermo    -> {key}_*_thermo.json written")

    # ---- End-of-run summary -------------------------------------------------
    print("\n" + "=" * 68)
    print(f"SUMMARY  ({session})")
    print("=" * 68)
    ok = sum(1 for _, st, _ in summary if st == "ok")
    for key, status, detail in summary:
        flag = {"ok": "  ok ", "run-failed": "FAIL", "thermo-failed": "WARN"}[status]
        print(f"  [{flag}] {key:<14} {detail}")
    print(f"\n{ok}/{len(summary)} trap(s) fully completed.")

    # Show what landed on disk.
    thermo = [p.name for p in list_runs(session) if p.stem.endswith("_thermo")]
    print(f"{len(thermo)} thermo file(s) in session.")

    if ok == 0:
        raise SystemExit("No trap completed end-to-end; nothing for stage 2 to plot.")
    return session


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Stage 1: generate evap_cool run + thermo JSON data (no figures)."
    )
    p.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated trap keys to run (e.g. 'box,quadrupole'). "
             "Default: the full roster.",
    )
    p.add_argument(
        "--no-zoom", action="store_true",
        help="Single coarse pass instead of the two-stage BEC/Fermi zoom (faster).",
    )
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    if _args.no_zoom:
        ZOOM = False
    main(only=_args.only)