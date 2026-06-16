#!/usr/bin/env python3
"""run_pipeline.py — the whole evap_cool pipeline, end to end.

Chains the two stages and hands the freshly generated session straight from
stage 1 to stage 2 (no "guess the newest folder" in between):

    stage 1   evap_cool_data.py               runs + thermo JSON  (data only)
    stage 2   generate_dimensionless_plots.py normalize + dimensionless figures

Run from the repository root (the folder containing the ``evap_cool`` package,
``runs/`` and ``figures/``):

    python run_pipeline.py                      # full roster, prudent defaults
    python run_pipeline.py --only box           # one trap end to end
    python run_pipeline.py --only box,quadrupole --no-zoom

Per-stage configuration still lives in each stage's own CONFIG block; this
orchestrator only forwards --only / --no-zoom and pins the session.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the two stage modules importable regardless of the current directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evap_cool_data as data_stage              # noqa: E402
import generate_dimensionless_plots as plot_stage  # noqa: E402


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Run the full evap_cool pipeline (data -> dimensionless figures)."
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


def main() -> None:
    args = _parse_args()

    # Forward the run-scope knobs into stage 1's module-level config.
    if args.no_zoom:
        data_stage.ZOOM = False

    print("#" * 68)
    print("# evap_cool pipeline — stage 1/2: data generation")
    print("#" * 68)
    session = data_stage.main(only=args.only)      # raises SystemExit if nothing completed

    print("\n" + "#" * 68)
    print("# evap_cool pipeline — stage 2/2: dimensionless figures")
    print("#" * 68)
    plot_stage.main(session_override=session)       # pinned to the session above

    print("\n" + "=" * 68)
    print("Pipeline complete.")
    print(f"  Session : {session}")
    print(f"  Figures : {plot_stage.FIG_DIR.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()