#!/usr/bin/env python3
"""Reproduce the dimensionless figures from the committed canonical dataset.

This is the friction-free reproduction path: it points the existing stage-2
plotting pipeline at the committed canonical session (base + thermo JSON under
``data/paper_run/``), regenerates the ``*_norm.json`` files, and writes the four
paper figures into ``figures/``.

    python reproduce_figures.py

After running, ``git status`` should show no changes to the committed PNGs —
that is your proof the published figures were reproduced bit-for-bit.

For the full recompute-from-scratch path (regenerating the raw simulation
output as well), use ``run_pipeline.py`` instead.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Make the stage-2 module importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_dimensionless_plots as plot_stage  # noqa: E402

CANON = Path(__file__).resolve().parent / "data" / "paper_run"


def main() -> None:
    if not CANON.exists():
        raise SystemExit(f"Canonical dataset not found: {CANON}")
    # generate_dimensionless_plots.main accepts a session_override (this is the
    # same hook run_pipeline.py uses to pin stage 2 to a specific session).
    plot_stage.main(session_override=CANON)
    print(f"Figures regenerated into: {plot_stage.FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()