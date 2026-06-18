# evap_cool

Semiclassical evaporative cooling of ideal quantum and classical gases, across
five trap geometries and three quantum statistics, using arbitrary-precision
arithmetic (`mpmath`) and Newton–Raphson rethermalization at each evaporation
step.

> Companion to *Arvizu-Velázquez et al., "Semi-classical evaporative cooling:
> classical and quantum distributions"* (arXiv, 2026).

## Features

- **Five traps** (by density-of-states exponent *s*): Box (*s*=3/2),
  2D-Box+1D-Oscillator (*s*=2), 2D-Oscillator+1D-Box (*s*=5/2),
  Harmonic Oscillator (*s*=3), Quadrupole (*s*=9/2).
- **Three statistics:** Maxwell–Boltzmann, Bose–Einstein, Fermi–Dirac.
- Equilibrium thermodynamics (Ω, S, P, H, F, G) and thermal coefficients,
  plus dimensionless (self-normalized) cross-trap figures.

## Installation

```bash
git clone https://github.com/arvizu-god/evaporative_cooling.git
cd evaporative_cooling
pip install -e .
```

## Quick start

```python
import numpy as np
from evap_cool import (
    BoxTrap, build_cutoff_schedule, create_result_dict,
    initialize_quantum_state, run_quantum_evaporation,
    make_session_dir, save_run, process_and_save_run,
)

trap = BoxTrap(V=6e-9)
Q_schedule = build_cutoff_schedule(Q0=5e-4, dQ=1e-8, n_steps=10000)

results = create_result_dict()
results["Q"] = Q_schedule
initialize_quantum_state(results, N0, T0, mu0, E0, Omega0)

outcome = run_quantum_evaporation(
    results, trap, N0, n_steps=10000,
    dT=T0*1e-12, dmu=abs(mu0)*1e-12, sign=+1,
)

session = make_session_dir()                       # runs/<date>/<time>/
save_run(results, session / "box_bosons.json", trap=trap, outcome=outcome)
process_and_save_run(session / "box_bosons.json", trap, sign=+1)
```

## Reproducing the paper figures

Two paths:

**1. From the committed canonical data (fast):**
```bash
python reproduce_figures.py        # writes figures/ from data/paper_run/
```

**2. Full recompute from scratch:**
```bash
python run_pipeline.py             # stage 1 (data) -> stage 2 (figures)
python run_pipeline.py --only box  # a single trap, end to end
```
Determinism comes from fixed parameters and the `mpmath` working precision — the
solver is deterministic (no random seeds).

## Repository layout

| Path | Purpose |
|---|---|
| `evap_cool/` | the package |
| `data/paper_run/` | canonical inputs for the paper figures |
| `figures/` | the committed paper figures |
| `run_pipeline.py` | full recompute orchestrator |
| `reproduce_figures.py` | regenerate figures from canonical data |

## Citing

See `CITATION.cff`, or cite the archived release DOI (Zenodo) and the paper.

[![DOI](https://zenodo.org/badge/399955091.svg)](https://doi.org/10.5281/zenodo.20741474)

## License

BSD-3-Clause — see `LICENSE`.