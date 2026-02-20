# Evaporative cooling of quantum trapped gases on an external potential

This repository now includes a modular Python implementation of the evaporative
cooling models that were originally notebook-based.

## New modular implementation

The file `evap_cooling.py` provides:

- OOP model classes for the three trap versions used in the project:
  - `HarmonicTrapModel` (oscillator)
  - `BoxTrapModel` (box)
  - `QuadrupoleTrapModel` (quadrupole)
- A shared physics base class (`QuantumEvaporationModel`) that centralizes
  number/energy equations for bosons and fermions.
- A refined Newton-Raphson solver for:
  - 1D roots (`NewtonRaphson1D`) with bracket-aware damping.
  - 2D systems (`NewtonRaphsonSystem`) with Jacobian solve + residual-based
    backtracking line search.

## Quick usage

```python
import numpy as np
from evap_cooling import HarmonicTrapModel

model = HarmonicTrapModel(omega=2*np.pi*100)

N0 = 1e7
T0 = 5e-5

# Solve dimensionless x = mu/(k_B T)
x = model.solve_chemical_potential(
    n_atoms=N0,
    temperature=T0,
    boson=False,
    guess=-11.8,
    bracket=(-13, -10),
)
```

The model API can also solve coupled `(T, mu)` states from `(N, E)` targets via
`solve_state(...)`.

## Example notebook

- `example_oscillator_plots.ipynb`: end-to-end example that reproduces the oscillator plots from the original notebook using the modular OOP API.
