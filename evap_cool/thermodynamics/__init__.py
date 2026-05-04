"""Thermodynamics subpackage: trap classes and the Maxwell–Boltzmann kernel.

Public API
----------
BoxTrap
    3D box trap (s = 3/2). SI units by default.
QuadrupoleTrap
    3D quadrupole trap with linear potential (s = 9/2). eV units by default.
OscillatorTrap
    3D isotropic harmonic oscillator (s = 3). eV units by default.
Trap
    Abstract base class. Exposed for type-hinting and for users defining
    custom traps (e.g. mixed geometries from sections 3.4 and 3.5 of the
    paper).
mb_particle_number
    Trap-independent particle-number recurrence in the classical limit.
mb_temperature
    Universal MB temperature kernel parameterized by the density-of-states
    exponent `s`. Trap classes use this internally; exposed for callers
    that want to apply the classical limit without instantiating a trap.

Typical usage
-------------
    from evap_cool.thermodynamics import BoxTrap, mb_particle_number

    trap = BoxTrap(V=6e-9)
    N1, E1, Omega1 = trap.truncated_NEO(N0, T0, mu0, E0, Omega0, Q, sign=+1)
    T1 = trap.mb_temperature(Q, T0)
"""

from .base import Trap
from .box import BoxTrap
from .quadrupole import QuadrupoleTrap
from .oscillator import OscillatorTrap
from .maxwell_boltzmann import mb_particle_number, mb_temperature

__all__ = [
    "Trap",
    "BoxTrap",
    "QuadrupoleTrap",
    "OscillatorTrap",
    "mb_particle_number",
    "mb_temperature",
]