"""Maxwell–Boltzmann (classical) limit of the evaporation recurrences.

Two functions live here:
  - `mb_particle_number`: trap-independent, used directly by the MB
    evaporation loop.
  - `mb_temperature`: shared kernel parameterized by the density-of-states
    exponent `s`. Trap classes invoke this from their own `mb_temperature`
    method, supplying `s = 1.5` (box), `4.5` (quadrupole), `3` (oscillator).

This file contains no trap-specific logic. The classical limit is a
universal calculation parameterized by a single number (s); per-trap
boilerplate would re-introduce the duplication the package was
refactored to eliminate.
"""

import mpmath as mp
import scipy.special as ss


def mb_particle_number(N0, Q, T):
    """Remaining particle number after evaporation cut at Q (classical MB).

    Trap-independent: depends only on the dimensionless ratio  eta = Q/T.

        N_1 / N_0 = erf(sqrt(eta)) - (2/sqrt(pi)) * sqrt(eta) * exp(-eta)

    Parameters
    ----------
    N0 : float
        Particle number before the cut.
    Q : float
        Cut-off temperature [same units as T].
    T : float
        Sample temperature.

    Returns
    -------
    float
        Particle number after the cut.
    """
    if not (Q > 0):
        raise ValueError(
            f"mb_temperature requires Q > 0, got Q = {Q!r}. "
            f"The classical MB cut is only defined for a positive cut-off "
            f"temperature; consider clipping the cut-off schedule before this point."
        )
    if not (T > 0):
        raise ValueError(f"mb_temperature requires T > 0, got T = {T!r}.")

    eta = Q / T
    sqrt_eta = float(mp.sqrt(eta))
    return float(N0 * (ss.erf(sqrt_eta) - (2 / mp.sqrt(mp.pi)) * sqrt_eta * mp.exp(-eta)))


def mb_temperature(s, Q, T):
    """Post-cut MB temperature for any pure-geometry trap with exponent `s`.

    Density of states  g(eps) ~ eps^(s-1),  so  d/2 = s.  The energy ratio
    in the MB cut acquires a coefficient  c2 = 2/(s sqrt(pi)) * eta^(3/2)
    [equivalently  4/(2s sqrt(pi)) = 4/(d sqrt(pi)) ].

    Parameters
    ----------
    s : float
        Density-of-states exponent. 1.5 = box, 3 = harmonic, 4.5 = quadrupole.
    Q : float
        Cut-off temperature.
    T : float
        Sample temperature.

    Returns
    -------
    float
        New sample temperature after the cut.
    """

    if not (Q > 0):
        raise ValueError(
            f"mb_temperature requires Q > 0, got Q = {Q!r}. "
            f"The classical MB cut is only defined for a positive cut-off "
            f"temperature; consider clipping the cut-off schedule before this point."
        )
    if not (T > 0):
        raise ValueError(f"mb_temperature requires T > 0, got T = {T!r}.")

    eta = Q / T
    sqrt_eta = mp.sqrt(eta)
    sqrt_pi = mp.sqrt(mp.pi)

    exp_term = mp.exp(-eta)
    erf_term = ss.erf(float(sqrt_eta))

    c1 = 2 / sqrt_pi
    c2 = 2 / (s * sqrt_pi)        # 4 / (d sqrt(pi)) with d = 2s

    num = erf_term - c1 * sqrt_eta * exp_term - c2 * eta ** mp.mpf("1.5") * exp_term
    den = erf_term - c1 * sqrt_eta * exp_term
    return float(T * num / den)