"""Maxwell–Boltzmann (classical) limit of the evaporation recurrences.

Evaporation kernels (used by the MB loop)
-----------------------------------------
  - `mb_particle_number` : trap-independent, used directly by the MB
    evaporation loop.
  - `mb_temperature`     : shared kernel parameterized by the density-of-
    states exponent `s`. Trap classes invoke this from their own
    `mb_temperature` method.

Equilibrium kernels (used by post-processing)
---------------------------------------------
In the classical limit α → −∞, every polylog ratio g_{ν+k}/g_ν collapses
to 1, so the quantum expressions for the equilibrium thermodynamics
reduce to closed-form algebraic functions of (s, N, T, α). See the
unified summary in `mb_limit_based.pdf`, Sec. 5:

    N      = A(T) e^α          (defines α = ln(N/A(T)) given (N, T))
    E      = s N kB T
    Ω      = -N kB T
    P      = N kB T / V_g
    S      = N kB [(s+1) - α]
    H      = (s+1) N kB T
    F      = N kB T (α - 1)
    G      = µ N = N kB T α
    C_V    = s N kB
    C_P    = (s+1) N kB
    κ_T    = V_g / (N kB T)
    B_P    = 1 / T

In contrast to the quantum case, µ and E are not independent state
variables: given (N, T) and the trap-specific A(T), both are fixed. The
state-functions kernel therefore returns Mu and E alongside Ω, S, …, so
the MB post-processing output is self-contained.

The kernels are statistics-independent — there is no `sign` argument.
This file contains no trap-specific logic. Per-trap MB boilerplate is
avoided exactly as it is for the quantum equilibrium kernel.
"""

import mpmath as mp
import scipy.special as ss
from ..constants import ConstantsSI, ConstantsEV

def thermal_wavelength(T):
        return ConstantsEV.h / mp.sqrt(2 * mp.pi * ConstantsEV.m_Na23 * ConstantsEV.kB * mp.mpf(T))


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

# ---------------------------------------------------------------------------
# Equilibrium thermodynamics — Maxwell-Boltzmann limit
# ---------------------------------------------------------------------------
def mb_state_functions_pure_geometry(s, N, T, alpha, V_g, kB):
    """Closed-form MB state functions for any pure-geometry trap.

    Maxwell-Boltzmann is statistics-independent (no `sign` argument), and
    every polylog ratio collapses to unity. The returned dict carries
    Mu and E in addition to the six state functions, since in the MB
    limit these are determined by (N, T) rather than supplied as inputs.

    Parameters
    ----------
    s : float
        Density-of-states exponent. 1.5 = box, 3 = oscillator,
        4.5 = quadrupole.
    N, T : float
        Particle number and temperature.
    alpha : float
        Reduced chemical potential α = µ/(kB T). The caller computes
        this from (N, T) via the trap-specific A(T) prefactor (see
        `Trap.mb_alpha`); this kernel does not know about A(T).
    V_g : float
        Trap global volume. Constant for a given trap.
    kB : float
        Boltzmann constant in the trap's unit system.

    Returns
    -------
    dict
        Keys: Omega, S, P, H, F, G, alpha, Mu, E. Values are mpf.
    """
    T = mp.mpf(T)
    N = mp.mpf(N)
    alpha = mp.mpf(alpha)

    NkT = N * kB * T              # appears in nearly every formula
    Mu = alpha * kB * T
    lam = thermal_wavelength(T)

    Omega = -NkT
    P     = NkT / V_g             # = -Omega / V_g
    S     = N * kB * (s + 1) * [mp.log(V_g/(N * lam**3))+1]
    H     = (s + 1) * NkT
    F     = -NkT * (s + 1) * [mp.log(V_g/(N * lam**3))-1]
    G     = NkT * mp.log(V_g/(N * lam**3))                # = NkT * alpha
    E     = s * NkT

    return {
        "Omega": Omega,
        "S": S,
        "P": P,
        "H": H,
        "F": F,
        "G": G,
        "alpha": alpha,
        "Mu": Mu,
        "E": E,
    }


def mb_thermal_coefficients_pure_geometry(s, N, T, V_g, kB):
    """Closed-form MB thermal coefficients for any pure-geometry trap.

    No polylog evaluations; no `alpha` needed (it cancels out of the
    classical-limit expressions). Statistics-independent.

    Parameters
    ----------
    s, N, T, V_g, kB
        Same conventions as `mb_state_functions_pure_geometry`.

    Returns
    -------
    dict
        Keys: CV, CP, kappa_T, B_P. Values are mpf.
    """
    T = mp.mpf(T)
    N = mp.mpf(N)
    NkB = N * kB

    CV      = s * NkB
    CP      = (s + 1) * NkB
    kappa_T = V_g / (NkB * T)
    B_P     = 1 / T

    return {
        "CV": CV,
        "CP": CP,
        "kappa_T": kappa_T,
        "B_P": B_P,
    }