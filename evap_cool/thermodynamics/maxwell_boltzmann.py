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

    N      = A(T) e^α                       (defines α = ln(N/A(T)))
    E      = s N kB T                                       [unchanged]
    Ω      = -N kB T                                        [unchanged]
    P      = N kB T / V_g                                   [unchanged]
    S      = N kB (s+1) { ln(V/Nλ³) + 1 }   = N kB (s+1) (1 - α)
    H      = (s+1) N kB T                                   [unchanged]
    F      = -N kB T (s+1) { ln(V/Nλ³) - 1 } = N kB T (s+1) (α + 1)
    G      = N kB T ln(V/Nλ³)                = -N kB T α
    C_V    = s N kB                                         [unchanged]
    C_P    = (s+1) N kB                                     [unchanged]
    κ_T    = V_g / (N kB T)                                 [unchanged]
    B_P    = 1 / T                                          [unchanged]

Revision (June 2026, per Poveda-Cuevas)
---------------------------------------
S, F, G were updated to the closed forms above, expressed through the
thermal wavelength as  ln(V/Nλ³).  Ω, E, P, H and all four thermal
coefficients are unchanged.

Two implementation points are worth recording explicitly:

  * **ln(V/Nλ³) is evaluated as -α, not as a literal log.**  For the box
    A(T) = V/λ³, so  ln(V/Nλ³) = ln(A(T)/N) = -α  exactly.  For the other
    geometries (oscillator, quadrupole, the two mixed traps) `V_g` is NOT
    a volume — it is 1/ω³, 1/Ā³, or a fixed literal scale — so the literal
    quotient V_g/λ³ is dimensionally inconsistent and unit-system
    dependent.  Using -α (with the trap-correct A(T) = `_prefactor_N`)
    keeps the kernel dimensionless, unit-safe, and identical to the box
    λ-form, with no per-trap MB code.

  * **Earlier (superseded) forms.**  Prior to this revision the kernel
    used  S = N kB [(s+1) - α],  F = N kB T (α - 1),  G = N kB T α  — the
    canonical, Sackur-Tetrode-consistent classical limit.  They are kept
    here only as a record; the active expressions are those above.

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
    """De Broglie thermal wavelength λ(T) = h / sqrt(2 π m kB T).

    .. deprecated::
        No longer used by any kernel in this module.  The equilibrium
        state-functions kernel now expresses ln(V/Nλ³) as -α (see module
        docstring), so it never forms λ explicitly.  Each trap class
        carries its own `thermal_wavelength` in a *consistent* unit system
        (BoxTrap/mixed: SI; QuadrupoleTrap: its own constants); use those
        if you need λ.  This module-level helper mixes ConstantsEV energy
        units with a kg mass and is therefore numerically inconsistent —
        retained only so existing imports do not break.
    """
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

    S, F, G follow the June-2026 revision (see module docstring):

        S = N kB (s+1) { ln(V/Nλ³) + 1 }
        F = -N kB T (s+1) { ln(V/Nλ³) - 1 }
        G =  N kB T ln(V/Nλ³)

    with the log term evaluated as  ln(V/Nλ³) = ln(A(T)/N) = -α.  This is
    the box λ-form exactly, and the correct dimensionless generalization
    for every other geometry (where V_g is not a volume).  Ω, P, H, E are
    unchanged from the canonical classical limit.

    Parameters
    ----------
    s : float
        Density-of-states exponent. 1.5 = box, 2 = box2d_osc1d,
        2.5 = osc2d_box1d, 3 = oscillator, 4.5 = quadrupole.
    N, T : float
        Particle number and temperature.
    alpha : float
        Reduced chemical potential α = µ/(kB T) = ln(N / A(T)). The caller
        computes this from (N, T) via the trap-specific A(T) prefactor
        (see `Trap.mb_alpha`); this kernel does not know about A(T).
    V_g : float
        Trap global volume. Constant for a given trap. Enters only the
        pressure P = N kB T / V_g (unchanged by the revision); it is NOT
        used to form the log term, which is taken as -α (see above).
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

    # ln(V/Nλ³) = ln(A(T)/N) = -α  for every trap (A(T) = _prefactor_N).
    # Box: A(T) = V/λ³, so this is literally ln(V/Nλ³). Other traps: V_g is
    # not a volume, so the literal λ-form would be unit-inconsistent; -α is
    # the dimensionless, unit-safe generalization.
    log_term = -alpha

    Omega = -NkT
    P     = NkT / V_g                                  # = -Omega / V_g
    S     = N * kB * (log_term + (s + 1))          # Nk(s+1){ln(V/Nλ³)+1}
    H     = (s + 1) * NkT
    F     = -NkT * (log_term - 1)            # -NkT(s+1){ln(V/Nλ³)-1}
    G     = -NkT * log_term                             # NkT·ln(V/Nλ³) = -NkT·α
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
    classical-limit expressions). Statistics-independent. Unchanged by the
    June-2026 revision.

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