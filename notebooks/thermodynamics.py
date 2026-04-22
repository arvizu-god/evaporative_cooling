"""
thermodynamics.py
=================

Equilibrium thermodynamic quantities for the semiclassical evaporative-cooling
simulation.  This module is a POST-PROCESSING layer: given a converged state
(T, α, N) at each evaporation step, it evaluates closed-form expressions for
the grand-canonical potentials and derived state functions.

Organization (three layers)
---------------------------
Layer 1 — Core scalar functions: one per (geometry, quantity) pair.  Pure,
          stateless, return ``mpmath.mpf``.  Take (T, alpha, trap params, sign).
Layer 2 — Per-geometry bundler: ``all_potentials_<geom>`` returns a dict of
          floats for a single evaporation step.
Layer 3 — Results adapter: ``compute_thermodynamics_<geom>`` walks an
          evaporation results dict and populates numpy arrays.

Currently implemented
---------------------
* 3D BOX geometry, SI units.
* BE (sign=+1), FD (sign=-1), and MB (statistics-free).

Conventions
-----------
* SI units throughout.  T [K], V [m³], energies [J], α dimensionless.
* ``results['Mu']`` stores the REDUCED chemical potential α = μ/(k_B T),
  NOT μ in joules.  The physical μ is recovered as μ = α · k_B · T.
* Sign convention (matches ``g_bar``/``g_standard`` in evap_cool_utils):
      g_s^(+)(α) =  Li_s(  e^α )   for bosons  (sign = +1)
      g_s^(-)(α) = -Li_s( -e^α )   for fermions (sign = -1)
* All internal arithmetic uses mpmath; values are cast to float at the
  adapter boundary (Layer 3) for numpy-friendly arrays.

Reference
---------
F. J. Poveda-Cuevas, *Semiclassical distribution for Bosons*, §1–§2.1.
"""

import numpy as np
import mpmath as mp

from evap_cooling_utils import ConstantsSI, g_standard


# ---------------------------------------------------------------------------
# Module-level constants and helpers
# ---------------------------------------------------------------------------

_C = ConstantsSI


def _lambda_dB(T):
    """de Broglie thermal wavelength λ = h / sqrt(2π m k_B T)  [SI, mpf]."""
    T = mp.mpf(T)
    return _C.h / mp.sqrt(2 * mp.pi * _C.m_Na23 * _C.kB * T)


# The polylog orders we use are half-integers; keep them as mpf so the
# polylog call never sees Python floats where precision matters.
_S_HALF = mp.mpf('0.5')
_S_THREE_HALF = mp.mpf('1.5')
_S_FIVE_HALF = mp.mpf('2.5')


# =============================================================================
# 3D BOX — Layer 1 (core scalar functions)
# =============================================================================
#
# Analytical expressions  (ideal quantum gas, box of volume V):
#
#     Ω(T, α, V)  =  -(V k_B T / λ³) g_{5/2}(α)
#     N(T, α, V)  =   (V / λ³)       g_{3/2}(α)
#     E(T, α, V)  =  (3/2)(V k_B T / λ³) g_{5/2}(α)   =  -(3/2) Ω
#     P(T, α)     =   (k_B T / λ³) g_{5/2}(α)          =  -Ω / V
#     S(T, α, V)  =  (V k_B / λ³) [ (5/2) g_{5/2}(α) - α g_{3/2}(α) ]
#     F           =  Ω + μN                                 (= E - TS)
#     G           =  μN                                     (= α k_B T N)
#     H           =  E + PV  =  (5/2)(V k_B T / λ³) g_{5/2}(α)   (= -(5/2) Ω)
#
# All of these are evaluated on the *rethermalized* equilibrium state that
# follows each cut-off step, NOT on the truncated distribution.
# =============================================================================

def omega_box(T, alpha, V, sign=+1):
    """Grand-canonical potential Ω [J].  Ω = -(V k_B T / λ³) g_{5/2}(α)."""
    T = mp.mpf(T)
    lam = _lambda_dB(T)
    g52 = g_standard(_S_FIVE_HALF, alpha, sign)
    return -V * _C.kB * T / lam ** 3 * g52


def number_box(T, alpha, V, sign=+1):
    """Particle number from Ω: N = -(∂Ω/∂μ) = (V/λ³) g_{3/2}(α).

    Used as a CONSISTENCY CHECK against results['N'] from the Newton step.
    """
    T = mp.mpf(T)
    lam = _lambda_dB(T)
    g32 = g_standard(_S_THREE_HALF, alpha, sign)
    return V / lam ** 3 * g32


def energy_box(T, alpha, V, sign=+1):
    """Mean energy E [J].  E = (3/2)(V k_B T / λ³) g_{5/2}(α) = -(3/2) Ω."""
    T = mp.mpf(T)
    lam = _lambda_dB(T)
    g52 = g_standard(_S_FIVE_HALF, alpha, sign)
    return mp.mpf('1.5') * V * _C.kB * T / lam ** 3 * g52


def pressure_box(T, alpha, sign=+1):
    """Pressure P [Pa].  P = -Ω/V = (k_B T / λ³) g_{5/2}(α).

    Note: V does not appear because P is intensive — the box volume
    cancels in Ω/V.
    """
    T = mp.mpf(T)
    lam = _lambda_dB(T)
    g52 = g_standard(_S_FIVE_HALF, alpha, sign)
    return _C.kB * T / lam ** 3 * g52


def entropy_box(T, alpha, V, sign=+1):
    """Entropy S [J/K].  S = (V k_B / λ³) [ (5/2) g_{5/2}(α) - α g_{3/2}(α) ].

    Derivation: S = -(∂Ω/∂T)_{μ,V}, using α = μ/(k_B T) and
    ∂g_s(α)/∂α = g_{s-1}(α).
    """
    T = mp.mpf(T)
    alpha = mp.mpf(alpha)
    lam = _lambda_dB(T)
    g52 = g_standard(_S_FIVE_HALF, alpha, sign)
    g32 = g_standard(_S_THREE_HALF, alpha, sign)
    return V * _C.kB / lam ** 3 * (mp.mpf('2.5') * g52 - alpha * g32)


def helmholtz_box(T, alpha, N, V, sign=+1):
    """Helmholtz free energy F [J].  F = Ω + μN = Ω + α k_B T N.

    We use F = Ω + μN (rather than F = E - TS) because it avoids a
    large cancellation between the two (5/2) g_{5/2} contributions in
    E and TS, yielding better numerical stability.
    """
    T_mp = mp.mpf(T)
    alpha_mp = mp.mpf(alpha)
    mu_joules = alpha_mp * _C.kB * T_mp
    return omega_box(T, alpha, V, sign) + mu_joules * mp.mpf(N)


def gibbs_box(T, alpha, N):
    """Gibbs free energy G [J].  G = μN = α k_B T N.

    For the box (homogeneous), μ = ∂G/∂N at constant (T, P) and
    G is simply μN.  Statistics-independent at fixed (T, α, N).
    """
    return mp.mpf(alpha) * _C.kB * mp.mpf(T) * mp.mpf(N)


def enthalpy_box(T, alpha, V, sign=+1):
    """Enthalpy H [J].  H = E + PV = (5/2)(V k_B T / λ³) g_{5/2}(α) = -(5/2) Ω.

    For the ideal quantum gas in a box, H = (5/3) E — a clean
    consistency identity worth spot-checking.
    """
    T = mp.mpf(T)
    lam = _lambda_dB(T)
    g52 = g_standard(_S_FIVE_HALF, alpha, sign)
    return mp.mpf('2.5') * V * _C.kB * T / lam ** 3 * g52


# =============================================================================
# 3D BOX — Layer 2 (bundler)
# =============================================================================

def all_potentials_box(T, alpha, N, V, sign=+1):
    """
    Evaluate every thermodynamic potential at a single (T, α, N) point for
    the 3D box geometry.

    Parameters
    ----------
    T : float
        Temperature [K].
    alpha : float
        Reduced chemical potential α = μ/(k_B T), dimensionless.
    N : float
        Particle number (from the converged Newton step).
    V : float
        Box volume [m³].
    sign : int, optional
        +1 for bosons, -1 for fermions.  Default +1.

    Returns
    -------
    dict of float
        Keys: 'Omega', 'S', 'E', 'F', 'G', 'H', 'P', 'N_check'.
        'N_check' is N recomputed from (V/λ³) g_{3/2}(α) — should match
        the input N to Newton-Raphson tolerance.
    """
    Omega   = omega_box(T, alpha, V, sign)
    S       = entropy_box(T, alpha, V, sign)
    E       = energy_box(T, alpha, V, sign)
    P       = pressure_box(T, alpha, sign)
    H       = enthalpy_box(T, alpha, V, sign)
    F       = helmholtz_box(T, alpha, N, V, sign)
    G       = gibbs_box(T, alpha, N)
    N_check = number_box(T, alpha, V, sign)

    return {
        'Omega':   float(Omega),
        'S':       float(S),
        'E':       float(E),
        'F':       float(F),
        'G':       float(G),
        'H':       float(H),
        'P':       float(P),
        'N_check': float(N_check),
    }


# =============================================================================
# 3D BOX — Layer 3 (results adapter)
# =============================================================================

def compute_thermodynamics_box(results, V, sign):
    """
    Walk an evaporation results dict and populate arrays of thermodynamic
    quantities at every step.

    Parameters
    ----------
    results : dict
        Produced by ``run_quantum_evaporation``.  Must contain 'T', 'Mu', 'N'
        as lists, one entry per evaporation step.  'Mu' stores α (dimensionless).
        'E' (from the recurrence) is optional; if present it will be diffed
        against the post-processed 'E_thermo'.
    V : float
        Box volume [m³].
    sign : int
        +1 for bosons, -1 for fermions.

    Side effects
    ------------
    Adds the following numpy arrays (length == number of steps) to ``results``:
        'Omega', 'S', 'E_thermo', 'F', 'G', 'H', 'P', 'N_check'

    The original 'E' array (from the evaporation recurrence) is preserved
    untouched; the analytical version is stored as 'E_thermo' so the two
    can be compared in ``check_consistency_box``.

    Returns
    -------
    dict
        Same ``results`` reference, now augmented.
    """
    T_arr  = results['T']
    Mu_arr = results['Mu']   # α = μ/(k_B T), dimensionless
    N_arr  = results['N']
    n_steps = len(T_arr)

    keys = ['Omega', 'S', 'E_thermo', 'F', 'G', 'H', 'P', 'N_check']
    out = {k: np.zeros(n_steps) for k in keys}

    for i in range(n_steps):
        d = all_potentials_box(T_arr[i], Mu_arr[i], N_arr[i], V, sign)
        out['Omega'][i]    = d['Omega']
        out['S'][i]        = d['S']
        out['E_thermo'][i] = d['E']
        out['F'][i]        = d['F']
        out['G'][i]        = d['G']
        out['H'][i]        = d['H']
        out['P'][i]        = d['P']
        out['N_check'][i]  = d['N_check']

    results.update(out)
    return results


# =============================================================================
# 3D BOX (MB) — Layer 1 (core scalar functions)
# =============================================================================
#
# Classical (Maxwell-Boltzmann) limit of the quantum expressions: every
# polylog g_s(α) collapses to exp(α).  The full set of closed forms:
#
#     Ω  = -(V k_B T / λ³) e^α  =  -N k_B T
#     N  =  (V / λ³) e^α
#     E  =  (3/2) N k_B T
#     P  =  N k_B T / V                             (ideal gas law)
#     S  =  N k_B (5/2 - α)                         (Sackur-Tetrode)
#     F  =  N k_B T (α - 1)
#     G  =  α k_B T N
#     H  =  (5/2) N k_B T
#
# The MB evaporation recurrence does NOT track α.  We recover it exactly
# by inverting N = (V/λ³) e^α:   α = ln(N λ³ / V).  No Newton-Raphson.
# =============================================================================

def alpha_from_NT_box(T, N, V):
    """Recover α = μ/(k_B T) from (N, T, V) in the MB limit.

    Exact inversion of N = (V/λ³) e^α:

        α = ln(N λ³ / V)

    Used by the MB adapter to generate a per-step α that downstream
    code can treat uniformly with the BE/FD results.
    """
    T = mp.mpf(T)
    lam = _lambda_dB(T)
    return mp.log(mp.mpf(N) * lam ** 3 / V)


def omega_box_mb(T, N):
    """MB grand-canonical potential Ω [J].  Ω = -N k_B T."""
    return -mp.mpf(N) * _C.kB * mp.mpf(T)


def number_box_mb(T, alpha, V):
    """MB particle number N.  N = (V/λ³) e^α  (inverse of ``alpha_from_NT_box``)."""
    T = mp.mpf(T)
    lam = _lambda_dB(T)
    return V / lam ** 3 * mp.exp(mp.mpf(alpha))


def energy_box_mb(T, N):
    """MB energy E [J] (equipartition).  E = (3/2) N k_B T."""
    return mp.mpf('1.5') * mp.mpf(N) * _C.kB * mp.mpf(T)


def pressure_box_mb(T, N, V):
    """MB pressure P [Pa] (ideal gas law).  P = N k_B T / V."""
    return mp.mpf(N) * _C.kB * mp.mpf(T) / V


def entropy_box_mb(T, alpha, N):
    """MB entropy S [J/K] (Sackur-Tetrode form).  S = N k_B (5/2 - α)."""
    return mp.mpf(N) * _C.kB * (mp.mpf('2.5') - mp.mpf(alpha))


def helmholtz_box_mb(T, alpha, N):
    """MB Helmholtz free energy F [J].  F = N k_B T (α - 1) = Ω + μN."""
    return mp.mpf(N) * _C.kB * mp.mpf(T) * (mp.mpf(alpha) - 1)


def gibbs_box_mb(T, alpha, N):
    """MB Gibbs free energy G [J].  G = α k_B T N = μN."""
    return mp.mpf(alpha) * _C.kB * mp.mpf(T) * mp.mpf(N)


def enthalpy_box_mb(T, N):
    """MB enthalpy H [J].  H = (5/2) N k_B T = -(5/2) Ω."""
    return mp.mpf('2.5') * mp.mpf(N) * _C.kB * mp.mpf(T)


# =============================================================================
# 3D BOX (MB) — Layer 2 (bundler)
# =============================================================================

def all_potentials_box_mb(T, N, V):
    """
    Evaluate every MB thermodynamic potential at (T, N) for the 3D box.

    α is derived internally from α = ln(N λ³ / V); no ``sign`` argument —
    MB is statistics-free.

    Parameters
    ----------
    T : float
        Temperature [K].
    N : float
        Particle number (from the MB evaporation recurrence).
    V : float
        Box volume [m³].

    Returns
    -------
    dict of float
        Keys: 'Omega', 'S', 'E', 'F', 'G', 'H', 'P', 'alpha'.
        'alpha' is the derived reduced chemical potential α = ln(Nλ³/V),
        stored alongside the potentials so downstream code has it.
    """
    alpha = alpha_from_NT_box(T, N, V)

    Omega = omega_box_mb(T, N)
    E     = energy_box_mb(T, N)
    P     = pressure_box_mb(T, N, V)
    S     = entropy_box_mb(T, alpha, N)
    F     = helmholtz_box_mb(T, alpha, N)
    G     = gibbs_box_mb(T, alpha, N)
    H     = enthalpy_box_mb(T, N)

    return {
        'Omega': float(Omega),
        'S':     float(S),
        'E':     float(E),
        'F':     float(F),
        'G':     float(G),
        'H':     float(H),
        'P':     float(P),
        'alpha': float(alpha),
    }


# =============================================================================
# 3D BOX (MB) — Layer 3 (results adapter)
# =============================================================================

def compute_thermodynamics_box_mb(results, V):
    """
    Walk an MB evaporation results dict and populate thermodynamic arrays
    at every step.

    Parameters
    ----------
    results : dict
        Produced by ``run_mb_evaporation``.  Must contain 'T' and 'N' as
        lists, one entry per step.  Note that the MB recurrence does NOT
        track α or E; the adapter derives both from (T, N, V) analytically.
    V : float
        Box volume [m³].

    Side effects
    ------------
    Adds the following numpy arrays (length == number of steps) to ``results``:
        'Omega', 'S', 'E_thermo', 'F', 'G', 'H', 'P', 'Mu'

    'Mu' stores the derived α = ln(N λ³ / V) at each step, matching the
    BE/FD convention where ``results['Mu']`` holds the reduced chemical
    potential.  This lets downstream plotting treat all three statistics
    uniformly.

    Returns
    -------
    dict
        Same ``results`` reference, now augmented.
    """
    T_arr = results['T']
    N_arr = results['N']
    n_steps = len(T_arr)

    keys = ['Omega', 'S', 'E_thermo', 'F', 'G', 'H', 'P', 'Mu']
    out = {k: np.zeros(n_steps) for k in keys}

    for i in range(n_steps):
        d = all_potentials_box_mb(T_arr[i], N_arr[i], V)
        out['Omega'][i]    = d['Omega']
        out['S'][i]        = d['S']
        out['E_thermo'][i] = d['E']
        out['F'][i]        = d['F']
        out['G'][i]        = d['G']
        out['H'][i]        = d['H']
        out['P'][i]        = d['P']
        out['Mu'][i]       = d['alpha']

    results.update(out)
    return results


# =============================================================================
# Consistency checks
# =============================================================================

def check_consistency_box(results, tol_euler=1e-8, tol_N=1e-6, tol_E=1e-6,
                          verbose=True):
    """
    Verify grand-canonical identities on every evaporation step.

    Three independent checks:

    1. **Euler relation**:  Ω + TS + μN  ≈  E_thermo
       (Legendre-transform identity; fails if any of Ω, S, E is wrong.)
    2. **Number consistency**:  N_check  ≈  N
       (Verifies the Newton-Raphson (T, μ) inversion actually converged.)
    3. **Energy consistency**:  E_thermo  ≈  E_evap
       (The recurrence-relation E and the analytic Ω-derivative E must agree.)

    Parameters
    ----------
    results : dict
        Must have been populated by ``compute_thermodynamics_box``.
    tol_euler, tol_N, tol_E : float
        Relative tolerances.  Defaults assume SI + mpmath default precision.
    verbose : bool
        If True, print a summary and warn on exceeded tolerances.

    Returns
    -------
    dict
        {'euler': max_rel_residual, 'N': ..., 'E': ...}
    """
    T  = np.asarray(results['T'],  dtype=float)
    Mu = np.asarray(results['Mu'], dtype=float)   # α (dimensionless)
    N  = np.asarray(results['N'],  dtype=float)

    Omega    = np.asarray(results['Omega'])
    S        = np.asarray(results['S'])
    E_thermo = np.asarray(results['E_thermo'])
    N_check  = np.asarray(results['N_check'])

    # Convert α back to μ [J] for the Euler relation.
    mu_joules = Mu * _C.kB * T

    euler_lhs = Omega + T * S + mu_joules * N
    euler_res = np.max(np.abs(euler_lhs - E_thermo) / np.abs(E_thermo))
    N_res = np.max(np.abs(N_check - N) / np.abs(N))

    if 'E' in results:
        E_evap = np.asarray(results['E'], dtype=float)
        E_res = np.max(np.abs(E_thermo - E_evap) / np.abs(E_evap))
    else:
        E_res = float('nan')

    residuals = {'euler': euler_res, 'N': N_res, 'E': E_res}

    if verbose:
        print("Consistency check (3D box):")
        print(f"  Euler residual    (Ω + TS + μN vs E):  {euler_res:.2e}"
              f"    {'OK' if euler_res <= tol_euler else 'FAIL'}")
        print(f"  N residual        (N_check vs N):      {N_res:.2e}"
              f"    {'OK' if N_res <= tol_N else 'FAIL'}")
        if not np.isnan(E_res):
            print(f"  E residual        (thermo vs recurrence): {E_res:.2e}"
                  f"    {'OK' if E_res <= tol_E else 'FAIL'}")
        else:
            print("  E residual        (skipped — 'E' not in results)")

    return residuals

def check_consistency_box_mb(results, V, tol=1e-12, verbose=True):
    """
    Verify MB thermodynamic identities on every evaporation step.

    Three checks (all tautological given exact MB closed forms — so they
    test for coding errors, not physics):

    1. **Ideal gas law**:   PV ≈ N k_B T
    2. **Equipartition**:   E  ≈ (3/2) N k_B T
    3. **Euler relation**:  Ω + TS + μN ≈ E

    Because there are no polylog series to truncate, residuals should
    sit near machine precision (~1e-15).  Anything larger indicates a
    typo or a unit mismatch.

    Parameters
    ----------
    results : dict
        Must have been populated by ``compute_thermodynamics_box_mb``.
    V : float
        Box volume [m³] — needed to reconstruct PV from P.
    tol : float
        Relative tolerance (default 1e-12, loose enough for float64).
    verbose : bool
        If True, print a summary and flag exceedances.

    Returns
    -------
    dict
        {'ideal_gas': max_rel_residual, 'equipartition': ..., 'euler': ...}
    """
    T  = np.asarray(results['T'],  dtype=float)
    N  = np.asarray(results['N'],  dtype=float)
    Mu = np.asarray(results['Mu'], dtype=float)   # derived α

    Omega = np.asarray(results['Omega'])
    S     = np.asarray(results['S'])
    E     = np.asarray(results['E_thermo'])
    P     = np.asarray(results['P'])

    NkT = N * _C.kB * T
    mu_joules = Mu * _C.kB * T

    ideal_res = np.max(np.abs(P * V - NkT) / np.abs(NkT))
    equip_res = np.max(np.abs(E - 1.5 * NkT) / np.abs(E))
    euler_res = np.max(np.abs(Omega + T * S + mu_joules * N - E) / np.abs(E))

    residuals = {
        'ideal_gas':     ideal_res,
        'equipartition': equip_res,
        'euler':         euler_res,
    }

    if verbose:
        print("Consistency check (3D box, MB):")
        print(f"  Ideal gas law  (PV vs NkT):       {ideal_res:.2e}"
              f"    {'OK' if ideal_res <= tol else 'FAIL'}")
        print(f"  Equipartition  (E vs (3/2)NkT):   {equip_res:.2e}"
              f"    {'OK' if equip_res <= tol else 'FAIL'}")
        print(f"  Euler relation (Ω+TS+μN vs E):    {euler_res:.2e}"
              f"    {'OK' if euler_res <= tol else 'FAIL'}")

    return residuals