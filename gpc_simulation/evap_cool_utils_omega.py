"""
Shared utilities for (N, Ω) evaporative cooling simulations.

This module extends evap_cool_utils.py to support an alternative formulation
of the evaporation protocol that uses the grand canonical potential Ω as the
second recurrence relation (instead of the energy E).

The (N, Ω) pair is structurally equivalent to (N, E) — both provide two
independent equations to solve for (T₁, μ₁) after each evaporation step via
Newton–Raphson.  The Ω recurrence has only 2 terms (vs. 3 for E), resulting
in slightly cleaner formulas.

Sign convention note
--------------------
The Ω recurrence derived from first principles gives:

    Ω₁/Ω₀ = g̃_{s_Ω} / g_{s_Ω} − (2/√π) η_c^{1/2} ḡ_{s_ḡ_Ω} / g_{s_Ω}

i.e. MINUS on the ḡ term, consistent with the N and E recurrences.
The Bosons PDF (eqs. 43, 51, 54, 56) writes a PLUS sign; we believe
this is a systematic sign error in the manuscript.  The minus sign is
used here because:
  (a) it follows from the truncated-integral algebra,
  (b) it is the only sign that guarantees |Ω₁| < |Ω₀| for all η_c.

If you want to test the paper's convention, pass ``omega_gbar_sign=+1``
to the factory functions.

Reference
---------
Arvizu-Velázquez et al., "Semi-classical evaporative cooling: classical
and quantum distributions", arXiv (2026).

Poveda-Cuevas, "Semiclassical distribution for Bosons" (2026).
"""

import numpy as np
import mpmath as mp
import time

# Re-export everything from the base module so notebooks only need one import.
from evap_cool_utils import (                          # noqa: F401
    ConstantsEV, ConstantsSI,
    g_tilde, g_bar,
    newton_raphson_1var,
    newton_raphson_2var_fused,
    newton_raphson_2var_fused_real,
    mb_particle_number,
    mb_temperature_box,
    mb_temperature_quadrupole,
    mb_temperature_oscillator,
    create_mb_result_dict,
    build_cutoff_schedule,
    initialize_mb_state,
    run_mb_evaporation,
    plot_combined_overview,
    plot_individual_panels,
    PLOT_COLORS, PLOT_LABELS,
)


# ═══════════════════════════════════════════════════════════════════════
# Polylogarithm-order tables for the three main potentials
# ═══════════════════════════════════════════════════════════════════════
#
# Each potential is characterised by orders (s_N, s_ḡ_N, s_Ω, s_ḡ_Ω):
#
#   Potential     ν    s_N    s_ḡ_N    s_Ω    s_ḡ_Ω    p (T-exponent of C)
#   ─────────    ───   ────   ──────   ────   ──────   ────────────────────
#   3D Box        3    3/2      1      5/2      2        3/2
#   Harmonic      6     3      5/2      4      7/2        3
#   Quadrupole    9    9/2      4      11/2     5          6
#
# s_N, s_ḡ_N  — from the existing (N, E) code (validated).
# s_Ω, s_ḡ_Ω — from the Ω recurrence in the Bosons PDF.
#   Box: eq. 43;  Harmonic: eq. 51;  Quadrupole: derived by analogy.
#
# NOTE on the harmonic N recurrence:
#   The PDF eq. 48 writes ḡ_{5/2}, matching the derivation above.
#   The existing validated code uses g_bar(2, …) for N in the harmonic
#   notebook — effectively ḡ₂.  We preserve the CODE convention here
#   (s_ḡ_N = 2 for the harmonic) so that the N recurrence matches the
#   existing notebooks exactly.  This discrepancy should be checked.
# ═══════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────
# Data containers
# ───────────────────────────────────────────────────────────────────────

def create_omega_result_dict():
    """
    Create an empty results dictionary for the (N, Ω) evaporation protocol.

    Stores the same fields as the (N, E) dict, plus 'Omega' and 'Omegaf'
    (Ω ratio).  E is back-computed from the EOS after each NR step so that
    the two approaches can be compared side-by-side.
    """
    return {
        'N': [], 'T': [], 'Mu': [], 'Omega': [], 'E': [],
        'Q': [], 'Nf': [], 'Tf': [], 'Omegaf': [],
    }


def initialize_omega_state(results, N0, T0, mu0, Omega0, E0=None):
    """Append the initial thermodynamic state to an Omega results dict."""
    results['N'].append(N0)
    results['T'].append(T0)
    results['Mu'].append(mu0)
    results['Omega'].append(Omega0)
    if E0 is not None:
        results['E'].append(E0)


# ───────────────────────────────────────────────────────────────────────
# Initial Ω₀ from the equation of state
# ───────────────────────────────────────────────────────────────────────

def compute_initial_omega(sign, kB, T0, dos_prefactor_T0, s_Omega, alpha0):
    """
    Compute the initial grand canonical potential Ω₀ from the EOS.

        Ω₀ = −kB T₀ · C(T₀) · sign · Li_{s_Ω}(z)

    Parameters
    ----------
    sign : int
        +1 bosons, −1 fermions.
    kB : float
        Boltzmann constant (in the same units as T₀).
    T0 : float
        Initial temperature.
    dos_prefactor_T0 : float or mpf
        Density-of-states prefactor C(T₀), the same quantity used in
        the N equation of state: N = sign · C(T) · Li_{s_N}(z).
    s_Omega : float
        Polylogarithm order for Ω (= s_N + 1).
    alpha0 : float
        Reduced chemical potential α₀ = μ₀ / (kB T₀).

    Returns
    -------
    float
        Ω₀ (negative for a physical Bose or Fermi gas).
    """
    z = sign * mp.exp(alpha0)
    g_Omega = mp.polylog(s_Omega, z)
    Omega0 = float(-kB * T0 * dos_prefactor_T0 * sign * g_Omega)
    return Omega0


# ───────────────────────────────────────────────────────────────────────
# Truncated (N₁, Ω₁) factories
# ───────────────────────────────────────────────────────────────────────

def make_truncated_NOmega(sign, s_N, s_gbar_N, s_Omega, s_gbar_Omega,
                          kB, omega_gbar_sign=-1):
    """
    Factory that returns a fused (N₁, Ω₁) truncation function.

    Parameters
    ----------
    sign : int
        +1 bosons, −1 fermions.
    s_N, s_gbar_N : float
        Polylogarithm orders for the N recurrence (g̃ and ḡ).
    s_Omega, s_gbar_Omega : float
        Polylogarithm orders for the Ω recurrence (g̃ and ḡ).
    kB : float
        Boltzmann constant.
    omega_gbar_sign : int, optional
        Sign in front of the ḡ term in the Ω recurrence.
        −1 (default, derived from first principles) or
        +1 (paper convention, eqs. 43 / 51).

    Returns
    -------
    callable(Ni, Omega_i, Ti, Mui, Qc) → (N1, Omega1)
    """
    def truncated_NOmega(Ni, Omega_i, Ti, Mui, Qc):
        eta_c = Qc / Ti
        alpha = Mui / (kB * Ti)
        z_full = sign * mp.exp(alpha)

        # ── N recurrence ────────────────────────────────────────
        gt_N = g_tilde(s_N, alpha, eta_c, sign)
        g_N_full = sign * mp.polylog(s_N, z_full)
        gb_N = g_bar(s_gbar_N, alpha, eta_c, sign)

        # ── Ω recurrence ────────────────────────────────────────
        gt_Om = g_tilde(s_Omega, alpha, eta_c, sign)
        g_Om_full = sign * mp.polylog(s_Omega, z_full)
        gb_Om = g_bar(s_gbar_Omega, alpha, eta_c, sign)

        c1 = (2 / mp.sqrt(mp.pi)) * mp.sqrt(eta_c)

        N1 = (gt_N / g_N_full - c1 * gb_N / g_N_full) * Ni

        O1 = (gt_Om / g_Om_full + omega_gbar_sign * c1 * gb_Om / g_Om_full) * Omega_i

        return (N1, O1)

    return truncated_NOmega


def make_truncated_NOmega_real(sign, s_N, s_gbar_N, s_Omega, s_gbar_Omega,
                               kB, omega_gbar_sign=-1, precision=35):
    """
    Same as make_truncated_NOmega but casts results to real with given
    precision (needed for the quadrupole where polylogs may acquire tiny
    imaginary parts).
    """
    base_func = make_truncated_NOmega(
        sign, s_N, s_gbar_N, s_Omega, s_gbar_Omega, kB, omega_gbar_sign,
    )

    def truncated_NOmega_real(Ni, Omega_i, Ti, Mui, Qc):
        N1, Om1 = base_func(Ni, Omega_i, Ti, Mui, Qc)
        return (
            float(mp.nstr(mp.re(N1), precision) or 0),
            float(mp.nstr(mp.re(Om1), precision) or 0),
        )

    return truncated_NOmega_real


# ───────────────────────────────────────────────────────────────────────
# Newton–Raphson solvers for the (N, Ω) system
# ───────────────────────────────────────────────────────────────────────
#
# The 2×2 system is:
#
#     f(T, μ) = N(T, μ) − N₁ = 0
#     g(T, μ) = Ω(T, μ) − Ω₁ = 0
#
# where  N = sign · C(T) · Li_{s_N}(z),
#        Ω = −kB T · C(T) · sign · Li_{s_Ω}(z),
#   and  z = sign · e^{μ/(kB T)}.
#
# The Jacobian entries use the same three polylogs as the (N, E) solver:
#   Li_{s_N−1}(z),  Li_{s_N}(z),  Li_{s_Ω}(z) = Li_{s_N+1}(z).
#
# Key thermodynamic identities:
#   ∂Ω/∂μ = −N        (exact, all potentials)
#   ∂Ω/∂T = −S        (entropy, with appropriate sign conventions)
# ───────────────────────────────────────────────────────────────────────

def make_nr_solver_omega(sign, s_N, s_Omega, p_exp, dos_prefactor_func,
                         kB, use_real=False, precision=50):
    """
    Factory for the (N, Ω) Newton–Raphson solver.

    Parameters
    ----------
    sign : int
        +1 bosons, −1 fermions.
    s_N : float
        Polylogarithm order for N (e.g. 3/2, 3, 9/2).
    s_Omega : float
        Polylogarithm order for Ω (= s_N + 1).
    p_exp : float
        Temperature exponent of C(T), defined so that C(T) ∝ T^{p_exp}.
        (3/2 for box, 3 for harmonic, 6 for quadrupole.)
    dos_prefactor_func : callable(T) → mpf
        Returns C(T), the density-of-states prefactor such that
        N = sign · C(T) · Li_{s_N}(z).
    kB : float
        Boltzmann constant.
    use_real : bool
        If True, cast the NR result to real (for quadrupole).
    precision : int
        Digits kept when use_real is True.

    Returns
    -------
    callable(T_init, mu_init, dT, dmu, Ni, Omega_i) → [T_new, mu_new]
        A one-step Newton–Raphson solver.
    """
    s_lo = s_N - 1          # Li_{s_N−1}  (also used in ∂N/∂μ)
    p1 = p_exp + 1          # T-exponent of  kB T · C(T)

    def solver(T_init, mu_init, dT, dmu, Ni, Omega_i):

        def jacobian(T, mu):
            z = sign * mp.exp(mu / (kB * T))

            # Three polylogs (same as the (N, E) solver)
            g_lo  = mp.polylog(s_lo, z)        # Li_{s_N−1}
            g_mid = mp.polylog(s_N, z)         # Li_{s_N}
            g_hi  = mp.polylog(s_Omega, z)     # Li_{s_Ω}

            C = dos_prefactor_func(T)

            # ── f = N(T, μ) − N₁ ──────────────────────────────
            f_val = sign * C * g_mid - Ni

            # ∂N/∂μ = sign · C · Li_{s_N−1}(z) / (kB T)
            f_y = sign * C * g_lo / (kB * T)

            # ∂N/∂T = sign · C / (kB T²) · [p · kB T · Li_{s_N} − μ · Li_{s_N−1}]
            f_x = sign * C * (p_exp * kB * T * g_mid - mu * g_lo) / (kB * T**2)

            # ── g = Ω(T, μ) − Ω₁ ─────────────────────────────
            g_val = -kB * T * C * sign * g_hi - Omega_i

            # ∂Ω/∂T = −(p+1) kB C sign Li_{s_Ω}  +  (μ/T) C sign Li_{s_N}
            g_x = -p1 * kB * C * sign * g_hi + (mu / T) * C * sign * g_mid

            # ∂Ω/∂μ = −C sign Li_{s_N} = −N(T, μ) = −(f_val + Ni)
            g_y = -C * sign * g_mid

            return (f_val, g_val, f_x, f_y, g_x, g_y)

        if use_real:
            return newton_raphson_2var_fused_real(
                jacobian, T_init, mu_init, dT, dmu, precision,
            )
        else:
            return newton_raphson_2var_fused(
                jacobian, T_init, mu_init, dT, dmu,
            )

    return solver


# ───────────────────────────────────────────────────────────────────────
# Simulation loop (replaces run_quantum_evaporation for the Ω approach)
# ───────────────────────────────────────────────────────────────────────

def run_quantum_evaporation_omega(results, truncated_NOmega_func,
                                  nr_func, N0, n_steps, dT, dmu,
                                  eos_energy_func=None):
    """
    Run the recursive (N, Ω) evaporation protocol for a quantum gas.

    Parameters
    ----------
    results : dict
        Pre-initialised results dictionary (from create_omega_result_dict).
    truncated_NOmega_func : callable(N, Omega, T, mu, Q) → (N1, Omega1)
        Fused truncated particle-number and grand-canonical-potential function.
    nr_func : callable(T, mu, dT, dmu, N1, Omega1) → [T_new, mu_new]
        Newton–Raphson solver for (T, μ) using the (N, Ω) pair.
    N0 : float
        Initial particle number (for computing Nf = N/N₀).
    n_steps : int
        Number of evaporation steps.
    dT, dmu : float
        NR perturbation offsets for temperature and chemical potential.
    eos_energy_func : callable(N, T, mu) → E, optional
        If provided, computes E from the EOS after each step so that the
        energy trajectory can be compared with the (N, E) approach.
    """
    T0 = results['T'][0]
    Omega0 = results['Omega'][0]

    for i in range(n_steps):
        N_new, Om_new = truncated_NOmega_func(
            results['N'][i], results['Omega'][i],
            results['T'][i], results['Mu'][i],
            results['Q'][i],
        )
        results['N'].append(N_new)
        results['Nf'].append(results['N'][i] / N0)
        results['Omega'].append(Om_new)
        results['Omegaf'].append(results['Omega'][i] / Omega0)

        T_mu = nr_func(
            results['T'][i], results['Mu'][i],
            dT, dmu,
            results['N'][i + 1], results['Omega'][i + 1],
        )
        results['T'].append(T_mu[0])
        results['Tf'].append(results['T'][i] / T0)
        results['Mu'].append(T_mu[1])

        # Optionally back-compute E for comparison
        if eos_energy_func is not None:
            E_new = eos_energy_func(
                results['N'][i + 1],
                results['T'][i + 1],
                results['Mu'][i + 1],
            )
            results['E'].append(E_new)


# ───────────────────────────────────────────────────────────────────────
# Convenience EOS energy functions (for back-computation)
# ───────────────────────────────────────────────────────────────────────
#
# E = (ν/2) · N · kB T · g_{s_Ω}(α) / g_{s_N}(α)
#
# where ν/2 = 3/2 (box), 3 (HO), 9/2 (quadrupole).
# ───────────────────────────────────────────────────────────────────────

def make_eos_energy(sign, s_N, s_Omega, nu_half, kB):
    """
    Factory for E(N, T, μ) from the equation of state.

    Parameters
    ----------
    sign : int
        +1 bosons, −1 fermions.
    s_N : float
        Polylogarithm order for N.
    s_Omega : float
        Polylogarithm order for Ω (= s_N + 1, also the E-denominator order).
    nu_half : float
        Half the density-of-states exponent: 3/2 (box), 3 (HO), 9/2 (quad).
    kB : float
        Boltzmann constant.

    Returns
    -------
    callable(N, T, mu) → float
    """
    def eos_energy(N, T, mu):
        alpha = mu / (kB * T)
        z = sign * mp.exp(alpha)
        g_N = mp.polylog(s_N, z)
        g_Om = mp.polylog(s_Omega, z)
        # For fermions: sign · Li_s(sign · e^α) gives the appropriate g^{(±)}
        ratio = g_Om / g_N
        return float(nu_half * N * kB * T * ratio)

    return eos_energy


# ───────────────────────────────────────────────────────────────────────
# Comparison plotting
# ───────────────────────────────────────────────────────────────────────

def plot_omega_vs_energy_comparison(results_NE, results_NOm, trap_name,
                                    n_NE=None, n_NOm=None):
    """
    Overlay (N, E) and (N, Ω) trajectories to validate equivalence.

    Plots N-vs-T and (if available) E-vs-step for both approaches.
    """
    from matplotlib import pyplot as plt

    n_NE  = n_NE  or min(len(results_NE.get('Tf', results_NE['T'])),
                          len(results_NE['N']))
    n_NOm = n_NOm or min(len(results_NOm.get('Tf', results_NOm['T'])),
                          len(results_NOm['N']))

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # ── Panel 0: N vs T ──────────────────────────────────────
    ax = axes[0]
    ax.scatter(results_NE['T'][:n_NE], results_NE['N'][:n_NE],
               c='tab:blue', s=10, alpha=0.6, label='(N, E)')
    ax.scatter(results_NOm['T'][:n_NOm], results_NOm['N'][:n_NOm],
               c='tab:orange', s=10, alpha=0.6, marker='x', label=r'(N, $\Omega$)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Temperature [K]', fontsize=13)
    ax.set_ylabel('Number of atoms', fontsize=13)
    ax.set_title(f'{trap_name}: N vs T — comparison', fontsize=14)
    ax.legend(fontsize=11)

    # ── Panel 1: T vs Q ──────────────────────────────────────
    ax = axes[1]
    ax.scatter(results_NE['Q'][:n_NE], results_NE['T'][:n_NE],
               c='tab:blue', s=10, alpha=0.6, label='(N, E)')
    ax.scatter(results_NOm['Q'][:n_NOm], results_NOm['T'][:n_NOm],
               c='tab:orange', s=10, alpha=0.6, marker='x', label=r'(N, $\Omega$)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Cut-off temperature $Q$ [K]', fontsize=13)
    ax.set_ylabel('Sample temperature [K]', fontsize=13)
    ax.set_title(f'{trap_name}: T vs Q — comparison', fontsize=14)
    ax.legend(fontsize=11)

    # ── Panel 2: relative difference ─────────────────────────
    ax = axes[2]
    n_cmp = min(n_NE, n_NOm)
    if n_cmp > 1:
        T_ne  = np.array([float(t) for t in results_NE['T'][:n_cmp]])
        T_nom = np.array([float(t) for t in results_NOm['T'][:n_cmp]])
        N_ne  = np.array([float(n) for n in results_NE['N'][:n_cmp]])
        N_nom = np.array([float(n) for n in results_NOm['N'][:n_cmp]])

        with np.errstate(divide='ignore', invalid='ignore'):
            dT_rel = np.abs((T_nom - T_ne) / T_ne)
            dN_rel = np.abs((N_nom - N_ne) / N_ne)

        steps = np.arange(n_cmp)
        ax.semilogy(steps, dT_rel, 'tab:red',  lw=1.5, label=r'$|\Delta T/T|$')
        ax.semilogy(steps, dN_rel, 'tab:green', lw=1.5, label=r'$|\Delta N/N|$')
        ax.set_xlabel('Evaporation step', fontsize=13)
        ax.set_ylabel('Relative difference', fontsize=13)
        ax.set_title(f'{trap_name}: (N,E) vs (N,Ω) agreement', fontsize=14)
        ax.legend(fontsize=11)

    fig.tight_layout()
    return fig