"""
Shared utilities for semiclassical evaporative cooling simulations.

This module provides common functions used across the three trapping potential
geometries (3D box, quadrupole, harmonic oscillator) for simulating evaporative
cooling of quantum and classical gases.

Reference:
    Arvizu-Velázquez et al., "Semi-classical evaporative cooling: classical
    and quantum distributions", arXiv (2026).
"""

import numpy as np
import mpmath as mp
import scipy.special as ss
from matplotlib import pyplot as plt
import time

import json
from pathlib import Path
from datetime import datetime


def save_results_snapshot(results, path, metadata=None):
    """
    Serialize an evaporation results dict to JSON.

    Converts mpmath mpf values to float for portability. Safe to call on
    partial results (e.g. after early halt) — only writes what's present.

    Parameters
    ----------
    results : dict
        Evaporation results dict (possibly partial).
    path : str or Path
        Output file path. Parent directory is created if missing.
    metadata : dict, optional
        Extra info to embed (halt reason, step count, physical params, etc).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _to_float(x):
        # mpmath mpf, numpy scalars, and regular numbers all survive float()
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    serializable = {
        key: [_to_float(v) for v in vals] if isinstance(vals, list) else vals
        for key, vals in results.items()
    }

    payload = {
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'metadata': metadata or {},
        'results': serializable,
    }

    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_results_snapshot(path):
    """Load a snapshot saved by save_results_snapshot. Returns (results, metadata)."""
    with open(path) as f:
        payload = json.load(f)
    return payload['results'], payload['metadata']


# ---------------------------------------------------------------------------
# Physical constants (eV-based units used in quadrupole & oscillator notebooks)
# ---------------------------------------------------------------------------
class ConstantsEV:
    """Physical constants in eV-based units (h in eV·s, kB in eV/K)."""
    h = 4.135667696e-15       # Planck constant [eV·s]
    hbar = h / (2 * np.pi)    # Reduced Planck constant [eV·s]
    kB = 8.617333262e-5       # Boltzmann constant [eV/K]
    m_Na23 = 3.817545e-26     # Mass of ²³Na [kg]


class ConstantsSI:
    """Physical constants in SI units."""
    h = 6.62607004e-34        # Planck constant [J·s]
    hbar = h / (2 * np.pi)    # Reduced Planck constant [J·s]
    kB = 1.38064852e-23       # Boltzmann constant [J/K]
    m_Na23 = 3.817545e-26     # Mass of ²³Na [kg]


# ---------------------------------------------------------------------------
# Modified polylogarithms: g̃ (tilde) and ḡ (bar)
#
# Defined in eqs. 35–36 of the ArXiv paper.  These are the building blocks
# of the quantum recurrence relations (eqs. 32–33) for truncated N and E.
# ---------------------------------------------------------------------------

def g_tilde(s, alpha, sigma, sign):
    """
    Compute the modified polylogarithm g̃_s^(±)(α, σ) from eq. 36.

        g̃_s^(±)(α, σ) = Σ_{j=1}^{∞} (±1)^{j+1} · e^{jα} / j^s · erf(√(jσ))

    The summation index j appears *inside* the erf argument, so this
    function cannot be reduced to a single standard polylogarithm call.
    Uses mpmath.nsum with Euler-Maclaurin + Richardson extrapolation
    for automatic convergence, which is critical near degeneracy
    (α → 0⁻) where the series converges slowly.

    Parameters
    ----------
    s : float
        Polylogarithm order (e.g. 3/2, 5/2, 3, 9/2, 11/2).
    alpha : float or mpf
        Reduced chemical potential  α = μ / (kB T).  Must be < 0.
    sigma : float or mpf
        Reduced cut-off energy  σ = η_c = Q_c / T.
    sign : int
        +1 for bosons (BE), -1 for fermions (FD).

    Returns
    -------
    mpf
        Value of g̃_s^(±)(α, σ).
    """
    alpha = mp.mpf(alpha)
    sigma = mp.mpf(sigma)

    def term(j):
        return (sign ** (j + 1)) * mp.exp(j * alpha) / mp.power(j, s) * mp.erf(mp.sqrt(j * sigma))

    return mp.nsum(term, [1, mp.inf])


def g_bar(s, alpha, sigma, sign):
    """
    Compute the modified polylogarithm ḡ_s^(±)(α, σ) from eq. 35.

        ḡ_s^(±)(α, σ) = Σ_{j=1}^{∞} (±1)^{j+1} · e^{j(α−σ)} / j^s
                       = Li_s(± e^{α−σ})       [standard polylogarithm]

    Unlike g̃, the bar function has no j-dependent coupling in its terms,
    so it reduces to a standard polylogarithm with a shifted argument.

    Parameters
    ----------
    s : float
        Polylogarithm order.
    alpha : float or mpf
        Reduced chemical potential  α = μ / (kB T).
    sigma : float or mpf
        Reduced cut-off energy  σ = η_c.
    sign : int
        +1 for bosons (BE), -1 for fermions (FD).

    Returns
    -------
    mpf
        Value of ḡ_s^(±)(α, σ).
    """
    z_shifted = sign * mp.exp(mp.mpf(alpha) - mp.mpf(sigma))
    result = mp.polylog(s, z_shifted)
    if sign == -1:
        return -result
    return result


# ---------------------------------------------------------------------------
# Newton-Raphson solvers
# ---------------------------------------------------------------------------
def newton_raphson_1var(func, dfunc, x_lo, x_hi, dx, tol=1e-7,
                        max_iter=10000, precision=20):
    """
    Find a root of *func* in [x_lo, x_hi] using bracketing + Newton-Raphson.

    The interval is scanned in steps of *dx*; the first sign change triggers
    Newton-Raphson refinement from that point.

    Parameters
    ----------
    func : callable
        Scalar function whose root is sought.
    dfunc : callable
        Derivative of *func*.
    x_lo, x_hi : float
        Bracketing window.
    dx : float
        Scanning step size.
    tol : float
        Convergence tolerance on successive iterates.
    max_iter : int
        Maximum Newton-Raphson iterations.
    precision : int
        Number of significant digits kept via mpmath.

    Returns
    -------
    float
        Approximate root, or None if no sign change is found.
    """
    x = x_lo
    while x < x_hi:
        if func(x) * func(x + dx) < 0:
            x_n = x
            for _ in range(max_iter):
                x_prev = x_n
                x_n = x_n - func(x_n) / dfunc(x_n)
                if abs(x_n - x_prev) < tol:
                    break
            nstr_result = mp.nstr(x_n, precision)
            if nstr_result is None:
                raise ValueError("mpmath.nstr returned None unexpectedly")
            return float(nstr_result)
        x += dx
    return None


def newton_raphson_2var_fused(jacobian_func, T_init, mu_init, dT, dmu):
    """
    One step of the 2-variable Newton-Raphson method using a fused Jacobian.

    Unlike the non-fused variant, *jacobian_func* computes all needed polylogs
    **once** per (T, μ) evaluation and returns all six quantities together.
    This eliminates redundant polylog calls across f, g, and their partials.

    Parameters
    ----------
    jacobian_func : callable(T, mu) -> tuple of 6 mpf
        Must return (f_val, g_val, f_x_val, f_y_val, g_x_val, g_y_val)
        where f and g are the two equations, and the rest are their partial
        derivatives with respect to T (x) and μ (y).
    T_init, mu_init : float
        Current guess for temperature and chemical potential.
    dT, dmu : float
        Initial perturbation offsets applied before the NR step.

    Returns
    -------
    list of float
        [T_new, mu_new] after one NR correction.
    """
    T_n = T_init + dT
    mu_n = mu_init - dmu

    f_val, g_val, f_x_val, f_y_val, g_x_val, g_y_val = jacobian_func(T_n, mu_n)

    det = f_x_val * g_y_val - f_y_val * g_x_val

    T_new = T_n + (f_y_val * g_val - f_val * g_y_val) / det
    mu_new = mu_n + (f_val * g_x_val - f_x_val * g_val) / det

    return [T_new, mu_new]


def newton_raphson_2var_fused_real(jacobian_func, T_init, mu_init,
                                   dT, dmu, precision=50):
    """
    Same as newton_raphson_2var_fused but returns real parts with specified
    precision.  Useful when mpmath polylog evaluations introduce negligible
    imaginary parts (e.g. quadrupole trap).
    """
    result = newton_raphson_2var_fused(jacobian_func, T_init, mu_init, dT, dmu)
    nstr_T = mp.nstr(mp.re(result[0]), precision)
    nstr_mu = mp.nstr(mp.re(result[1]), precision)
    if nstr_T is None or nstr_mu is None:
        raise ValueError("mpmath.nstr returned None unexpectedly")
    return [float(nstr_T), float(nstr_mu)]


# ---------------------------------------------------------------------------
# Maxwell-Boltzmann functions (potential-independent classical limit)
# ---------------------------------------------------------------------------
def mb_particle_number(N0, Q_cutoff, T):
    """
    Remaining particle number after evaporation cut at Q_cutoff (classical MB).

    Parameters
    ----------
    N0 : float
        Number of particles before cut.
    Q_cutoff : float
        Cut-off temperature [K].
    T : float
        Sample temperature [K].

    Returns
    -------
    float
        Number of particles after evaporation cut.
    """
    eta = np.sqrt(Q_cutoff / T)
    return N0 * (ss.erf(eta) - (2 / np.sqrt(np.pi)) * eta * np.exp(-Q_cutoff / T))


def mb_temperature_box(Q_cutoff, T):
    """
    New sample temperature after MB evaporation cut (3D box, d/2 = 3/2).

    Uses the energy ratio with the factor 4/(3√π) for the box geometry.
    """
    eta = np.sqrt(Q_cutoff / T)
    exp_term = np.exp(-Q_cutoff / T)
    erf_term = ss.erf(eta)
    c1 = 2 / np.sqrt(np.pi)
    c2 = 4 / (3 * np.sqrt(np.pi))

    num = erf_term - c1 * eta * exp_term - c2 * (Q_cutoff / T)**1.5 * exp_term
    den = erf_term - c1 * eta * exp_term
    return T * (num / den)


def mb_temperature_quadrupole(Q_cutoff, T):
    """
    New sample temperature after MB evaporation cut (quadrupole, d/2 = 9/2).

    Uses the energy ratio with the factor 4/(9√π) for the quadrupole geometry.
    """
    eta = np.sqrt(Q_cutoff / T)
    exp_term = np.exp(-Q_cutoff / T)
    erf_term = ss.erf(eta)
    c1 = 2 / np.sqrt(np.pi)
    c2 = 4 / (9 * np.sqrt(np.pi))

    num = erf_term - c1 * eta * exp_term - c2 * (Q_cutoff / T)**1.5 * exp_term
    den = erf_term - c1 * eta * exp_term
    return T * (num / den)


def mb_temperature_oscillator(Q_cutoff, T):
    """
    New sample temperature after MB evaporation cut (harmonic oscillator, d/2 = 3).

    Uses the energy ratio with the factor 4/(6√π) = 2/(3√π) for the oscillator.
    """
    eta = np.sqrt(Q_cutoff / T)
    exp_term = np.exp(-Q_cutoff / T)
    erf_term = ss.erf(eta)
    c1 = 2 / np.sqrt(np.pi)
    c2 = 4 / (6 * np.sqrt(np.pi))

    num = erf_term - c1 * eta * exp_term - c2 * (Q_cutoff / T)**1.5 * exp_term
    den = erf_term - c1 * eta * exp_term
    return T * (num / den)


# ---------------------------------------------------------------------------
# Data container for evaporation results
# ---------------------------------------------------------------------------
def create_result_dict():
    """Create an empty results dictionary for one particle statistics type."""
    return {'N': [], 'T': [], 'E': [], 'Mu': [], 'Q': [], 'Nf': [], 'Tf': []}


def create_mb_result_dict():
    """Create an empty results dictionary for Maxwell-Boltzmann statistics."""
    return {'N': [], 'T': [], 'Q': [], 'Nf': [], 'Tf': []}


def build_cutoff_schedule(Q0, dQ, n_steps):
    """
    Build a linearly decreasing cut-off temperature schedule.

    Parameters
    ----------
    Q0 : float
        Initial cut-off temperature [K].
    dQ : float
        Step decrement [K].
    n_steps : int
        Number of evaporation steps.

    Returns
    -------
    list of float
        Cut-off temperatures for each step.
    """
    return [Q0 - i * dQ for i in range(1, n_steps + 1)]


def initialize_boson_state(results, N0, T0, mu0, E0):
    """Append initial thermodynamic state to a boson/fermion results dict."""
    results['N'].append(N0)
    results['T'].append(T0)
    results['Mu'].append(mu0)
    results['E'].append(E0)


def initialize_mb_state(results, N0, T0):
    """Append initial thermodynamic state to a Maxwell-Boltzmann results dict."""
    results['N'].append(N0)
    results['T'].append(T0)


# ---------------------------------------------------------------------------
# Evaporation simulation loop
# ---------------------------------------------------------------------------
"""
def run_quantum_evaporation(results, truncated_NE_func, nr_func,
                            N0, n_steps, dT, dmu):
    Run the recursive evaporation protocol for a quantum gas (BE or FD).

    Parameters
    ----------
    results : dict
        Pre-initialized results dictionary with initial state and Q schedule.
    truncated_NE_func : callable(N, T, mu, E, Q) -> (N1, E1)
        Fused truncated particle-number and energy function.  Computing
        both together avoids redundant polylog evaluations that share the
        same fugacity and cut-off arguments.
    nr_func : callable(T, mu, dT, dmu, N_new, E_new) -> [T_new, mu_new]
        Two-variable Newton-Raphson solver for (T, mu).  Should use a
        fused Jacobian that precomputes polylogs once per (T, mu) point.
    N0 : float
        Initial particle number (for computing Nf = N/N0).
    n_steps : int
        Number of evaporation steps.
    dT, dmu : float
        NR perturbation offsets for temperature and chemical potential.
    T0 = results['T'][0]
    for i in range(n_steps):
        N_new, E_new = truncated_NE_func(
            results['N'][i], results['T'][i],
            results['Mu'][i], results['E'][i], results['Q'][i],
        )
        results['N'].append(N_new)
        results['Nf'].append(results['N'][i] / N0)
        results['E'].append(E_new)

        T_mu = nr_func(results['T'][i], results['Mu'][i],
                       dT, dmu, results['N'][i + 1], results['E'][i + 1])
        results['T'].append(T_mu[0])
        results['Tf'].append(results['T'][i] / T0)
        results['Mu'].append(T_mu[1])
"""

def run_quantum_evaporation(results, truncated_NE_func, nr_func,
                            N0, n_steps, dT, dmu,
                            sign=+1, alpha_floor=-1e-3, verbose=True,
                            save_path=None, save_on_halt=True,
                            save_metadata=None):
    """
    ... [existing docstring] ...

    Parameters (added)
    ------------------
    save_path : str or Path, optional
        If given, write a JSON snapshot of `results` to this path when the run
        ends (early halt or completion). Existing file is overwritten.
    save_on_halt : bool
        If True (default), save only on early halt. If False, save always.
    save_metadata : dict, optional
        Physical parameters (V, N0, T0, Q schedule params, ...) to embed
        in the snapshot for later reference.
    """
    T0 = results['T'][0]
    kB = ConstantsSI.kB
    halt_reason = None
    halt_step = n_steps  # assume completion unless we break early

    for i in range(n_steps):
        Ni  = results['N'][i]
        Ti  = results['T'][i]
        Mui = results['Mu'][i]
        Ei  = results['E'][i]
        Qi  = results['Q'][i]

        if sign == +1:
            alpha_i = Mui / (kB * Ti)
            if alpha_i > alpha_floor:
                halt_reason = (f"BEC proximity: alpha = {float(alpha_i):.3e} "
                               f"exceeds floor {alpha_floor:.1e}")
                halt_step = i
                if verbose:
                    print(f"  [halt @ step {i}] {halt_reason}")
                break

        try:
            N_new, E_new = truncated_NE_func(Ni, Ti, Mui, Ei, Qi)
            T_mu = nr_func(Ti, Mui, dT, dmu, N_new, E_new)
            T_new, mu_new = T_mu[0], T_mu[1]

            if not (T_new > 0):
                raise ValueError(f"non-positive or NaN T = {T_new}")
            if sign == +1 and not (mu_new < 0):
                raise ValueError(
                    f"mu = {mu_new} >= 0 (crossed BEC boundary or NaN)"
                )

        except (ZeroDivisionError, ValueError, ArithmeticError, TypeError) as e:
            halt_reason = f"{type(e).__name__}: {e}"
            halt_step = i
            if verbose:
                print(f"  [halt @ step {i}] NR solver failed: {halt_reason}")
            break

        # Commit step.
        results['N'].append(N_new)
        results['Nf'].append(Ni / N0)
        results['E'].append(E_new)
        results['T'].append(T_new)
        results['Tf'].append(Ti / T0)
        results['Mu'].append(mu_new)

    # --- Save snapshot on exit ---------------------------------------------
    halted_early = halt_reason is not None
    if save_path is not None and (halted_early or not save_on_halt):
        meta = dict(save_metadata or {})
        meta.update({
            'sign': sign,
            'alpha_floor': alpha_floor,
            'n_steps_requested': n_steps,
            'n_steps_completed': halt_step,
            'halted_early': halted_early,
            'halt_reason': halt_reason,
        })
        save_results_snapshot(results, save_path, metadata=meta)
        if verbose:
            print(f"  [saved] {halt_step} steps -> {save_path}")

    return halt_step

def run_mb_evaporation(results, mb_n_func, mb_t_func, N0, n_steps):
    """
    Run the evaporation protocol for a classical Maxwell-Boltzmann gas.

    Parameters
    ----------
    results : dict
        Pre-initialized MB results dictionary with initial state and Q schedule.
    mb_n_func : callable(N, Q, T) -> float
        MB particle number after cut.
    mb_t_func : callable(Q, T) -> float
        MB temperature after cut.
    N0 : float
        Initial particle number.
    n_steps : int
        Number of evaporation steps.
    """
    T0 = results['T'][0]
    for i in range(n_steps):
        results['N'].append(mb_n_func(results['N'][i], results['Q'][i],
                                       results['T'][i]))
        results['T'].append(mb_t_func(results['Q'][i], results['T'][i]))
        results['Nf'].append(results['N'][i] / N0)
        results['Tf'].append(results['T'][i] / T0)


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------
PLOT_COLORS = {
    'bosons': 'tab:green',
    'fermions': 'tab:red',
    'mb': 'tab:blue',
}

PLOT_LABELS = {
    'bosons': 'Bose-Einstein',
    'fermions': 'Fermi-Dirac',
    'mb': 'Maxwell-Boltzmann',
}


def plot_combined_overview(results_b, results_f, results_mb, trap_name,
                           n_b=None, n_f=None, n_mb=None):
    """
    Plot a 1×3 combined overview: T vs Q, N vs Q, and N vs T.

    Parameters
    ----------
    results_b, results_f, results_mb : dict
        Results dictionaries for bosons, fermions, and MB.
    trap_name : str
        Name of the trapping potential (for titles).
    n_b, n_f, n_mb : int or None
        Number of data points to plot (defaults to all available).
    """
    n_b = n_b or len(results_b['Tf'])
    n_f = n_f or len(results_f['Tf'])
    n_mb = n_mb or len(results_mb['Tf'])

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # T vs Q
    ax = axes[0]
    ax.scatter(results_mb['Q'][:n_mb], results_mb['Tf'][:n_mb],
               c=PLOT_COLORS['mb'], s=8, label=PLOT_LABELS['mb'])
    ax.scatter(results_b['Q'][:n_b], results_b['Tf'][:n_b],
               c=PLOT_COLORS['bosons'], s=8, label=PLOT_LABELS['bosons'])
    ax.scatter(results_f['Q'][:n_f], results_f['Tf'][:n_f],
               c=PLOT_COLORS['fermions'], s=8, label=PLOT_LABELS['fermions'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Cut-off temperature $Q$ [K]', fontsize=14)
    ax.set_ylabel('$T_i / T_0$', fontsize=14)
    ax.set_title(f'{trap_name}: Temperature vs. Cut-off', fontsize=15)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=12)

    # N vs Q
    ax = axes[1]
    ax.scatter(results_mb['Q'][:n_mb], results_mb['Nf'][:n_mb],
               c=PLOT_COLORS['mb'], s=8, label=PLOT_LABELS['mb'])
    ax.scatter(results_b['Q'][:n_b], results_b['Nf'][:n_b],
               c=PLOT_COLORS['bosons'], s=8, label=PLOT_LABELS['bosons'])
    ax.scatter(results_f['Q'][:n_f], results_f['Nf'][:n_f],
               c=PLOT_COLORS['fermions'], s=8, label=PLOT_LABELS['fermions'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Cut-off temperature $Q$ [K]', fontsize=14)
    ax.set_ylabel('$N_i / N_0$', fontsize=14)
    ax.set_title(f'{trap_name}: Particle fraction vs. Cut-off', fontsize=15)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=12)

    # N vs T
    ax = axes[2]
    ax.scatter(results_mb['Tf'][:n_mb], results_mb['Nf'][:n_mb],
               c=PLOT_COLORS['mb'], s=8, label=PLOT_LABELS['mb'])
    ax.scatter(results_b['Tf'][:n_b], results_b['Nf'][:n_b],
               c=PLOT_COLORS['bosons'], s=8, label=PLOT_LABELS['bosons'])
    ax.scatter(results_f['Tf'][:n_f], results_f['Nf'][:n_f],
               c=PLOT_COLORS['fermions'], s=8, label=PLOT_LABELS['fermions'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$T_i / T_0$', fontsize=14)
    ax.set_ylabel('$N_i / N_0$', fontsize=14)
    ax.set_title(f'{trap_name}: Particle fraction vs. Temperature', fontsize=15)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=12)

    fig.tight_layout()
    return fig


def plot_individual_panels(results_b, results_f, results_mb, trap_name,
                           n_b=None, n_f=None, n_mb=None):
    """
    Plot a 3×3 grid showing T vs Q, N vs Q, and N vs T for each statistics.

    Rows: T-vs-Q, N-vs-Q, N-vs-T.
    Columns: Maxwell-Boltzmann, Bosons, Fermions.
    """
    n_b = n_b or len(results_b.get('Tf', results_b['T']))
    n_f = n_f or len(results_f.get('Tf', results_f['T']))
    n_mb = n_mb or len(results_mb.get('Tf', results_mb['T']))

    datasets = [
        ('Maxwell-Boltzmann', results_mb, n_mb, PLOT_COLORS['mb']),
        ('Bose-Einstein', results_b, n_b, PLOT_COLORS['bosons']),
        ('Fermi-Dirac', results_f, n_f, PLOT_COLORS['fermions']),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))

    for col, (label, data, n_pts, color) in enumerate(datasets):
        # Row 0: T vs Q
        axes[0, col].scatter(data['Q'][:n_pts], data['T'][:n_pts], c=color, s=6)
        axes[0, col].set_xscale('log')
        axes[0, col].set_yscale('log')
        axes[0, col].set_title(f'{label} — {trap_name}: T vs Q', fontsize=11)
        axes[0, col].set_xlabel('Cut-off temperature [K]')
        axes[0, col].set_ylabel('Sample temperature [K]')

        # Row 1: N vs Q
        axes[1, col].scatter(data['Q'][:n_pts], data['N'][:n_pts], c=color, s=6)
        axes[1, col].set_title(f'{label} — {trap_name}: N vs Q', fontsize=11)
        axes[1, col].set_xlabel('Cut-off temperature [K]')
        axes[1, col].set_ylabel('Number of atoms')

        # Row 2: N vs T
        axes[2, col].scatter(data['T'][:n_pts], data['N'][:n_pts], c=color, s=6)
        axes[2, col].set_title(f'{label} — {trap_name}: N vs T', fontsize=11)
        axes[2, col].set_xlabel('Sample temperature [K]')
        axes[2, col].set_ylabel('Number of atoms')

    fig.tight_layout()
    return fig
