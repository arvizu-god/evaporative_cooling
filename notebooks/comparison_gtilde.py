"""
Side-by-side comparison of the CORRECT g̃ (mp.nsum) vs the WRONG g̃ (mp.polylog)
as the evaporation simulation progresses step by step.

Simulates a simplified evaporation trajectory for each potential, computing
α = μ/(kBT) and η_c = Q_c/T at each step, then evaluating both formulas.
"""

import mpmath as mp
import math
import numpy as np

mp.dps = 25

# ──────────────────────────────────────────────────────────────────────
# The two implementations
# ──────────────────────────────────────────────────────────────────────

def g_tilde_correct(s, alpha, sigma, sign):
    """CORRECT: mp.nsum with j-dependent erf(√(jσ))."""
    alpha = mp.mpf(alpha)
    sigma = mp.mpf(sigma)
    def term(j):
        return (sign ** (j + 1)) * mp.exp(j * alpha) / mp.power(j, s) * mp.erf(mp.sqrt(j * sigma))
    return mp.nsum(term, [1, mp.inf])


def g_tilde_wrong(s, alpha, sigma, sign):
    """WRONG: erf(√σ) treated as constant, collapsed into polylog argument."""
    alpha = mp.mpf(alpha)
    sigma = mp.mpf(sigma)
    z = sign * mp.exp(alpha) * mp.erf(mp.sqrt(sigma))
    result = mp.polylog(s, z)
    if sign == -1:
        return -result
    return result


# ──────────────────────────────────────────────────────────────────────
# Simulated evaporation trajectories
# ──────────────────────────────────────────────────────────────────────

def simulate_trajectory(alpha_start, Q0, T0, dQ, n_steps):
    """
    Generate a simplified trajectory of (α, η_c) values.

    α starts at alpha_start and drifts toward 0 as T drops.
    η_c = Q_c / T changes as both Q_c decreases and T decreases.
    We use a simple model: T decreases proportionally to Q_c.
    """
    points = []
    T = T0
    for i in range(n_steps):
        Q = Q0 - i * dQ
        if Q <= 0 or T <= 0:
            break
        eta_c = Q / T

        # α drifts toward 0 (less negative) as we cool
        progress = i / n_steps
        alpha = alpha_start * (1.0 - 0.95 * progress)  # approaches ~5% of initial

        points.append((i, alpha, eta_c, T, Q))

        # T drops slightly each step (simplified model)
        T = T * (1 - 0.0003)

    return points


# ──────────────────────────────────────────────────────────────────────
# Run comparisons
# ──────────────────────────────────────────────────────────────────────

configs = [
    {
        'name': '3D Box',
        's_N': 1.5,    # polylog order for particle number
        's_E': 2.5,    # polylog order for energy
        'alpha_start': -15.27,
        'Q0': 5e-4, 'T0': 5e-5, 'dQ': 1e-7,
        'n_steps': 5000,
        'sign': +1,
        'stat': 'Boson',
    },
    {
        'name': '3D Box',
        's_N': 1.5,
        's_E': 2.5,
        'alpha_start': -15.27,
        'Q0': 5e-4, 'T0': 5e-5, 'dQ': 1e-7,
        'n_steps': 5000,
        'sign': -1,
        'stat': 'Fermion',
    },
    {
        'name': 'Harmonic Oscillator',
        's_N': 3.0,
        's_E': 4.0,
        'alpha_start': -11.64,
        'Q0': 5e-4, 'T0': 5e-5, 'dQ': 1e-7,
        'n_steps': 6000,
        'sign': +1,
        'stat': 'Boson',
    },
    {
        'name': 'Harmonic Oscillator',
        's_N': 3.0,
        's_E': 4.0,
        'alpha_start': -11.64,
        'Q0': 5e-4, 'T0': 5e-5, 'dQ': 1e-7,
        'n_steps': 6000,
        'sign': -1,
        'stat': 'Fermion',
    },
    {
        'name': 'Quadrupole',
        's_N': 4.5,
        's_E': 5.5,
        'alpha_start': -3.79,
        'Q0': 5e-4, 'T0': 5e-5, 'dQ': 1e-6,
        'n_steps': 1000,
        'sign': +1,
        'stat': 'Boson',
    },
    {
        'name': 'Quadrupole',
        's_N': 4.5,
        's_E': 5.5,
        'alpha_start': -3.79,
        'Q0': 5e-4, 'T0': 5e-5, 'dQ': 1e-6,
        'n_steps': 1000,
        'sign': -1,
        'stat': 'Fermion',
    },
]

for cfg in configs:
    name = cfg['name']
    stat = cfg['stat']
    sign = cfg['sign']
    s_N = cfg['s_N']

    print("\n" + "=" * 130)
    print(f"  {name} — {stat} (sign={sign:+d})  |  g̃ order s={s_N} (particle number truncation)")
    print("=" * 130)

    trajectory = simulate_trajectory(
        cfg['alpha_start'], cfg['Q0'], cfg['T0'], cfg['dQ'], cfg['n_steps']
    )

    # Sample ~25 points across the trajectory
    n_pts = len(trajectory)
    sample_indices = sorted(set(
        [0, 1, 2, 5, 10] +
        list(range(0, n_pts, max(1, n_pts // 20))) +
        [n_pts - 1]
    ))

    print(f"\n  {'step':>6s}  {'α':>10s}  {'η_c':>10s}  {'g̃ CORRECT':>20s}  {'g̃ WRONG':>20s}  "
          f"{'rel. error':>12s}  {'severity':>10s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*20}  {'─'*20}  {'─'*12}  {'─'*10}")

    for idx in sample_indices:
        if idx >= len(trajectory):
            continue
        step, alpha, eta_c, T, Q = trajectory[idx]

        gt_correct = g_tilde_correct(s_N, alpha, eta_c, sign)
        gt_wrong = g_tilde_wrong(s_N, alpha, eta_c, sign)

        gt_c = float(mp.re(gt_correct))
        gt_w = float(mp.re(gt_wrong))

        if abs(gt_c) > 1e-50:
            rel_err = abs(gt_c - gt_w) / abs(gt_c)
        else:
            rel_err = abs(gt_c - gt_w)

        if rel_err < 1e-6:
            severity = "OK"
        elif rel_err < 1e-2:
            severity = "WARN"
        elif rel_err < 0.1:
            severity = "** BAD **"
        else:
            severity = "*** SEVERE ***"

        print(f"  {step:6d}  {alpha:10.4f}  {eta_c:10.4f}  {gt_c:20.10e}  {gt_w:20.10e}  "
              f"{rel_err:12.4e}  {severity:>14s}")

print("\n" + "=" * 130)
print("  LEGEND:")
print("    α = μ/(kBT)     — reduced chemical potential (→ 0 near degeneracy)")
print("    η_c = Q_c/T     — reduced cut-off energy")
print("    g̃ CORRECT       — mp.nsum implementation with erf(√(j·σ))")
print("    g̃ WRONG         — mp.polylog with [erf(√σ)]^j  (current code)")
print("    rel. error       — |correct - wrong| / |correct|")
print("=" * 130)