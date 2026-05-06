"""Equilibrium thermodynamic quantities for pure-geometry traps.

After rethermalization, each (T, μ) pair from the evaporation loop is
an equilibrium grand-canonical state. This module evaluates Ω, S, P
directly from polylog formulas (rather than the algebraic identity
Ω = -E/s) — the two paths agreeing serves as a consistency check on
the Newton-Raphson rethermalization. H, F, G then follow algebraically.

Thermal coefficients (C_V, C_P, κ_T, B_P) require an additional
polylog of order s-1 and are kept in a separate function so callers
who only want state functions don't pay for them.

For a pure-geometry trap with density-of-states exponent s, all
quantities reduce to dimensionless ratios

    r_hi = g_{s+1}(α) / g_s(α),
    r_lo = g_{s-1}(α) / g_s(α),     α = μ / (k_B T)

times constant prefactors (N, k_B T, V_g). The global volume V_g is
constant through the evaporation run, so the trap class supplies it
once via `volume_global`.

Identities used
---------------
State functions (polylog-evaluated):
    Ω = -N k_B T · r_hi
    S = k_B N · [(s+1) r_hi - α]
    P = -Ω / V_g

State functions (algebraic from the above plus E):
    H = E - Ω
    F = Ω + μN
    G = μN          (Euler relation, holds generally)

Thermal coefficients:
    C_V     = s N k_B · [(s+1) r_hi - s/r_lo]
    κ_T     = (V_g / N k_B T) · r_lo
    B_P     = (1/T) · r_lo · [(s+1) r_hi - s/r_lo]
    C_P     = C_V + V_g T B_P² / κ_T

These reduce to the per-trap formulas in `Cuadrupolo_2.pdf` and
`Oscilador.pdf`:
    s = 3   (oscillator):   r_hi = g_4/g_3,        r_lo = g_2/g_3
    s = 9/2 (quadrupole):   r_hi = g_{11/2}/g_{9/2}, r_lo = g_{7/2}/g_{9/2}

The same generic form will apply to the box (s = 3/2) once the box
equilibrium derivation in the source notes is completed.
"""

from __future__ import annotations

import mpmath as mp

from ..polylog import g_full


def equilibrium_state_functions_pure_geometry(
    s, N, T, mu, E, V_g, kB, sign,
):
    """Compute Ω, S, P (via polylogs) and H, F, G (algebraic) at (T, μ).

    Two polylog evaluations: g_s, g_{s+1}.

    Parameters
    ----------
    s : float
        Density-of-states exponent. 1.5 = box, 3 = oscillator,
        4.5 = quadrupole.
    N, T, mu, E : float
        Equilibrium state at this step (from the saved evap run).
    V_g : float
        Trap global volume. Constant for a given trap.
    kB : float
        Boltzmann constant in the trap's unit system.
    sign : int
        +1 bosons, -1 fermions.

    Returns
    -------
    dict
        Keys: Omega, S, P, H, F, G, alpha. Values are mpf.
    """
    T = mp.mpf(T)
    mu = mp.mpf(mu)
    alpha = mu / (kB * T)

    # Two shared polylog evaluations
    g_zero = g_full(s,     alpha, sign)
    g_hi   = g_full(s + 1, alpha, sign)
    r_hi   = g_hi / g_zero

    # Polylog-based (grand-canonical definitions)
    Omega = -N * kB * T * r_hi
    S     = kB * N * ((s + 1) * r_hi - alpha)
    P     = -Omega / V_g

    # Algebraic from Ω, μ, N, E
    G = mu * N
    F = Omega + G          # F = G - PV = G + Ω
    H = E - Omega          # H = E + PV = E - Ω

    return {
        "Omega": Omega,
        "S": S,
        "P": P,
        "H": H,
        "F": F,
        "G": G,
        "alpha": alpha,
    }


def equilibrium_thermal_coefficients_pure_geometry(
    s, N, T, mu, V_g, kB, sign,
):
    """Compute C_V, C_P, κ_T, B_P at (T, μ).

    Three polylog evaluations: g_{s-1}, g_s, g_{s+1}.

    Parameters
    ----------
    s, N, T, mu, V_g, kB, sign
        Same conventions as `equilibrium_state_functions_pure_geometry`.

    Returns
    -------
    dict
        Keys: CV, CP, kappa_T, B_P. Values are mpf.
    """
    T = mp.mpf(T)
    mu = mp.mpf(mu)
    alpha = mu / (kB * T)

    g_lo   = g_full(s - 1, alpha, sign)
    g_zero = g_full(s,     alpha, sign)
    g_hi   = g_full(s + 1, alpha, sign)
    r_lo = g_lo / g_zero
    r_hi = g_hi / g_zero

    # The bracket  (s+1) r_hi - s/r_lo  appears in both C_V and B_P.
    bracket = (s + 1) * r_hi - s / r_lo

    CV      = s * N * kB * bracket
    kappa_T = V_g / (N * kB * T) * r_lo
    B_P     = (1 / T) * r_lo * bracket
    CP      = CV + V_g * T * B_P ** 2 / kappa_T

    return {
        "CV": CV,
        "CP": CP,
        "kappa_T": kappa_T,
        "B_P": B_P,
    }