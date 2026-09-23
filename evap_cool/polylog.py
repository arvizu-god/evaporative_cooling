import mpmath as mp

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

def g_full(s, alpha, sign):
    """
    Compute the standard (untruncated) polylogarithm g_s^(±)(α).
 
        g_s^(+)(α) = Li_s(  e^α )           [bosons]
        g_s^(-)(α) = -Li_s(-e^α )           [fermions]
 
    Equivalent to ``g_bar(s, alpha, 0, sign)`` but written as a
    dedicated helper for post-processing thermodynamic calculations
    on the *rethermalized* state (no cut-off).  This is the building
    block of Ω, S, E, etc. in the grand-canonical ensemble.
 
    Parameters
    ----------
    s : float
        Polylogarithm order (e.g. 1/2, 3/2, 5/2).
    alpha : float or mpf
        Reduced chemical potential  α = μ / (kB T).
        For bosons α < 0 (α → 0⁻ at the BE condensation transition).
        For fermions α can take either sign (α > 0 in the degenerate regime).
    sign : int
        +1 for bosons (BE), -1 for fermions (FD).
 
    Returns
    -------
    mpf
        Value of g_s^(±)(α).
    """
    alpha = mp.mpf(alpha)
    z = sign * mp.exp(alpha)
    result = mp.polylog(s, z)
    if sign == -1:
        return -result
    return result


# Extra bits g_inc carries above the caller's working precision.
_GUARD_BITS = 16


def g_inc(s, alpha, eta, sign):
    """
    Compute the lower incomplete Bose/Fermi integral g_s^(±)(α; η).

        g_s^(±)(α; η) = (1/Γ(s)) ∫_0^η t^{s−1} / (e^{t−α} ∓ 1) dt
                      = Σ_{j=1}^{∞} (±1)^{j+1} · e^{jα} / j^s · P(s, jη)

    with P(s, x) = γ(s, x) / Γ(s) the regularized lower incomplete gamma
    function.  For a density of states ρ(ε) ∝ ε^{s−1} this is the
    population below the energy cut-off ε_c = η kB T, i.e. the building
    block of the energy-cut truncation (Luiten, Reynolds & Walraven,
    eqs. 25 and 31).  It tends to g_s^(±)(α) as η → ∞.

    Evaluated in the complement form

        g_s^(±)(α; η) = g_s^(±)(α) − Σ_{j=1}^{∞} (±1)^{j+1} · e^{jα} / j^s · Q(s, jη)

    with Q = 1 − P the regularized upper incomplete gamma function.  Since
    Q(s, x) ~ x^{s−1} e^{−x} / Γ(s), the complement terms decay like
    e^{j(α−η)}, so the sum converges geometrically for every α ≤ 0 and for
    fermions with α < η.  The direct P-series converges only as fast as
    the polylogarithm itself, e^{jα} / j^s, and is slow at α → 0⁻, which
    is exactly where the BEC branch needs this function.

    Special regimes:

    * η ≤ 0: nothing lies below the cut-off; returns 0.
    * η − α < 1: the complement ratio e^{α−η} exceeds 1/e.  For fermions
      with α ≥ η the series diverges (Fermi level above the cut-off); for
      bosons with α → 0⁻ and η ≪ 1 the terms stay ~ j^{−s} up to j ~ s/η
      and nsum's extrapolation loses accuracy.  The defining integral is
      evaluated instead by tanh-sinh quadrature after t = u², which makes
      the integrand analytic at u = 0 for integer and half-integer s, with
      a breakpoint at t = |α|.
    * Large η: for s ≥ 1 the complement terms decrease with ratio at most
      e^{α−η} ≤ 1/e, so twice the leading term bounds the whole sum.  Once
      that bound is below the working precision relative to g_s^(±)(α),
      g_s^(±)(α) is returned directly without calling nsum.

    The subtraction cancels about log2(g_s(α) / g_s(α; η)) bits, which is
    large for η ≪ 1.  That loss is bounded a priori through
    g_s^(±)(α; η) ≥ P(s, η) · e^α / (1 + e^α) and the complement form is
    evaluated with that many extra bits, so the result carries the full
    working precision relative to its own size.

    Parameters
    ----------
    s : float
        Polylogarithm order (e.g. 3/2, 2, 5/2, 3, 9/2 and their s + 1).
    alpha : float or mpf
        Reduced chemical potential  α = μ / (kB T).
        For bosons α ≤ 0; for fermions α can take either sign.
    eta : float or mpf
        Reduced cut-off energy  η = Q / T.
    sign : int
        +1 for bosons (BE), -1 for fermions (FD).

    Returns
    -------
    mpf or mpc
        Value of g_s^(±)(α; η).  For fermions with α > 0 the complement
        form inherits g_full's continuation of the polylogarithm through
        z = −1 and may carry a spurious imaginary residue; it is returned
        as-is and callers decide (cf. ``real_part_on_mpc`` in
        ``run_quantum_evaporation``).
    """
    alpha = mp.mpf(alpha)
    eta = mp.mpf(eta)

    if eta <= 0:
        return mp.mpf(0)

    if eta - alpha < 1:
        def integrand(u):
            x = u * u - alpha
            denominator = mp.exp(x) + 1 if sign == -1 else mp.expm1(x)
            return 2 * mp.power(u, 2 * s - 1) / denominator

        with mp.extraprec(_GUARD_BITS):
            if 0 < abs(alpha) < eta:
                points = [0, mp.sqrt(abs(alpha)), mp.sqrt(eta)]
            else:
                points = [0, mp.sqrt(eta)]
            result = mp.quad(integrand, points) / mp.gamma(s)
        return +result

    full = g_full(s, alpha, sign)
    leading = mp.exp(alpha) * mp.gammainc(s, eta, mp.inf, regularized=True)
    if s >= 1 and 2 * leading <= mp.eps * abs(full):
        return full

    lower_bound = (mp.gammainc(s, 0, eta, regularized=True)
                   * mp.exp(alpha) / (1 + mp.exp(alpha)))
    cancelled_bits = max(0, mp.mag(full) - mp.mag(lower_bound))

    def term(j):
        return ((sign ** (j + 1)) * mp.exp(j * alpha) / mp.power(j, s)
                * mp.gammainc(s, j * eta, mp.inf, regularized=True))

    with mp.extraprec(cancelled_bits + _GUARD_BITS):
        result = g_full(s, alpha, sign) - mp.nsum(term, [1, mp.inf])
    return +result
