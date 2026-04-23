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

def g_standard(s, alpha, sign):
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