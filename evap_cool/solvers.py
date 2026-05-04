import mpmath as mp

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