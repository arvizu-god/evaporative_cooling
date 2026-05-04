from .polylog import g_tilde, g_bar, g_full
from dataclasses import dataclass, field
from typing import Callable, Literal, Any
import mpmath as mp


@dataclass(frozen=True)
class PolylogTerm:
    """One additive term inside the bracket of a recurrence ratio.

    Represents a quantity of the form
        sign * eta_coeff(eta_c) * G(order, alpha, eta_c, statistics_sign)
    where G is g_tilde if kind == "tilde" or g_bar if kind == "bar".
    """
    kind: Literal["tilde", "bar"]
    order: float                              # polylog order s (can be half-integer)
    eta_coeff: Callable[[Any], Any]     # function of eta_c, returns mpf
    sign: int = +1                            # +1 or -1, the term's sign in the sum


@dataclass(frozen=True)
class Recurrence:
    """A ratio X_1/X_0 = (sum of PolylogTerms) / g_full(denominator_order, alpha)."""
    numerator_terms: tuple[PolylogTerm, ...]
    denominator_order: float


def evaluate_recurrence(
    rec: Recurrence,
    alpha: Any,
    eta_c: Any,
    statistics_sign: int,
) -> Any:
    """Evaluate the ratio X_1/X_0 for a single recurrence.

    Parameters
    ----------
    rec : Recurrence
        Recurrence specification — list of numerator terms plus denominator order.
    alpha : mpf
        Reduced chemical potential alpha = mu / (kB T) at the *current* step.
    eta_c : mpf
        Reduced cut-off  eta_c = Q_c / T  at the *current* step.
    statistics_sign : int
        +1 for bosons, -1 for fermions.

    Returns
    -------
    mpf
        The dimensionless ratio  X_1 / X_0  for this observable.
    """
    alpha = mp.mpf(alpha)
    eta_c = mp.mpf(eta_c)

    numerator = mp.mpf(0)
    for term in rec.numerator_terms:
        if term.kind == "tilde":
            poly_value = g_tilde(term.order, alpha, eta_c, statistics_sign)
        elif term.kind == "bar":
            poly_value = g_bar(term.order, alpha, eta_c, statistics_sign)
        else:
            raise ValueError(f"Unknown polylog kind: {term.kind!r}")
        numerator += term.sign * term.eta_coeff(eta_c) * poly_value

    denominator = g_full(rec.denominator_order, alpha, statistics_sign)
    return numerator / denominator


def pure_geometry_recurrences(s: float) -> dict[str, Recurrence]:
    """Build N, E, Omega recurrences for any pure-geometry trap.

    Pure geometries (box: s=3/2, harmonic: s=3, quadrupole: s=9/2) all share
    the same structural form:
        N/N0   ratio uses orders {s, s-1/2}            and denominator g_s
        E/E0   ratio uses orders {s+1, s+1/2, s-1/2}   and denominator g_{s+1}
        Omega/Omega0 ratio uses {s+1, s+1/2}           and denominator g_{s+1}

    Parameters
    ----------
    s : float
        Trap exponent. Density of states g(eps) ~ eps^(s-1).

    Returns
    -------
    dict
        {"N": Recurrence, "E": Recurrence, "Omega": Recurrence}
    """
    sqrt_pi = mp.sqrt(mp.pi)

    # Common coefficient functions
    one          = lambda eta: mp.mpf(1)
    two_sqrt_eta = lambda eta: 2 * mp.sqrt(eta) / sqrt_pi          # 2/√π · η^(1/2)
    e_second_bd  = lambda eta: 4 * eta**mp.mpf("1.5") / (s * sqrt_pi)  # 4/(s·2·√π) · η^(3/2) → simplifies; see note

    # The E second-boundary coefficient is (2 / (s √π)) η^(3/2). Writing it out:
    e_second_bd  = lambda eta: 2 * eta**mp.mpf("1.5") / (s * sqrt_pi)

    rec_N = Recurrence(
        numerator_terms=(
            PolylogTerm(kind="tilde", order=s,         eta_coeff=one,          sign=+1),
            PolylogTerm(kind="bar",   order=s - 0.5,   eta_coeff=two_sqrt_eta, sign=-1),
        ),
        denominator_order=s,
    )

    rec_E = Recurrence(
        numerator_terms=(
            PolylogTerm(kind="tilde", order=s + 1,     eta_coeff=one,           sign=+1),
            PolylogTerm(kind="bar",   order=s + 0.5,   eta_coeff=two_sqrt_eta,  sign=-1),
            PolylogTerm(kind="bar",   order=s - 0.5,   eta_coeff=e_second_bd,   sign=-1),
        ),
        denominator_order=s + 1,
    )

    rec_Omega = Recurrence(
        numerator_terms=(
            PolylogTerm(kind="tilde", order=s + 1,     eta_coeff=one,          sign=+1),
            PolylogTerm(kind="bar",   order=s + 0.5,   eta_coeff=two_sqrt_eta, sign=-1),
        ),
        denominator_order=s + 1,
    )

    return {"N": rec_N, "E": rec_E, "Omega": rec_Omega}


def evaluate_fused(
    recurrences: dict[str, Recurrence],
    alpha: Any,
    eta_c: Any,
    statistics_sign: int,
) -> dict[str, Any]:
    """Evaluate multiple recurrences with shared polylog computations.

    Identifies polylog evaluations needed by more than one recurrence and
    computes each unique (kind, order) pair only once. For pure-geometry
    traps, this saves one g_bar(s-1/2, ...) call per step (shared between
    N and E) and one g_full(s+1, ...) call (shared between E and Omega).

    Parameters
    ----------
    recurrences : dict
        Mapping like {"N": Recurrence, "E": Recurrence, "Omega": Recurrence}.
        Any subset of keys is allowed.
    alpha, eta_c : mpf
        Current dimensionless state.
    statistics_sign : int
        +1 for bosons, -1 for fermions.

    Returns
    -------
    dict
        Same keys as input, mapped to ratio values (mpf).
    """
    alpha = mp.mpf(alpha)
    eta_c = mp.mpf(eta_c)

    # Cache of (kind, order) -> mpf value, populated lazily.
    cache: dict[tuple[str, float], Any] = {}

    def get_polylog(kind: str, order: float) -> Any:
        key = (kind, order)
        if key not in cache:
            if kind == "tilde":
                cache[key] = g_tilde(order, alpha, eta_c, statistics_sign)
            elif kind == "bar":
                cache[key] = g_bar(order, alpha, eta_c, statistics_sign)
            elif kind == "full":
                cache[key] = g_full(order, alpha, statistics_sign)
            else:
                raise ValueError(f"Unknown polylog kind: {kind!r}")
        return cache[key]

    results: dict[str, Any] = {}
    for name, rec in recurrences.items():
        numerator = mp.mpf(0)
        for term in rec.numerator_terms:
            poly = get_polylog(term.kind, term.order)
            numerator += term.sign * term.eta_coeff(eta_c) * poly
        denominator = get_polylog("full", rec.denominator_order)
        results[name] = numerator / denominator

    return results