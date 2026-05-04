"""
Trap base class for semiclassical evaporative cooling.

A Trap encodes everything physically distinctive about a trapping
geometry: its density-of-states exponent (via the recurrence specs),
its dimensional parameters (volume, frequencies, gradients), and the
trap-specific portions of the Jacobian and equation-of-state.

Subclasses are expected to:
  - set `recurrences` (typically via `pure_geometry_recurrences(s)`),
  - set `s`  (the density-of-states exponent),
  - set `name` (human-readable identifier, used by storage and plots),
  - implement `equilibrium_N(T, mu, sign)` and `equilibrium_E(T, mu, sign)`,
  - implement `fused_jacobian(T, mu, N_target, E_target, sign)`,
  - implement `mb_temperature(Q, T)`.

The truncated-step recurrence machinery (truncated_NEO) lives on the
base class; it is generic across all pure geometries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import mpmath as mp

from evap_cool.recurrences import Recurrence, evaluate_fused
from evap_cool.solvers import newton_raphson_1var
from .maxwell_boltzmann import mb_temperature as _mb_temperature_kernel

@dataclass
class Trap(ABC):
    """Abstract base class for trapping potentials.

    Attributes
    ----------
    name : str
        Human-readable trap name (e.g. "box", "quadrupole", "oscillator").
    s : float
        Density-of-states exponent. g(eps) ~ eps^(s-1).
    recurrences : dict[str, Recurrence]
        Mapping {"N": ..., "E": ..., "Omega": ...} of recurrence specs.
    kB : float
        Boltzmann constant in the unit system used by this Trap instance.
        Bound at construction so the same Trap class can be used in SI
        or eV unit systems without global state.
    """
    name: str
    s: float
    recurrences: dict[str, Recurrence]
    kB: float

    # ------------------------------------------------------------------
    # Truncation step (generic across all pure geometries)
    # ------------------------------------------------------------------
    def truncated_NEO(
        self,
        N: float,
        T: float,
        mu: float,
        E: float,
        Omega: float,
        Q: float,
        sign: int,
    ) -> tuple[Any, Any, Any]:
        """Apply one evaporation cut: return new (N, E, Omega) before rethermalization.

        Computes the three ratios (N1/N0, E1/E0, Omega1/Omega0) via the
        fused recurrence evaluator (sharing polylog calls across the three
        observables), then multiplies by the current dimensional values.

        Parameters
        ----------
        N, T, mu, E, Omega : float
            Current thermodynamic state (dimensional).
        Q : float
            Cut-off temperature for this step (same units as T).
        sign : int
            +1 for bosons, -1 for fermions.

        Returns
        -------
        (N1, E1, Omega1) : tuple of mpf
            Post-cut, pre-rethermalization values.
        """
        alpha = mp.mpf(mu) / (self.kB * mp.mpf(T))
        eta_c = mp.mpf(Q) / mp.mpf(T)

        ratios = evaluate_fused(self.recurrences, alpha, eta_c, sign)
        return (ratios["N"] * N, ratios["E"] * E, ratios["Omega"] * Omega)

    # ------------------------------------------------------------------
    # Equilibrium equations of state — trap-specific
    # ------------------------------------------------------------------
    @abstractmethod
    def equilibrium_N(self, T: float, mu: float, sign: int) -> Any:
        """Equilibrium particle number  N(T, mu)  for this trap."""

    @abstractmethod
    def equilibrium_E(self, T: float, mu: float, sign: int) -> Any:
        """Equilibrium internal energy  E(T, mu)  for this trap."""

    def equilibrium_Omega(self, T: float, mu: float, sign: int) -> Any:
        """Equilibrium grand potential. Default uses Omega = -E/s.

        This identity holds only at thermodynamic equilibrium for pure-
        geometry traps. Override in subclasses if the relation differs
        (e.g. for mixed geometries).
        """
        return -self.equilibrium_E(T, mu, sign) / self.s

    # ------------------------------------------------------------------
    # Fused Jacobian for the (T, mu) Newton-Raphson rethermalization
    # ------------------------------------------------------------------
    @abstractmethod
    def fused_jacobian(
        self,
        T: float,
        mu: float,
        N_target: float,
        E_target: float,
        sign: int,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return (f, g, df/dT, df/dmu, dg/dT, dg/dmu) for the NR solver.

        f := equilibrium_N(T, mu) - N_target
        g := equilibrium_E(T, mu) - E_target

        Implementations should share polylog evaluations across f, g and
        their partials — a single (T, mu) point should require at most
        one evaluation of g_{s-1}, g_s, and g_{s+1}.

        See `evap_cool.solvers.newton_raphson_2var_fused` for the
        consuming side of this contract.
        """

    # ------------------------------------------------------------------
    # Maxwell-Boltzmann (classical) limit — trap-specific via d/2 factor
    # ------------------------------------------------------------------
    def mb_temperature(self, Q, T):
        """MB-limit post-cut temperature. Default implementation handles all
        pure-geometry traps; mixed traps may override."""
        return _mb_temperature_kernel(self.s, Q, T)

    # ------------------------------------------------------------------
    # Initial-condition helper:  given (N0, T0, sign), find alpha_0
    # ------------------------------------------------------------------
    def solve_alpha0(
        self,
        N0: float,
        T0: float,
        sign: int,
        alpha_lo: float = -20.0,
        alpha_hi: float = -1e-3,
        dx: float = 1e-3,
    ) -> float:
        """Find alpha_0 = mu_0 / (kB T0) such that equilibrium_N matches N0.

        Uses the 1-variable Newton-Raphson with bracketing. Default scan
        window covers strongly non-degenerate (alpha << 0) through near-
        degenerate (alpha -> 0^-) regimes for bosons; fermions can have
        alpha > 0, in which case extend `alpha_hi` at the call site.

        Returns
        -------
        float
            alpha_0. Convert to mu_0 via mu_0 = alpha_0 * kB * T0.
        """
        def f(alpha):
            mu = alpha * self.kB * T0
            return float(mp.re(self.equilibrium_N(T0, mu, sign))) - N0

        def df(alpha):
            # Numerical derivative is fine here; this is a one-time
            # initialization call, not in the inner loop.
            h = max(abs(alpha) * 1e-7, 1e-10)
            return (f(alpha + h) - f(alpha - h)) / (2 * h)

        result = newton_raphson_1var(f, df, alpha_lo, alpha_hi, dx)
        if result is None:
            raise RuntimeError(
                f"Could not solve for alpha_0 with N0={N0}, T0={T0}, sign={sign} "
                f"in window [{alpha_lo}, {alpha_hi}]. Widen the search window."
            )
        return result

    # ------------------------------------------------------------------
    # Repr — useful for storage metadata and debugging
    # ------------------------------------------------------------------
    def describe(self) -> dict:
        """Return a JSON-serializable description for storage metadata."""
        return {
            "name": self.name,
            "s": float(self.s),
            "kB": float(self.kB),
            "trap_class": type(self).__name__,
        }