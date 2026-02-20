"""Modular OOP implementation of evaporative-cooling models.

This module rewrites the notebook-style models into reusable classes and provides
an improved Newton-Raphson implementation for 1D and coupled systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import mpmath as mp


@dataclass(frozen=True)
class PhysicalConstants:
    h: float
    kb: float
    m: float

    @property
    def hbar(self) -> float:
        return self.h / (2 * mp.pi)


SI_CONSTANTS = PhysicalConstants(
    h=6.62607015e-34,
    kb=1.380649e-23,
    m=3.817545e-26,
)


class NewtonRaphsonError(RuntimeError):
    pass


class NewtonRaphson1D:
    """Robust Newton-Raphson with optional bracketing and step damping."""

    def __init__(self, tol: float = 1e-10, max_iter: int = 100):
        self.tol = tol
        self.max_iter = max_iter

    def solve(
        self,
        f: Callable[[float], float],
        df: Callable[[float], float],
        x0: float,
        bracket: Optional[Tuple[float, float]] = None,
    ) -> float:
        x = mp.mpf(x0)
        a, b = (None, None) if bracket is None else (mp.mpf(bracket[0]), mp.mpf(bracket[1]))

        for _ in range(self.max_iter):
            fx = f(x)
            if mp.fabs(fx) < self.tol:
                return float(x)

            dfx = df(x)
            if mp.fabs(dfx) < mp.mpf("1e-30"):
                raise NewtonRaphsonError("Derivative too close to zero in 1D Newton step.")

            step = fx / dfx
            x_candidate = x - step

            # If we have a bracket and candidate escapes, damp the step.
            if a is not None and b is not None:
                damp = mp.mpf(1)
                while not (a <= x_candidate <= b) and damp > mp.mpf("1e-6"):
                    damp *= mp.mpf("0.5")
                    x_candidate = x - damp * step
                if not (a <= x_candidate <= b):
                    x_candidate = (a + b) / 2

                # update bracket using sign change
                fa = f(a)
                fc = f(x_candidate)
                if fa * fc <= 0:
                    b = x_candidate
                else:
                    a = x_candidate

            if mp.fabs(x_candidate - x) < self.tol:
                return float(x_candidate)
            x = x_candidate

        raise NewtonRaphsonError("1D Newton-Raphson did not converge.")


class NewtonRaphsonSystem:
    """Damped Newton-Raphson for 2x2 systems with residual-based line search."""

    def __init__(self, tol: float = 1e-10, max_iter: int = 100):
        self.tol = tol
        self.max_iter = max_iter

    def solve(
        self,
        f: Callable[[float, float], float],
        g: Callable[[float, float], float],
        jacobian: Callable[[float, float], Tuple[float, float, float, float]],
        x0: float,
        y0: float,
    ) -> Tuple[float, float]:
        x = mp.mpf(x0)
        y = mp.mpf(y0)

        def residual_norm(xv: float, yv: float) -> mp.mpf:
            return mp.sqrt(f(xv, yv) ** 2 + g(xv, yv) ** 2)

        for _ in range(self.max_iter):
            r0 = residual_norm(x, y)
            if r0 < self.tol:
                return float(x), float(y)

            j11, j12, j21, j22 = jacobian(x, y)
            det = j11 * j22 - j12 * j21
            if mp.fabs(det) < mp.mpf("1e-30"):
                raise NewtonRaphsonError("Jacobian is singular/ill-conditioned.")

            fx = f(x, y)
            gx = g(x, y)

            dx = (j22 * fx - j12 * gx) / det
            dy = (-j21 * fx + j11 * gx) / det

            # Backtracking line-search for stability.
            alpha = mp.mpf(1)
            accepted = False
            while alpha > mp.mpf("1e-6"):
                xn = x - alpha * dx
                yn = y - alpha * dy
                rn = residual_norm(xn, yn)
                if rn < r0:
                    x, y = xn, yn
                    accepted = True
                    break
                alpha *= mp.mpf("0.5")

            if not accepted:
                raise NewtonRaphsonError("Line search failed in system Newton-Raphson.")

            if mp.sqrt((alpha * dx) ** 2 + (alpha * dy) ** 2) < self.tol:
                return float(x), float(y)

        raise NewtonRaphsonError("System Newton-Raphson did not converge.")


@dataclass
class State:
    n_atoms: float
    temperature: float
    chemical_potential: float
    energy: float


class QuantumEvaporationModel:
    """Base class for Boson/Fermion models with polylog-based state equations."""

    def __init__(self, constants: PhysicalConstants, prefactor_c0: float, p_order: float):
        self.c = constants
        self.c0 = mp.mpf(prefactor_c0)
        self.p = mp.mpf(p_order)
        self.nr1d = NewtonRaphson1D()
        self.nr2d = NewtonRaphsonSystem()

    def _li(self, order: float, mu: float, t: float, boson: bool) -> mp.mpf:
        z = mp.e ** (mu / (self.c.kb * t))
        return mp.polylog(order, z if boson else -z)

    def number(self, t: float, mu: float, boson: bool) -> mp.mpf:
        return self.c0 * (t ** self.p) * self._li(self.p, mu, t, boson)

    def energy(self, n_atoms: float, t: float, mu: float, boson: bool) -> mp.mpf:
        ratio = self._li(self.p + 1, mu, t, boson) / self._li(self.p, mu, t, boson)
        return self.p * n_atoms * self.c.kb * t * ratio

    def solve_chemical_potential(
        self,
        n_atoms: float,
        temperature: float,
        boson: bool,
        guess: float,
        bracket: Optional[Tuple[float, float]] = None,
    ) -> float:
        f = lambda x: self.number(temperature, x * self.c.kb * temperature, boson) - n_atoms
        df = lambda x: self.c0 * (temperature ** self.p) * self._li(self.p - 1, x * self.c.kb * temperature, temperature, boson)
        return self.nr1d.solve(f, df, x0=guess, bracket=bracket)

    def solve_state(
        self,
        n_target: float,
        e_target: float,
        boson: bool,
        t_guess: float,
        mu_guess: float,
    ) -> State:
        def f(t: float, mu: float) -> mp.mpf:
            return self.number(t, mu, boson) - n_target

        def g(t: float, mu: float) -> mp.mpf:
            return self.energy(n_target, t, mu, boson) - e_target

        def jac(t: float, mu: float) -> Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
            li_p = self._li(self.p, mu, t, boson)
            li_pm1 = self._li(self.p - 1, mu, t, boson)
            li_pp1 = self._li(self.p + 1, mu, t, boson)

            df_dmu = self.c0 * (t ** (self.p - 1)) * li_pm1 / self.c.kb
            df_dt = self.c0 * (
                self.p * (t ** (self.p - 1)) * li_p
                - (mu / self.c.kb) * (t ** (self.p - 2)) * li_pm1
            )

            ratio = (li_pp1 * li_pm1) / (li_p**2)
            dg_dmu = self.p * n_target * (1 - ratio)
            dg_dt = self.p * n_target * (
                self.c.kb * (li_pp1 / li_p) + (mu / t) * (ratio - 1)
            )
            return df_dt, df_dmu, dg_dt, dg_dmu

        t_sol, mu_sol = self.nr2d.solve(f, g, jac, x0=t_guess, y0=mu_guess)
        e_sol = float(self.energy(n_target, t_sol, mu_sol, boson))
        return State(n_target, t_sol, mu_sol, e_sol)


class HarmonicTrapModel(QuantumEvaporationModel):
    def __init__(self, omega: float, constants: PhysicalConstants = SI_CONSTANTS):
        c0 = (constants.kb**3) / ((constants.hbar**3) * (omega**3))
        super().__init__(constants=constants, prefactor_c0=c0, p_order=3.0)


class BoxTrapModel(QuantumEvaporationModel):
    def __init__(self, volume: float, constants: PhysicalConstants = SI_CONSTANTS):
        c0 = volume * ((2 * mp.pi * constants.m * constants.kb) / (constants.h**2)) ** (3 / 2)
        super().__init__(constants=constants, prefactor_c0=c0, p_order=1.5)


class QuadrupoleTrapModel(QuantumEvaporationModel):
    def __init__(self, volume: float, constants: PhysicalConstants = SI_CONSTANTS):
        c0 = (64 * (constants.kb**6) * (constants.m**3) * (mp.pi**4) * volume) / (constants.h**6)
        super().__init__(constants=constants, prefactor_c0=c0, p_order=4.5)
