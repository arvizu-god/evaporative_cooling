"""3D box trap.

Density of states  g(eps) ~ eps^(1/2),  so  s = 3/2.
Equilibrium:
    N    = (V / lambda^3) * g_{3/2}(alpha)
    E    = (3/2) * N * kB * T * g_{5/2}(alpha) / g_{3/2}(alpha)

The truncation-step recurrences (one evaporation cut) are inherited from
the Trap base class via `truncated_NEO`, which consumes the recurrence
specifications produced by `pure_geometry_recurrences(1.5)` in __init__.
This file therefore contains only equilibrium thermodynamics and the
trap-specific Jacobian / MB temperature formulas.
"""

from dataclasses import dataclass

import mpmath as mp

from ..constants import ConstantsSI
from ..polylog import g_full
from ..recurrences import pure_geometry_recurrences
from .base import Trap


@dataclass
class BoxTrap(Trap):
    V: float = 0.0
    h: float = 0.0
    m: float = 0.0

    def __init__(
        self,
        V: float,
        m: float = ConstantsSI.m_Na23,
        h: float = ConstantsSI.h,
        kB: float = ConstantsSI.kB,
    ):
        super().__init__(
            name="box",
            s=1.5,
            recurrences=pure_geometry_recurrences(1.5),
            kB=kB,
        )
        self.V = V
        self.h = h
        self.m = m

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def thermal_wavelength(self, T):
        """De Broglie thermal wavelength  lambda(T) = h / sqrt(2 pi m kB T)."""
        return self.h / mp.sqrt(2 * mp.pi * self.m * self.kB * mp.mpf(T))

    def _prefactor_N(self, T):
        """Coefficient A(T) in  N_eq = A(T) * g_{3/2}(alpha).  Scales as T^{3/2}."""
        return self.V / self.thermal_wavelength(T) ** 3

    # ------------------------------------------------------------------
    # Equilibrium equations of state
    # ------------------------------------------------------------------
    def equilibrium_N(self, T, mu, sign):
        alpha = mp.mpf(mu) / (self.kB * mp.mpf(T))
        return self._prefactor_N(T) * g_full(1.5, alpha, sign)

    def equilibrium_E(self, T, mu, sign):
        alpha = mp.mpf(mu) / (self.kB * mp.mpf(T))
        N = self.equilibrium_N(T, mu, sign)
        return (
            1.5 * N * self.kB * mp.mpf(T)
            * g_full(2.5, alpha, sign) / g_full(1.5, alpha, sign)
        )

    # ------------------------------------------------------------------
    # Fused Jacobian for the (T, mu) Newton-Raphson rethermalization
    #
    # Residuals:
    #   f(T, mu) = N_eq(T, mu) - N_target
    #   g(T, mu) = E_eq(T, mu) - E_target
    #
    # Derivation:
    #   alpha     = mu / (kB T)
    #   d alpha / d T  | mu = -alpha / T
    #   d alpha / d mu | T  =  1 / (kB T)
    #   d g_s / d alpha = g_{s-1}                    (polylog recurrence)
    #   A(T) = V / lambda^3 ~ T^{3/2}, so dA/dT = (3/2) A / T
    #
    # With  u(alpha) := g_{s+1}/g_s  and  u'(alpha) = 1 - g_{s-1} g_{s+1} / g_s^2,
    #   g_val = (3/2) N kB T u
    #   dg/dmu = (3/2) N u'
    #   dg/dT  = (3/2) N kB [u - alpha u']
    # ------------------------------------------------------------------
    def fused_jacobian(self, T, mu, N_target, E_target, sign):
        T = mp.mpf(T)
        mu = mp.mpf(mu)
        alpha = mu / (self.kB * T)

        # Three polylog orders: s-1, s, s+1.  Each evaluated exactly once.
        g12 = g_full(0.5, alpha, sign)
        g32 = g_full(1.5, alpha, sign)
        g52 = g_full(2.5, alpha, sign)

        A = self._prefactor_N(T)               # N_eq = A * g32

        u  = g52 / g32
        up = 1 - g12 * g52 / g32**2

        f_val = A * g32 - N_target
        g_val = 1.5 * N_target * self.kB * T * u - E_target

        f_y_val = (A / (self.kB * T)) * g12
        f_x_val = A * ((1.5 / T) * g32 - (alpha / T) * g12)

        g_y_val = 1.5 * N_target * up
        g_x_val = 1.5 * N_target * self.kB * (u - alpha * up)

        return (f_val, g_val, f_x_val, f_y_val, g_x_val, g_y_val)

    # ------------------------------------------------------------------
    # Maxwell–Boltzmann temperature: d/2 = 3/2  →  factor 4/(3 sqrt(pi))
    # ------------------------------------------------------------------
    #def mb_temperature(self, Q, T):
        #eta = mp.sqrt(Q / T)
        #exp_term = mp.exp(-Q / T)
        #erf_term = ss.erf(float(eta))
        #c1 = 2 / mp.sqrt(mp.pi)
        #c2 = 4 / (3 * mp.sqrt(mp.pi))
        #num = 2 * erf_term - c1 * eta * exp_term - c2 * (Q / T) ** 1.5 * exp_term
        #den = erf_term - c1 * eta * exp_term
        #return float(T * num / den)

    # ------------------------------------------------------------------
    # Storage / debugging
    # ------------------------------------------------------------------
    def describe(self):
        return {**super().describe(), "V": self.V, "m": self.m, "h": self.h}