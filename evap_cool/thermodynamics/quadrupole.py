"""3D quadrupole trap.

Linear potential  U(r) = A_bar * r,
density of states  g(eps) ~ eps^(7/2),  so  s = 9/2.

Equilibrium:
    N = 8 pi (kB T / A_bar)^3 / lambda^3 * g_{9/2}(alpha)
    E = (9/2) * N * kB * T * g_{11/2}(alpha) / g_{9/2}(alpha)

The truncation-step recurrences are inherited from the Trap base class
via `truncated_NEO`, which consumes the specs produced by
`pure_geometry_recurrences(4.5)` in __init__.
"""

from dataclasses import dataclass

import mpmath as mp

from ..constants import ConstantsEV
from ..polylog import g_full
from ..recurrences import pure_geometry_recurrences
from .base import Trap


@dataclass
class QuadrupoleTrap(Trap):
    A_bar: float = 0.0
    h: float = 0.0
    m: float = 0.0

    def __init__(
        self,
        A_bar: float,
        m: float = ConstantsEV.m_Na23,
        h: float = ConstantsEV.h,
        kB: float = ConstantsEV.kB,
    ):
        super().__init__(
            name="quadrupole",
            s=4.5,
            recurrences=pure_geometry_recurrences(4.5),
            kB=kB,
        )
        self.A_bar = A_bar
        self.h = h
        self.m = m

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def thermal_wavelength(self, T):
        return self.h / mp.sqrt(2 * mp.pi * self.m * self.kB * mp.mpf(T))

    def _prefactor_N(self, T):
        """Coefficient A(T) in  N_eq = A(T) * g_{9/2}(alpha).  Scales as T^{9/2}."""
        lam = self.thermal_wavelength(T)
        return 8 * mp.pi * (self.kB * mp.mpf(T) / self.A_bar) ** 3 / lam**3

    # ------------------------------------------------------------------
    # Equilibrium equations of state
    # ------------------------------------------------------------------
    def equilibrium_N(self, T, mu, sign):
        alpha = mp.mpf(mu) / (self.kB * mp.mpf(T))
        return self._prefactor_N(T) * g_full(4.5, alpha, sign)

    def equilibrium_E(self, T, mu, sign):
        alpha = mp.mpf(mu) / (self.kB * mp.mpf(T))
        N = self.equilibrium_N(T, mu, sign)
        return (
            4.5 * N * self.kB * mp.mpf(T)
            * g_full(5.5, alpha, sign) / g_full(4.5, alpha, sign)
        )

    # ------------------------------------------------------------------
    # Fused Jacobian for the (T, mu) Newton-Raphson rethermalization
    #
    # Same structure as box.py with s = 9/2:
    #   A(T) ~ T^{9/2}, so dA/dT = (9/2) A / T
    #   u(alpha)  = g_{s+1} / g_s          = g_{11/2} / g_{9/2}
    #   u'(alpha) = 1 - g_{s-1} g_{s+1}/g_s^2  with s-1 = 7/2
    # ------------------------------------------------------------------
    def fused_jacobian(self, T, mu, N_target, E_target, sign):
        T = mp.mpf(T)
        mu = mp.mpf(mu)
        alpha = mu / (self.kB * T)

        # Three polylog orders: s-1, s, s+1  (i.e. 7/2, 9/2, 11/2)
        g_72  = g_full(3.5, alpha, sign)
        g_92  = g_full(4.5, alpha, sign)
        g_112 = g_full(5.5, alpha, sign)

        A = self._prefactor_N(T)

        u  = g_112 / g_92
        up = 1 - g_72 * g_112 / g_92**2

        f_val = A * g_92 - N_target
        g_val = 4.5 * N_target * self.kB * T * u - E_target

        f_y_val = (A / (self.kB * T)) * g_72
        f_x_val = A * ((4.5 / T) * g_92 - (alpha / T) * g_72)

        g_y_val = 4.5 * N_target * up
        g_x_val = 4.5 * N_target * self.kB * (u - alpha * up)

        return (f_val, g_val, f_x_val, f_y_val, g_x_val, g_y_val)

    # ------------------------------------------------------------------
    # Maxwell–Boltzmann temperature: d/2 = 9/2  →  factor 4/(9 sqrt(pi))
    # ------------------------------------------------------------------
    #def mb_temperature(self, Q, T):
        #eta = mp.sqrt(Q / T)
        #exp_term = mp.exp(-Q / T)
        #erf_term = ss.erf(float(eta))
        #c1 = 2 / mp.sqrt(mp.pi)
        #c2 = 4 / (9 * mp.sqrt(mp.pi))
        #num = (4/3) * erf_term - c1 * eta * exp_term - c2 * (Q / T) ** 1.5 * exp_term
        #den = erf_term - c1 * eta * exp_term
        #return float(T * num / den)

    # ------------------------------------------------------------------
    # Storage / debugging
    # ------------------------------------------------------------------
    def describe(self):
        return {**super().describe(),
                "A_bar": self.A_bar, "m": self.m, "h": self.h}