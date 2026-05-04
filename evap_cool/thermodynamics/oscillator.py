"""3D isotropic harmonic oscillator trap.

Potential  U(r) = (1/2) m omega^2 r^2,
density of states  g(eps) ~ eps^2,  so  s = 3.

Equilibrium:
    N = (kB T / (hbar omega))^3 * g_3(alpha)
    E = 3 * N * kB * T * g_4(alpha) / g_3(alpha)

The truncation-step recurrences are inherited from the Trap base class
via `truncated_NEO`, which consumes the specs produced by
`pure_geometry_recurrences(3.0)` in __init__.

Note on the recurrences:
    `pure_geometry_recurrences(3.0)` returns the corrected formula with
    half-integer bar-polylog orders (g_bar_{5/2} for the N term, g_bar_{7/2}
    and g_bar_{5/2} for the E term), per eqs. (48) and (50) of the paper.
    The legacy notebook implementation used integer orders (g_bar_2,
    g_bar_3) which is incorrect — see the refactor regression notes.
"""

from dataclasses import dataclass

import mpmath as mp

from ..constants import ConstantsEV
from ..polylog import g_full
from ..recurrences import pure_geometry_recurrences
from .base import Trap


@dataclass
class OscillatorTrap(Trap):
    omega: float = 0.0
    hbar: float = 0.0
    m: float = 0.0

    def __init__(
        self,
        omega: float,
        m: float = ConstantsEV.m_Na23,
        hbar: float = ConstantsEV.hbar,
        kB: float = ConstantsEV.kB,
    ):
        super().__init__(
            name="oscillator",
            s=3.0,
            recurrences=pure_geometry_recurrences(3.0),
            kB=kB,
        )
        self.omega = omega
        self.hbar = hbar
        self.m = m

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prefactor_N(self, T):
        """Coefficient A(T) in  N_eq = A(T) * g_3(alpha).  Scales as T^3."""
        return (self.kB * mp.mpf(T) / (self.hbar * self.omega)) ** 3

    # ------------------------------------------------------------------
    # Equilibrium equations of state
    # ------------------------------------------------------------------
    def equilibrium_N(self, T, mu, sign):
        alpha = mp.mpf(mu) / (self.kB * mp.mpf(T))
        return self._prefactor_N(T) * g_full(3, alpha, sign)

    def equilibrium_E(self, T, mu, sign):
        alpha = mp.mpf(mu) / (self.kB * mp.mpf(T))
        N = self.equilibrium_N(T, mu, sign)
        return (
            3 * N * self.kB * mp.mpf(T)
            * g_full(4, alpha, sign) / g_full(3, alpha, sign)
        )

    # ------------------------------------------------------------------
    # Fused Jacobian for the (T, mu) Newton-Raphson rethermalization
    #
    # Same structure as box.py with s = 3:
    #   A(T) ~ T^3, so dA/dT = 3 A / T
    #   u(alpha)  = g_4 / g_3
    #   u'(alpha) = 1 - g_2 g_4 / g_3^2
    # ------------------------------------------------------------------
    def fused_jacobian(self, T, mu, N_target, E_target, sign):
        T = mp.mpf(T)
        mu = mp.mpf(mu)
        alpha = mu / (self.kB * T)

        # Three polylog orders: s-1, s, s+1  (i.e. 2, 3, 4)
        g2 = g_full(2, alpha, sign)
        g3 = g_full(3, alpha, sign)
        g4 = g_full(4, alpha, sign)

        A = self._prefactor_N(T)

        u  = g4 / g3
        up = 1 - g2 * g4 / g3**2

        f_val = A * g3 - N_target
        g_val = 3 * N_target * self.kB * T * u - E_target

        f_y_val = (A / (self.kB * T)) * g2
        f_x_val = A * ((3 / T) * g3 - (alpha / T) * g2)

        g_y_val = 3 * N_target * up
        g_x_val = 3 * N_target * self.kB * (u - alpha * up)

        return (f_val, g_val, f_x_val, f_y_val, g_x_val, g_y_val)

    # ------------------------------------------------------------------
    # Maxwell–Boltzmann temperature: d/2 = 3  →  factor 4/(6 sqrt(pi)) = 2/(3 sqrt(pi))
    # ------------------------------------------------------------------
    #def mb_temperature(self, Q, T):
        #eta = mp.sqrt(Q / T)
        #exp_term = mp.exp(-Q / T)
        #erf_term = ss.erf(float(eta))
        #c1 = 2 / mp.sqrt(mp.pi)
        #c2 = 4 / (6 * mp.sqrt(mp.pi))
        #num = (3 / 2) * erf_term - c1 * eta * exp_term - c2 * (Q / T) ** 1.5 * exp_term
        #den = erf_term - c1 * eta * exp_term
        #return float(T * num / den)

    # ------------------------------------------------------------------
    # Storage / debugging
    # ------------------------------------------------------------------
    def describe(self):
        return {**super().describe(),
                "omega": self.omega, "m": self.m, "hbar": self.hbar}