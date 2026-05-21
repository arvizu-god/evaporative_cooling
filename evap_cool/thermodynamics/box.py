"""3D box trap.

Particles of mass m confined in a volume  V = L_x L_y L_z  with
U(r) = 0 inside the box.  Density of states  g(eps) ~ eps^(1/2),
so  s = 3/2.  Reference derivation: `Caja_Ideal.pdf`.

Equations of state (implemented here)
-------------------------------------
With  alpha = mu / (kB T)  and thermal wavelength
lambda_T = h / sqrt(2 pi m kB T):

    N = (V / lambda_T^3) * g_{3/2}(alpha)
    E = (3/2) * N * kB * T * g_{5/2}(alpha) / g_{3/2}(alpha)

Equilibrium state functions  (inherited via Trap → equilibrium.py)
-----------------------------------------------------------------
With  r_hi = g_{5/2}(alpha) / g_{3/2}(alpha):

    Omega = -N kB T * g_{5/2}(alpha) / g_{3/2}(alpha)
    P     = (kB T / lambda_T^3) * g_{5/2}(alpha)        [= -Omega / V]
    S     = N kB * ( (5/2) g_{5/2}(alpha) / g_{3/2}(alpha) - alpha )
    H     = (5/2) N kB T * g_{5/2}(alpha) / g_{3/2}(alpha)
    F     = -N kB T * g_{5/2}(alpha) / g_{3/2}(alpha)  +  N kB T * alpha
    G     = mu * N

Thermal coefficients  (inherited via Trap → equilibrium.py)
-----------------------------------------------------------
    C_V    = N kB * ( (15/4) g_{5/2}(alpha)/g_{3/2}(alpha)
                    - ( 9/4) g_{3/2}(alpha)/g_{1/2}(alpha) )
    kappa_T = V / (N kB T) * g_{1/2}(alpha) / g_{3/2}(alpha)
    B_P    = (1/T) * g_{1/2}(alpha)/g_{3/2}(alpha)
             * ( (5/2) g_{5/2}(alpha)/g_{3/2}(alpha)
               - (3/2) g_{3/2}(alpha)/g_{1/2}(alpha) )
    C_P    = C_V  +  V T B_P^2 / kappa_T
           = (5/2) N kB * ( g_{5/2}(alpha) g_{1/2}(alpha) / g_{3/2}(alpha)^2 )
                       * ( (5/2) g_{5/2}(alpha)/g_{3/2}(alpha)
                         - (3/2) g_{3/2}(alpha)/g_{1/2}(alpha) )

These are *not* re-implemented here: they follow from the pure-geometry
generic formulas in `evap_cool.thermodynamics.equilibrium` with
s = 3/2, V_g = self.V, evaluated at the rethermalized (T, mu).  The
algebraic reduction of those formulas at s = 3/2 reproduces the closed
forms above (and matches `Caja_Ideal.pdf` line-by-line).

Implementation map
------------------
    N, E                          → `equilibrium_N`, `equilibrium_E`  (this file)
    Omega, S, P, H, F, G          → `Trap.equilibrium_state_functions`
                                    → `equilibrium_state_functions_pure_geometry(s=1.5, V_g=V)`
    C_V, C_P, kappa_T, B_P        → `Trap.equilibrium_thermal_coefficients`
                                    → `equilibrium_thermal_coefficients_pure_geometry(s=1.5, V_g=V)`
    Global volume V_g             → `BoxTrap.volume_global = V`
    Truncation-step recurrences   → `pure_geometry_recurrences(1.5)`
                                    consumed by `Trap.truncated_NEO`
    MB-limit post-cut temperature → `Trap.mb_temperature`
                                    via `maxwell_boltzmann.mb_temperature(s=1.5, ...)`

This file therefore contains only:
  - the trap-specific equations of state  N(T, mu), E(T, mu),
  - the fused Jacobian used by the (T, mu) Newton-Raphson rethermalizer,
  - helpers (thermal_wavelength, _prefactor_N) and storage hooks.
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
    
    @property
    def volume_global(self):
        return self.V