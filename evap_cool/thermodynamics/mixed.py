"""Mixed-geometry traps: 2D harmonic oscillator + 1D box, and
1D harmonic oscillator + 2D box.

Reference: Sections 2.4, 2.5 (equations of state) and 3.4, 3.5
(evaporation recurrences) of the semiclassical evaporative-cooling
notes. Both mixed geometries collapse to a *single* power-law density
of states, exactly like the pure traps, so the entire generic machinery
applies with a non-standard exponent nu:

    OscBoxTrap   2D HO + 1D box   nu = 5/2     (DOS  g(eps) ~ eps^{3/2})
    BoxOscTrap   1D HO + 2D box   nu = 2       (DOS  g(eps) ~ eps^{1})

Equations of state (defined in this file)
-----------------------------------------
With  alpha = mu/(kB T)  and thermal wavelength  lambda = h/sqrt(2 pi m kB T):

  OscBoxTrap (2D HO + 1D box), eq. (25):
      N = [ L (2 pi kB T / m) / (omega_x omega_y lambda^3) ] * g_{5/2}(alpha)
      E = (5/2) N kB T * g_{7/2}(alpha) / g_{5/2}(alpha)

  BoxOscTrap (1D HO + 2D box), eqs. (28)-(29):
      N = [ Sigma sqrt(2 pi kB T / m) / (omega_z lambda^3) ] * g_2(alpha)
      E = 2 N kB T * g_3(alpha) / g_2(alpha)

Derivation of the prefactor: the momentum integral contributes the usual
1/lambda^3; each harmonic Gaussian space integral contributes a factor
sqrt(2 pi kB T / m) / (sqrt(j) omega), which raises the polylog order by
1/2 per harmonic dimension (j^{3/2} -> j^{5/2} for two HO dims; -> j^2 for
one). Each box dimension contributes its real-space extent (L, or the
surface area Sigma). Both prefactors are dimensionless in SI.

Everything else is inherited from the generic kernels
-----------------------------------------------------
    Truncation step  (N1, E1, Omega1)  -> pure_geometry_recurrences(nu)
                                          consumed by Trap.truncated_NEO
    Omega, S, P, H, F, G               -> Trap.equilibrium_state_functions
                                          (pure-geometry kernel, exponent nu)
    C_V, C_P, kappa_T, B_P             -> Trap.equilibrium_thermal_coefficients
    Maxwell-Boltzmann (all of it)      -> Trap.mb_* (mb_alpha reuses
                                          _prefactor_N; mb_temperature uses nu)
    Omega = -E/nu                      -> Trap.equilibrium_Omega default
                                          (correct here: single-power DOS)

So this file only defines, per trap: the dimensional prefactor A(T), the
equations of state N(T, mu) and E(T, mu), the fused Jacobian for the
(T, mu) rethermalizer, and the global volume.

Unit system
-----------
SI (ConstantsSI), like BoxTrap: these geometries have real-space box
extents (L, Sigma in metres) and a literal global volume in m^3, so SI
keeps lambda, the prefactors, Omega (J) and P (Pa) numerically consistent.

Global volume V_g -- a caveat
-----------------------------
Unlike the pure traps, V_g here is NOT derived from the geometry; it is a
fixed literal volume supplied at construction (default (100 um)^3 =
1e-12 m^3). V_g enters ONLY the pressure  P = -Omega/V_g  and the
isothermal compressibility  kappa_T = (V_g / N kB T) r_lo. It is constant
through a run (the trap does not change), so it acts as a fixed
multiplicative scale on P and kappa_T and nothing else: the (N, T, mu, E)
trajectory, Omega, S, H, F, G, mu, C_V, C_P and B_P are all independent of
V_g (in C_P the V_g of the correction term cancels against kappa_T). Treat
the mixed-trap P and kappa_T as calibrated only to the chosen V_g.
"""

from dataclasses import dataclass

import mpmath as mp

from ..constants import ConstantsSI
from ..polylog import g_full
from ..recurrences import pure_geometry_recurrences
from .base import Trap


# Default literal global volume: a cube 100 um on a side.
# If you meant 100 cubic micrometres instead, set this to 100 * (1e-6) ** 3.
_DEFAULT_V_G = (100e-6) ** 3      # (100 um)^3 = 1e-12 m^3


@dataclass
class OscBoxTrap(Trap):
    """2D harmonic oscillator (x, y) + 1D box (z).  nu = 5/2."""

    omega_x: float = 0.0
    omega_y: float = 0.0
    L: float = 0.0
    m: float = 0.0
    h: float = 0.0
    V_g: float = _DEFAULT_V_G

    def __init__(
        self,
        omega_x: float,
        omega_y: float,
        L: float,
        V_g: float = _DEFAULT_V_G,
        m: float = ConstantsSI.m_Na23,
        h: float = ConstantsSI.h,
        kB: float = ConstantsSI.kB,
    ):
        super().__init__(
            name="osc2d_box1d",
            s=2.5,
            recurrences=pure_geometry_recurrences(2.5),
            kB=kB,
        )
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.L = L
        self.V_g = V_g
        self.m = m
        self.h = h

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def thermal_wavelength(self, T):
        """De Broglie thermal wavelength  lambda(T) = h / sqrt(2 pi m kB T)."""
        return self.h / mp.sqrt(2 * mp.pi * self.m * self.kB * mp.mpf(T))

    def _prefactor_N(self, T):
        """Coefficient A(T) in  N_eq = A(T) * g_{5/2}(alpha).  Scales as T^{5/2}."""
        lam = self.thermal_wavelength(T)
        return (
            self.L
            * (2 * mp.pi * self.kB * mp.mpf(T) / self.m)
            / (self.omega_x * self.omega_y * lam ** 3)
        )

    # ------------------------------------------------------------------
    # Equilibrium equations of state
    # ------------------------------------------------------------------
    def equilibrium_N(self, T, mu, sign):
        alpha = mp.mpf(mu) / (self.kB * mp.mpf(T))
        return self._prefactor_N(T) * g_full(2.5, alpha, sign)

    def equilibrium_E(self, T, mu, sign):
        alpha = mp.mpf(mu) / (self.kB * mp.mpf(T))
        N = self.equilibrium_N(T, mu, sign)
        return (
            2.5 * N * self.kB * mp.mpf(T)
            * g_full(3.5, alpha, sign) / g_full(2.5, alpha, sign)
        )

    # ------------------------------------------------------------------
    # Fused Jacobian for the (T, mu) Newton-Raphson rethermalization
    #
    # Same structure as box.py / quadrupole.py with s = 5/2:
    #   A(T) ~ T^{5/2}, so dA/dT = (5/2) A / T
    #   u(alpha)  = g_{s+1} / g_s          = g_{7/2} / g_{5/2}
    #   u'(alpha) = 1 - g_{s-1} g_{s+1}/g_s^2  with s-1 = 3/2
    # ------------------------------------------------------------------
    def fused_jacobian(self, T, mu, N_target, E_target, sign):
        T = mp.mpf(T)
        mu = mp.mpf(mu)
        alpha = mu / (self.kB * T)

        # Three polylog orders: s-1, s, s+1  (i.e. 3/2, 5/2, 7/2)
        g_lo  = g_full(1.5, alpha, sign)
        g_mid = g_full(2.5, alpha, sign)
        g_hi  = g_full(3.5, alpha, sign)

        A = self._prefactor_N(T)

        u  = g_hi / g_mid
        up = 1 - g_lo * g_hi / g_mid ** 2

        f_val = A * g_mid - N_target
        g_val = 2.5 * N_target * self.kB * T * u - E_target

        f_y_val = (A / (self.kB * T)) * g_lo
        f_x_val = A * ((2.5 / T) * g_mid - (alpha / T) * g_lo)

        g_y_val = 2.5 * N_target * up
        g_x_val = 2.5 * N_target * self.kB * (u - alpha * up)

        return (f_val, g_val, f_x_val, f_y_val, g_x_val, g_y_val)

    # ------------------------------------------------------------------
    # Storage / debugging
    # ------------------------------------------------------------------
    def describe(self):
        return {
            **super().describe(),
            "omega_x": self.omega_x,
            "omega_y": self.omega_y,
            "L": self.L,
            "V_g": self.V_g,
            "m": self.m,
            "h": self.h,
        }

    @property
    def volume_global(self):
        return self.V_g


@dataclass
class BoxOscTrap(Trap):
    """1D harmonic oscillator (z) + 2D box (surface Sigma).  nu = 2."""

    omega_z: float = 0.0
    Sigma: float = 0.0
    m: float = 0.0
    h: float = 0.0
    V_g: float = _DEFAULT_V_G

    def __init__(
        self,
        omega_z: float,
        Sigma: float,
        V_g: float = _DEFAULT_V_G,
        m: float = ConstantsSI.m_Na23,
        h: float = ConstantsSI.h,
        kB: float = ConstantsSI.kB,
    ):
        super().__init__(
            name="box2d_osc1d",
            s=2.0,
            recurrences=pure_geometry_recurrences(2.0),
            kB=kB,
        )
        self.omega_z = omega_z
        self.Sigma = Sigma
        self.V_g = V_g
        self.m = m
        self.h = h

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def thermal_wavelength(self, T):
        """De Broglie thermal wavelength  lambda(T) = h / sqrt(2 pi m kB T)."""
        return self.h / mp.sqrt(2 * mp.pi * self.m * self.kB * mp.mpf(T))

    def _prefactor_N(self, T):
        """Coefficient A(T) in  N_eq = A(T) * g_2(alpha).  Scales as T^2."""
        lam = self.thermal_wavelength(T)
        return (
            self.Sigma
            * mp.sqrt(2 * mp.pi * self.kB * mp.mpf(T) / self.m)
            / (self.omega_z * lam ** 3)
        )

    # ------------------------------------------------------------------
    # Equilibrium equations of state
    # ------------------------------------------------------------------
    def equilibrium_N(self, T, mu, sign):
        alpha = mp.mpf(mu) / (self.kB * mp.mpf(T))
        return self._prefactor_N(T) * g_full(2.0, alpha, sign)

    def equilibrium_E(self, T, mu, sign):
        alpha = mp.mpf(mu) / (self.kB * mp.mpf(T))
        N = self.equilibrium_N(T, mu, sign)
        return (
            2.0 * N * self.kB * mp.mpf(T)
            * g_full(3.0, alpha, sign) / g_full(2.0, alpha, sign)
        )

    # ------------------------------------------------------------------
    # Fused Jacobian for the (T, mu) Newton-Raphson rethermalization
    #
    # Same structure as box.py / quadrupole.py with s = 2:
    #   A(T) ~ T^2, so dA/dT = 2 A / T
    #   u(alpha)  = g_{s+1} / g_s          = g_3 / g_2
    #   u'(alpha) = 1 - g_{s-1} g_{s+1}/g_s^2  with s-1 = 1
    # ------------------------------------------------------------------
    def fused_jacobian(self, T, mu, N_target, E_target, sign):
        T = mp.mpf(T)
        mu = mp.mpf(mu)
        alpha = mu / (self.kB * T)

        # Three polylog orders: s-1, s, s+1  (i.e. 1, 2, 3)
        g_lo  = g_full(1.0, alpha, sign)
        g_mid = g_full(2.0, alpha, sign)
        g_hi  = g_full(3.0, alpha, sign)

        A = self._prefactor_N(T)

        u  = g_hi / g_mid
        up = 1 - g_lo * g_hi / g_mid ** 2

        f_val = A * g_mid - N_target
        g_val = 2.0 * N_target * self.kB * T * u - E_target

        f_y_val = (A / (self.kB * T)) * g_lo
        f_x_val = A * ((2.0 / T) * g_mid - (alpha / T) * g_lo)

        g_y_val = 2.0 * N_target * up
        g_x_val = 2.0 * N_target * self.kB * (u - alpha * up)

        return (f_val, g_val, f_x_val, f_y_val, g_x_val, g_y_val)

    # ------------------------------------------------------------------
    # Storage / debugging
    # ------------------------------------------------------------------
    def describe(self):
        return {
            **super().describe(),
            "omega_z": self.omega_z,
            "Sigma": self.Sigma,
            "V_g": self.V_g,
            "m": self.m,
            "h": self.h,
        }

    @property
    def volume_global(self):
        return self.V_g