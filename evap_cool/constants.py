import numpy as np

class ConstantsEV:
    """Physical constants in eV-based units (h in eV·s, kB in eV/K)."""
    h = 4.135667696e-15       # Planck constant [eV·s]
    hbar = h / (2 * np.pi)    # Reduced Planck constant [eV·s]
    kB = 8.617333262e-5       # Boltzmann constant [eV/K]
    m_Na23 = 3.817545e-26     # Mass of ²³Na [kg]


class ConstantsSI:
    """Physical constants in SI units."""
    h = 6.62607004e-34        # Planck constant [J·s]
    hbar = h / (2 * np.pi)    # Reduced Planck constant [J·s]
    kB = 1.38064852e-23       # Boltzmann constant [J/K]
    m_Na23 = 3.817545e-26     # Mass of ²³Na [kg]

class BoxParameters:
    """Ideal box parameters for evaporative cooling."""
    # Physical constants (SI)
    h  = ConstantsSI.h
    hb = ConstantsSI.hbar
    kB = ConstantsSI.kB
    m  = ConstantsSI.m_Na23
    
    # Trap configurations
    V = 6e-12

    #Initial thermodynamic state
    N = 1e-7
    T = 5e-5

    #Inital cut-off
    Q = 5e-4
    dQ = 5e-8
    N_STEPS = 20000
    dT_nr = 1e-20
    dmu_nr = 1e-30

class QuadrupoleParameters:
    """Ideal quadrupole parameters for evaporative cooling."""
    # Physical constants (SI)
    h  = ConstantsEV.h
    hb = ConstantsEV.hbar
    kB = ConstantsEV.kB
    m  = ConstantsEV.m_Na23
    
    # Trap configurations
    A = 1e-15
    V = 1/A**3

    #Initial thermodynamic state
    N = 1e-7
    T = 5e-5

    #Inital cut-off
    Q = 5e-4
    dQ = 5e-8
    N_STEPS = 20000
    dT_nr = 1e-20
    dmu_nr = 1e-30

class HarmonicParameters:
    """Ideal quadrupole parameters for evaporative cooling."""
    # Physical constants (SI)
    h  = ConstantsEV.h
    hb = ConstantsEV.hbar
    kB = ConstantsEV.kB
    m  = ConstantsEV.m_Na23
    
    # Trap configurations
    Omega = 2*np.pi*100
    V = 1/Omega**3

    #Initial thermodynamic state
    N = 1e-7
    T = 5e-5

    #Inital cut-off
    Q = 5e-4
    dQ = 5e-8
    N_STEPS = 20000
    dT_nr = 1e-20
    dmu_nr = 1e-30