"""Evaporation simulation loops for quantum and classical (MB) gases.

This module orchestrates the recursive evaporation protocol. It does not
contain trap-specific physics — that lives in `evap_cool.thermodynamics`.
A simulation is parameterized by a `Trap` instance (which provides the
truncation, equilibrium, and Jacobian machinery) and the statistics sign.

The loops are deliberately decoupled from I/O: they return information
about completion (how many steps ran, why they halted), and the caller
is responsible for persisting results via `evap_cool.storage` if desired.
"""

from dataclasses import dataclass
from typing import Optional

import mpmath as mp

from .solvers import newton_raphson_2var_fused
from .thermodynamics.base import Trap
from .thermodynamics.maxwell_boltzmann import mb_particle_number


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
def create_result_dict() -> dict:
    """Empty results dictionary for one quantum-statistics run.

    Keys
    ----
    N, T, Mu, E, Omega : list
        Dimensional thermodynamic state at each step.
    Q : list
        Cut-off schedule (populated externally before the run starts).
    Nf, Tf : list
        Normalized N / N0 and T / T0 for plotting.
    """
    return {"N": [], "T": [], "Mu": [], "E": [], "Omega": [],
            "Q": [], "Nf": [], "Tf": []}


def create_mb_result_dict() -> dict:
    """Empty results dictionary for a Maxwell-Boltzmann run."""
    return {"N": [], "T": [], "Q": [], "Nf": [], "Tf": []}


def build_cutoff_schedule(Q0: float, dQ: float, n_steps: int) -> list:
    """Linearly decreasing cut-off temperature schedule.

    Returns Q values for steps 1..n_steps (i.e. Q0 - dQ, Q0 - 2*dQ, ...).
    """
    return [Q0 - i * dQ for i in range(1, n_steps + 1)]


def initialize_quantum_state(results: dict, N0, T0, mu0, E0, Omega0) -> None:
    """Append the initial state to a quantum results dict.

    Identical for bosons and fermions — the statistics sign enters only
    through the truncation and equilibrium calculations, not through the
    state vector itself.
    """
    results["N"].append(N0)
    results["T"].append(T0)
    results["Mu"].append(mu0)
    results["E"].append(E0)
    results["Omega"].append(Omega0)


def initialize_mb_state(results: dict, N0, T0) -> None:
    """Append the initial (N0, T0) to a MB results dict."""
    results["N"].append(N0)
    results["T"].append(T0)


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------
@dataclass
class RunOutcome:
    """Why and where a quantum evaporation run ended.

    Attributes
    ----------
    n_completed : int
        Number of evaporation steps successfully committed. Equals
        `n_steps_requested` if the run finished cleanly.
    n_steps_requested : int
        Number of steps the caller asked for.
    halted_early : bool
        True if the loop terminated before completing all requested steps.
    halt_reason : str or None
        Human-readable description of the early halt (None if completed).
    """
    n_completed: int
    n_steps_requested: int
    halted_early: bool
    halt_reason: Optional[str]


# ---------------------------------------------------------------------------
# Quantum evaporation loop
# ---------------------------------------------------------------------------
def run_quantum_evaporation(
    results: dict,
    trap: Trap,
    N0: float,
    n_steps: int,
    dT: float,
    dmu: float,
    sign: int = +1,
    alpha_floor: float = -1e-3,
    verbose: bool = True,
) -> RunOutcome:
    """Run the recursive evaporation protocol for a quantum gas.

    For each step:
      1. Apply the evaporation cut via `trap.truncated_NEO`, producing
         post-cut (N, E, Omega) before rethermalization.
      2. Solve the (T, mu) Newton-Raphson rethermalization using
         `trap.fused_jacobian`, recovering the new equilibrium state.
      3. Append the result and update normalized arrays.

    The loop halts early (returning a `RunOutcome` flagged accordingly) if:
      - For bosons: alpha = mu / (kB T) approaches 0 from below (BEC
        proximity), specifically alpha > `alpha_floor`.
      - The NR solver produces non-positive T, or non-negative mu for
        bosons, or raises an arithmetic exception.

    Parameters
    ----------
    results : dict
        Pre-initialized results dict with the initial state appended and
        the `Q` schedule pre-populated. See `create_result_dict` and
        `initialize_quantum_state`.
    trap : Trap
        Trap object providing `truncated_NEO`, `fused_jacobian`, and `kB`.
    N0 : float
        Initial particle number (used for normalized N/N0 plotting).
    n_steps : int
        Number of evaporation steps to attempt.
    dT, dmu : float
        Initial perturbations passed to the 2-variable Newton-Raphson
        fused solver.
    sign : int
        +1 for bosons, -1 for fermions.
    alpha_floor : float
        Boson-only safety threshold. Loop halts before step `i` if the
        current alpha exceeds this value (i.e. approaches 0 from below).
        Ignored for fermions.
    verbose : bool
        If True, print a message when the loop halts early.

    Returns
    -------
    RunOutcome
        Information about how the run terminated.
    """
    T0 = results["T"][0]
    kB = trap.kB
    halt_reason: Optional[str] = None
    halt_step = n_steps

    # Wrap the trap's fused_jacobian so the NR solver gets the closure
    # signature it expects: (T, mu) -> 6-tuple.
    def _make_jacobian(N_target, E_target):
        def jacobian(T, mu):
            return trap.fused_jacobian(T, mu, N_target, E_target, sign)
        return jacobian

    for i in range(n_steps):
        Ni     = results["N"][i]
        Ti     = results["T"][i]
        Mui    = results["Mu"][i]
        Ei     = results["E"][i]
        Omegai = results["Omega"][i]
        Qi     = results["Q"][i]

        # Boson safety check: halt before stepping if too close to BEC.
        if sign == +1:
            alpha_i = Mui / (kB * Ti)
            if alpha_i > alpha_floor:
                halt_reason = (f"BEC proximity: alpha = {float(alpha_i):.3e} "
                               f"exceeds floor {alpha_floor:.1e}")
                halt_step = i
                break

        try:
            # 1. Truncation step.
            N_new, E_new, Omega_new = trap.truncated_NEO(
                Ni, Ti, Mui, Ei, Omegai, Qi, sign,
            )
            N_new     = float(N_new)
            E_new     = float(E_new)
            Omega_new = float(Omega_new)

            # 2. Rethermalization NR.
            jac = _make_jacobian(N_new, E_new)
            T_mu = newton_raphson_2var_fused(jac, Ti, Mui, dT, dmu)
            T_new = float(T_mu[0])
            mu_new = float(T_mu[1])

            # 3. Validate.
            if not (T_new > 0):
                raise ValueError(f"non-positive or NaN T = {T_new}")
            if sign == +1 and not (mu_new < 0):
                raise ValueError(
                    f"mu = {mu_new} >= 0 (crossed BEC boundary or NaN)"
                )

        except (ZeroDivisionError, ValueError, ArithmeticError, TypeError) as e:
            halt_reason = f"{type(e).__name__}: {e}"
            halt_step = i
            break

        # Commit step.
        results["N"].append(N_new)
        results["E"].append(E_new)
        results["Omega"].append(Omega_new)
        results["T"].append(T_new)
        results["Mu"].append(mu_new)
        results["Nf"].append(Ni / N0)
        results["Tf"].append(Ti / T0)

    halted_early = halt_reason is not None
    if halted_early and verbose:
        print(f"  [halt @ step {halt_step}] {halt_reason}")

    return RunOutcome(
        n_completed=halt_step,
        n_steps_requested=n_steps,
        halted_early=halted_early,
        halt_reason=halt_reason,
    )


# ---------------------------------------------------------------------------
# Maxwell-Boltzmann evaporation loop
# ---------------------------------------------------------------------------
def run_mb_evaporation(
    results: dict,
    trap: Trap,
    N0: float,
    n_steps: int,
    verbose: bool = True,
) -> RunOutcome:
    """Run the classical Maxwell-Boltzmann evaporation protocol.

    Halts early (returning a `RunOutcome` flagged accordingly) if:
      - Q_i becomes non-positive (cut-off out of the physical range), or
      - the MB kernel raises an arithmetic exception, or
      - the new temperature is non-positive / NaN.

    Parameters mirror `run_quantum_evaporation`.
    """
    T0 = results["T"][0]
    halt_reason: Optional[str] = None
    halt_step = n_steps

    for i in range(n_steps):
        Ni = results["N"][i]
        Ti = results["T"][i]
        Qi = results["Q"][i]

        # Physical guard: cut-off temperature must be strictly positive.
        if not (Qi > 0):
            halt_reason = f"non-positive cut-off Q = {Qi:.3e}"
            halt_step = i
            break

        try:
            N_new = mb_particle_number(Ni, Qi, Ti)
            T_new = trap.mb_temperature(Qi, Ti)

            if not (T_new > 0):
                raise ValueError(f"non-positive or NaN T = {T_new}")

        except (ZeroDivisionError, ValueError, ArithmeticError, TypeError) as e:
            halt_reason = f"{type(e).__name__}: {e}"
            halt_step = i
            break

        # Commit step.
        results["N"].append(N_new)
        results["T"].append(T_new)
        results["Nf"].append(Ni / N0)
        results["Tf"].append(Ti / T0)

    halted_early = halt_reason is not None
    if halted_early and verbose:
        print(f"  [halt @ step {halt_step}] {halt_reason}")

    return RunOutcome(
        n_completed=halt_step,
        n_steps_requested=n_steps,
        halted_early=halted_early,
        halt_reason=halt_reason,
    )