"""Semiclassical evaporative cooling for ideal quantum and classical gases.

Reference:
    Arvizu-Velázquez et al., "Semi-classical evaporative cooling: classical
    and quantum distributions", arXiv (2026).

Quick start
-----------
    from evap_cool import (
        ConstantsSI, BoxTrap,
        create_result_dict, create_mb_result_dict, build_cutoff_schedule,
        initialize_quantum_state, initialize_mb_state,
        run_quantum_evaporation, run_mb_evaporation,
        save_run, plot_combined_overview,
    )

    trap = BoxTrap(V=6e-9)
    Q_schedule = build_cutoff_schedule(Q0=5e-4, dQ=1e-8, n_steps=10000)

    results_b = create_result_dict()
    results_b["Q"] = Q_schedule
    initialize_quantum_state(results_b, N0, T0, mu0_b, E0_b, Omega0_b)

    outcome = run_quantum_evaporation(
        results_b, trap, N0, n_steps=10000,
        dT=T0*1e-12, dmu=abs(mu0_b)*1e-12, sign=+1,
    )

    save_run(results_b, "runs/box_bosons.json",
             trap=trap, parameters={...}, outcome=outcome)

Package layout
--------------
    constants       Physical constants (SI, eV unit systems)
    polylog         Modified polylogarithms g_tilde, g_bar, g_full
    solvers         Newton-Raphson root finders
    recurrences     Term-list representation of evaporation recurrences
    thermodynamics  Trap classes and Maxwell-Boltzmann kernel
    evaporation     Simulation loops and result containers
    storage         JSON persistence with schema versioning
    plots           Matplotlib visualizations
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
from .constants import ConstantsSI, ConstantsEV

# ---------------------------------------------------------------------------
# Mathematical primitives
# ---------------------------------------------------------------------------
from .polylog import g_tilde, g_bar, g_full
from .solvers import (
    newton_raphson_1var,
    newton_raphson_2var_fused,
    newton_raphson_2var_fused_real,
)

# ---------------------------------------------------------------------------
# Recurrence machinery
# ---------------------------------------------------------------------------
from .recurrences import (
    PolylogTerm,
    Recurrence,
    pure_geometry_recurrences,
    evaluate_recurrence,
    evaluate_fused,
)

# ---------------------------------------------------------------------------
# Trap classes and classical limit
# ---------------------------------------------------------------------------
from .thermodynamics import (
    Trap,
    BoxTrap,
    QuadrupoleTrap,
    OscillatorTrap,
    mb_particle_number,
    mb_temperature,
)

# ---------------------------------------------------------------------------
# Simulation loops and result containers
# ---------------------------------------------------------------------------
from .evaporation import (
    RunOutcome,
    create_result_dict,
    create_mb_result_dict,
    build_cutoff_schedule,
    initialize_quantum_state,
    initialize_mb_state,
    run_quantum_evaporation,
    run_mb_evaporation,
)

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
from .storage import save_run, load_run, list_runs, SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
from .plots import (
    plot_combined_overview,
    plot_individual_panels,
    align_results,
    PLOT_COLORS,
    PLOT_LABELS,
)


__all__ = [
    # Version
    "__version__",
    # Constants
    "ConstantsSI", "ConstantsEV",
    # Polylogs
    "g_tilde", "g_bar", "g_full",
    # Solvers
    "newton_raphson_1var",
    "newton_raphson_2var_fused",
    "newton_raphson_2var_fused_real",
    # Recurrences
    "PolylogTerm", "Recurrence",
    "pure_geometry_recurrences",
    "evaluate_recurrence", "evaluate_fused",
    # Traps
    "Trap", "BoxTrap", "QuadrupoleTrap", "OscillatorTrap",
    "mb_particle_number", "mb_temperature",
    # Evaporation
    "RunOutcome",
    "create_result_dict", "create_mb_result_dict",
    "build_cutoff_schedule",
    "initialize_quantum_state", "initialize_mb_state",
    "run_quantum_evaporation", "run_mb_evaporation",
    # Storage
    "save_run", "load_run", "list_runs", "SCHEMA_VERSION",
    # Plots
    "plot_combined_overview", "plot_individual_panels", "align_results",
    "PLOT_COLORS", "PLOT_LABELS",
]