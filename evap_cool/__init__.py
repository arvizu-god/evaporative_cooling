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
        make_session_dir, save_run, plot_combined_overview,
        process_and_save_run,
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

    # One session folder per script invocation -> no overwriting.
    session = make_session_dir()                       # runs/2026-05-06/14h32m17s/
    save_run(results_b, session / "box_bosons.json",
             trap=trap, parameters={...}, outcome=outcome)

    # Post-process: equilibrium Omega, S, P, H, F, G + thermal coefficients
    process_and_save_run(session / "box_bosons.json", trap, sign=+1)
    # writes -> session / "box_bosons_thermo.json"

Package layout
--------------
    constants        Physical constants (SI, eV unit systems)
    polylog          Modified polylogarithms g_tilde, g_bar, g_full
    solvers          Newton-Raphson root finders
    recurrences      Term-list representation of evaporation recurrences
    thermodynamics   Trap classes, Maxwell-Boltzmann kernel, equilibrium
                     thermodynamics (Omega, S, P, H, F, G + coefficients)
    evaporation      Simulation loops and result containers
    storage          JSON persistence with schema versioning and
                     timestamped session folders
    post_processing  Equilibrium thermodynamics from saved runs (sibling JSON)
    plots            Matplotlib visualizations (evap overview + equilibrium
                     state functions + thermal coefficients)
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
# Persistence (with timestamped session folders)
# ---------------------------------------------------------------------------
from .storage import (
    save_run,
    load_run,
    list_runs,
    list_sessions,
    make_session_dir,
    make_run_filename,
    SCHEMA_VERSION,
)

# ---------------------------------------------------------------------------
# Post-processing (equilibrium thermodynamics from saved runs)
# ---------------------------------------------------------------------------
from .post_processing import (
    compute_run_thermodynamics,
    save_thermodynamics,
    load_thermodynamics,
    process_and_save_run,
    THERMO_SCHEMA_VERSION,
)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
from .plots import (
    plot_combined_overview,
    plot_individual_panels,
    plot_state_functions,
    plot_thermal_coefficients,
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
    "save_run", "load_run", "list_runs", "list_sessions",
    "make_session_dir", "make_run_filename", "SCHEMA_VERSION",
    # Post-processing
    "compute_run_thermodynamics", "save_thermodynamics",
    "load_thermodynamics", "process_and_save_run",
    "THERMO_SCHEMA_VERSION",
    # Plots
    "plot_combined_overview", "plot_individual_panels",
    "plot_state_functions", "plot_thermal_coefficients",
    "align_results",
    "PLOT_COLORS", "PLOT_LABELS",
]