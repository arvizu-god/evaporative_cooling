"""Post-processing thermodynamics for saved evaporation runs.

A saved quantum run records the equilibrium state (N, T, μ, E) at each
step. This module evaluates state functions (Ω, S, P, H, F, G) and
thermal coefficients (C_V, C_P, κ_T, B_P) at every (T_i, μ_i) using the
trap's equilibrium methods, and writes the result to a sibling JSON
file linked to its source via metadata.

Saved Maxwell-Boltzmann runs (which carry only N and T) are handled by
the parallel pair `compute_mb_run_thermodynamics` /
`process_and_save_mb_run`. In the MB limit µ and E are slaved to (N, T)
via the trap-specific A(T) prefactor and are reconstructed by the trap;
no `sign` is needed.

The thermo file uses the same payload shape as a run file
(`metadata` + `results`) but with its own schema version so the two
formats can evolve independently.

Source linkage
--------------
Each thermo file's `metadata.source_run` block stores both the source
file's path AND a snapshot of its key metadata (saved_at, trap descriptor,
outcome). The path is the primary lookup; the snapshot is a content
fingerprint that lets you re-associate files even after they've been
moved or reorganized.

Quick usage
-----------
    from evap_cool import (
        BoxTrap, make_session_dir,
        compute_run_thermodynamics, save_thermodynamics,
        process_and_save_run,
        process_and_save_mb_run,
    )

    session = make_session_dir()
    trap = BoxTrap(V=6e-12)

    # Quantum run -> thermo (needs sign)
    process_and_save_run(session / "box_bosons.json", trap, sign=+1)
    # writes -> session / "box_bosons_thermo.json"

    # MB run -> thermo (no sign)
    process_and_save_mb_run(session / "box_mb.json", trap)
    # writes -> session / "box_mb_thermo.json"

Notes
-----
* Quantum and MB post-processing are separate entry points. Calling
  `compute_run_thermodynamics` on an MB run raises KeyError pointing
  to the MB function.
* Halted runs are handled transparently — the loop iterates over the
  length of the saved T array, which is whatever was committed before
  the halt.
* Per-step evaluation failures (e.g. polylog blow-up near a degeneracy
  boundary, or N <= 0 in the MB log) are recorded as None rather than
  aborting; this preserves array alignment with the source run.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from .storage import load_run, _to_float, _serialize_results
from .thermodynamics.base import Trap


THERMO_SCHEMA_VERSION = 1

# Required keys in the source run's `results` dict.
_REQUIRED_KEYS = ("N", "T", "Mu", "E")
_MB_REQUIRED_KEYS = ("N", "T")

# Output keys, in the order they're emitted in the JSON.
_OUTPUT_KEYS = (
    "Omega", "S", "P", "H", "F", "G", "alpha",
    "CV", "CP", "kappa_T", "B_P",
)
# MB output adds Mu and E (slaved to (N, T) in the classical limit).
_MB_OUTPUT_KEYS = (
    "Omega", "S", "P", "H", "F", "G", "alpha", "Mu", "E",
    "CV", "CP", "kappa_T", "B_P",
)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def compute_run_thermodynamics(
    source: Union[str, Path, dict],
    trap: Trap,
    sign: int,
    *,
    verbose: bool = False,
) -> dict:
    """Evaluate equilibrium thermodynamics at every step of a saved run.

    Parameters
    ----------
    source : str, Path, or dict
        Path to a JSON run file (loaded via `storage.load_run`) or an
        already-loaded payload dict. Passing the loaded dict avoids
        re-reading the file when chained with `save_thermodynamics`.
    trap : Trap
        Trap instance compatible with the source run. The caller is
        responsible for reconstructing it with the same parameters used
        in the original run — `metadata.trap` in the source file can
        guide this.
    sign : int
        +1 for bosons, -1 for fermions.
    verbose : bool
        If True, print a progress line every 1000 steps.

    Returns
    -------
    dict
        Keys Omega, S, P, H, F, G, alpha, CV, CP, kappa_T, B_P. Each
        value is a list of floats with length equal to the source run's
        T array. Steps where evaluation failed appear as None.

    Raises
    ------
    KeyError
        If the source run is missing required state variables (e.g. a
        Maxwell-Boltzmann run, which has no Mu or E).
    """
    if isinstance(source, (str, Path)):
        payload = load_run(source)
    else:
        payload = source

    results = payload.get("results", {})
    missing = [k for k in _REQUIRED_KEYS if k not in results]
    if missing:
        raise KeyError(
            f"Source run missing required keys {missing}. "
            f"`compute_run_thermodynamics` requires a quantum run with "
            f"N, T, Mu, E populated. For Maxwell-Boltzmann runs (which "
            f"only carry N and T), use `compute_mb_run_thermodynamics` "
            f"instead — µ and E are reconstructed from the trap's A(T)."
        )

    Ns, Ts, Mus, Es = (results[k] for k in _REQUIRED_KEYS)
    n = len(Ts)

    out: dict = {k: [] for k in _OUTPUT_KEYS}

    for i in range(n):
        N, T, mu, E = Ns[i], Ts[i], Mus[i], Es[i]
        try:
            sf = trap.equilibrium_state_functions(N, T, mu, E, sign)
            tc = trap.equilibrium_thermal_coefficients(N, T, mu, sign)
            row = {**sf, **tc}
            for k in _OUTPUT_KEYS:
                out[k].append(_to_float(row[k]))
        except (ZeroDivisionError, ValueError, ArithmeticError, TypeError):
            # Preserve list alignment with the source run on failure.
            for k in _OUTPUT_KEYS:
                out[k].append(None)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  thermo: {i + 1}/{n}")

    return out


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------
def save_thermodynamics(
    thermo: dict,
    source_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    *,
    trap: Optional[Trap] = None,
    sign: Optional[int] = None,
    extra_metadata: Optional[dict] = None,
    source_payload: Optional[dict] = None,
) -> Path:
    """Save post-processed thermodynamics to a sibling JSON file.

    The output sits next to the source run by default, so a session
    folder ends up containing both `box_bosons.json` and
    `box_bosons_thermo.json` side by side.

    Parameters
    ----------
    thermo : dict
        Output of `compute_run_thermodynamics`.
    source_path : str or Path
        Path to the source evap run file.
    out_path : str or Path, optional
        Output path. Defaults to `<source_stem>_thermo.json` next to
        the source file.
    trap : Trap, optional
        If provided, `trap.describe()` is written under `metadata.trap`.
    sign : int, optional
        Recorded under `metadata.sign` for downstream filtering.
    extra_metadata : dict, optional
        Free-form metadata to merge in.
    source_payload : dict, optional
        Already-loaded source run payload. Provided to avoid a second
        disk read when chaining from `compute_run_thermodynamics` /
        `process_and_save_run`. If None, the source is re-loaded so
        its metadata can be snapshotted into the thermo file.

    Returns
    -------
    Path
        The path that was written.
    """
    src = Path(source_path)
    out_path = (
        Path(out_path) if out_path is not None
        else src.with_name(src.stem + "_thermo.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Snapshot the source's metadata so the link survives file moves.
    if source_payload is None:
        try:
            source_payload = load_run(src)
        except (FileNotFoundError, OSError):
            source_payload = {}
    source_meta = source_payload.get("metadata", {}) if source_payload else {}

    metadata: dict = {
        "schema_version": THERMO_SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "source_run": {
            "path": str(src.resolve()),
            "filename": src.name,
            "saved_at": source_meta.get("saved_at"),
            "trap": source_meta.get("trap"),
            "outcome": source_meta.get("outcome"),
        },
    }
    if trap is not None:
        metadata["trap"] = trap.describe()
    if sign is not None:
        metadata["sign"] = sign
    if extra_metadata:
        metadata.update(extra_metadata)

    payload = {
        "metadata": metadata,
        "results": _serialize_results(thermo),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return out_path


def load_thermodynamics(path: Union[str, Path]) -> dict:
    """Load a thermo file written by `save_thermodynamics`.

    Returns
    -------
    dict
        Full payload with `metadata` and `results`. Schema version is
        checked against `THERMO_SCHEMA_VERSION`; mismatches raise rather
        than silently misinterpret.
    """
    path = Path(path)
    with open(path) as f:
        payload = json.load(f)

    schema = payload.get("metadata", {}).get("schema_version")
    if schema != THERMO_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: thermo schema_version {schema} not supported "
            f"by this version of evap_cool (expected {THERMO_SCHEMA_VERSION})."
        )
    return payload


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def process_and_save_run(
    source_path: Union[str, Path],
    trap: Trap,
    sign: int,
    out_path: Optional[Union[str, Path]] = None,
    *,
    verbose: bool = False,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Compute and save in one call, loading the source only once.

    Equivalent to chaining `compute_run_thermodynamics` and
    `save_thermodynamics`. Returns the output path.
    """
    source_payload = load_run(source_path)
    thermo = compute_run_thermodynamics(
        source_payload, trap, sign, verbose=verbose,
    )
    return save_thermodynamics(
        thermo, source_path, out_path,
        trap=trap, sign=sign,
        extra_metadata=extra_metadata,
        source_payload=source_payload,
    )

# ---------------------------------------------------------------------------
# Maxwell-Boltzmann post-processing
# ---------------------------------------------------------------------------
def compute_mb_run_thermodynamics(
    source: Union[str, Path, dict],
    trap: Trap,
    *,
    verbose: bool = False,
) -> dict:
    """Evaluate equilibrium thermodynamics at every step of a saved MB run.

    In the Maxwell-Boltzmann limit µ and E are not independent state
    variables — they are slaved to (N, T) through the trap-specific
    A(T) prefactor (`Trap._prefactor_N`). The source MB run therefore
    only needs to carry N and T; µ and E are reconstructed here and
    returned alongside the state functions.

    Statistics-independent: there is no `sign` argument.

    Parameters
    ----------
    source : str, Path, or dict
        Path to a JSON run file (loaded via `storage.load_run`) or an
        already-loaded payload dict. Passing the loaded dict avoids
        re-reading the file when chained with `save_thermodynamics`.
    trap : Trap
        Trap instance compatible with the source run. The caller is
        responsible for reconstructing it with the same parameters used
        in the original run — `metadata.trap` in the source file can
        guide this.
    verbose : bool
        If True, print a progress line every 1000 steps.

    Returns
    -------
    dict
        Keys Omega, S, P, H, F, G, alpha, Mu, E, CV, CP, kappa_T, B_P.
        Each value is a list of floats with length equal to the source
        run's T array. Steps where evaluation failed appear as None.

    Raises
    ------
    KeyError
        If the source run is missing N or T.
    """
    if isinstance(source, (str, Path)):
        payload = load_run(source)
    else:
        payload = source

    results = payload.get("results", {})
    missing = [k for k in _MB_REQUIRED_KEYS if k not in results]
    if missing:
        raise KeyError(
            f"MB source run missing required keys {missing}. "
            f"`compute_mb_run_thermodynamics` requires a Maxwell-Boltzmann "
            f"run with N and T populated (see `create_mb_result_dict` / "
            f"`run_mb_evaporation`)."
        )

    Ns, Ts = results["N"], results["T"]
    n = len(Ts)

    out: dict = {k: [] for k in _MB_OUTPUT_KEYS}

    for i in range(n):
        N, T = Ns[i], Ts[i]
        try:
            sf = trap.mb_state_functions(N, T)
            tc = trap.mb_thermal_coefficients(N, T)
            row = {**sf, **tc}
            for k in _MB_OUTPUT_KEYS:
                out[k].append(_to_float(row[k]))
        except (ZeroDivisionError, ValueError, ArithmeticError, TypeError):
            # Preserve list alignment with the source run on failure
            # (e.g. N <= 0 from a numerically degenerate step).
            for k in _MB_OUTPUT_KEYS:
                out[k].append(None)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  mb thermo: {i + 1}/{n}")

    return out


def process_and_save_mb_run(
    source_path: Union[str, Path],
    trap: Trap,
    out_path: Optional[Union[str, Path]] = None,
    *,
    verbose: bool = False,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Compute and save MB thermodynamics in one call.

    Equivalent to chaining `compute_mb_run_thermodynamics` and
    `save_thermodynamics`. Loads the source file once. Returns the
    output path.

    The default output is `<source_stem>_thermo.json` next to the
    source — same convention as `process_and_save_run`. `sign` is not
    written into the metadata because MB is statistics-independent.
    """
    source_payload = load_run(source_path)
    thermo = compute_mb_run_thermodynamics(
        source_payload, trap, verbose=verbose,
    )
    return save_thermodynamics(
        thermo, source_path, out_path,
        trap=trap, sign=None,
        extra_metadata=extra_metadata,
        source_payload=source_payload,
    )