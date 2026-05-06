"""Persistence for evaporation simulation results.

Writes a JSON file per run containing three top-level sections:
  - `parameters`: physical inputs (initial state, Q schedule, NR perturbations)
  - `metadata`:   bookkeeping (timestamp, trap descriptor, schema version,
                  outcome)
  - `results`:    the time-series arrays produced by the simulation loop

Session organization
--------------------
To avoid overwriting earlier runs, scripts should call `make_session_dir()`
once at startup and save all their runs into the returned folder. The
on-disk layout is:

    runs/
      2026-05-06/
        14h32m17s/
          box_bosons.json
          box_bosons_thermo.json
          quadrupole_bosons.json
          ...
        16h08m44s/
          ...

Each script invocation gets a fresh timestamped folder, so re-running a
script never clobbers earlier results. Runs from one script execution
share a folder, which makes it easy to compare across traps in a single
session.

mpmath `mpf` values are cast to `float` on write. This is lossy at the
precision used in the polylog inner loop, but the saved time series is
for plotting and downstream analysis, not for resuming computation;
callers who need bit-exact reproducibility should rerun from `parameters`.

This module knows nothing about traps directly: it consumes whatever
`Trap.describe()` returns. That keeps storage decoupled from the
trap-class hierarchy — adding a new trap type doesn't require touching
storage code.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from .evaporation import RunOutcome
from .thermodynamics.base import Trap


SCHEMA_VERSION = 1


# Date / time formats for session folders. Chosen for:
#  - lexicographic sortability (zero-padded);
#  - cross-platform safety (no colons, which Windows rejects in filenames);
#  - human readability.
_DATE_FMT = "%Y-%m-%d"
_TIME_FMT = "%Hh%Mm%Ss"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^\d{2}h\d{2}m\d{2}s")


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------
def _to_float(x: Any) -> Optional[float]:
    """Best-effort cast to a JSON-serializable float.

    mpmath mpf, numpy scalars, and python numerics all survive float().
    Anything that can't be cast becomes None — preserves list lengths so
    plotting code can index into time-series arrays consistently.
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _serialize_results(results: dict) -> dict:
    """Convert a results dict (possibly containing mpf values) for JSON."""
    out = {}
    for key, vals in results.items():
        if isinstance(vals, list):
            out[key] = [_to_float(v) for v in vals]
        else:
            out[key] = vals
    return out


# ---------------------------------------------------------------------------
# Session directories and filename helpers
# ---------------------------------------------------------------------------
def make_session_dir(
    base: Union[str, Path] = "runs",
    when: Optional[datetime] = None,
    label: Optional[str] = None,
) -> Path:
    """Create and return a timestamped session directory.

    Layout:
        <base>/<YYYY-MM-DD>/<HHhMMmSSs>[_<label>]/

    Each script invocation should call this once at startup; all
    `save_run` calls within that script then write to the returned
    directory, so different invocations never overwrite each other.

    Parameters
    ----------
    base : str or Path
        Root directory under which session folders are created.
        Default `"runs"`. Created if missing.
    when : datetime, optional
        Timestamp to use. Defaults to `datetime.now()`. Provided for
        reproducible tests.
    label : str, optional
        Suffix appended to the time component, e.g.
        `"14h32m17s_validation"`. Useful for annotating the purpose of
        a session or disambiguating multiple parallel runs.

    Returns
    -------
    Path
        The created session directory, ready to receive run files.

    Notes
    -----
    On collision (e.g. two scripts launched in the same second), an
    incrementing `_N` suffix is appended so directories are never
    silently shared.
    """
    when = when or datetime.now()
    date_part = when.strftime(_DATE_FMT)
    time_part = when.strftime(_TIME_FMT)
    if label:
        time_part = f"{time_part}_{label}"

    base = Path(base)
    candidate = base / date_part / time_part
    n = 1
    while candidate.exists():
        candidate = base / date_part / f"{time_part}_{n}"
        n += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def make_run_filename(
    trap: Optional[Trap] = None,
    sign: Optional[int] = None,
    *,
    name: Optional[str] = None,
) -> str:
    """Build a conventional run filename.

    By default constructs `<trap_name>_<statistics>.json`, e.g.
    `box_bosons.json`, `quadrupole_fermions.json`. Pass `name` to
    override entirely (e.g. for ad-hoc labels like
    `"box_bosons_highres"`).

    Parameters
    ----------
    trap : Trap, optional
        Trap whose `name` attribute is used. Required if `name` is None.
    sign : int, optional
        +1 -> "bosons", -1 -> "fermions". Required if `name` is None.
    name : str, optional
        Custom filename stem (without extension). If provided, `trap`
        and `sign` are ignored.

    Returns
    -------
    str
        Filename including the `.json` extension.
    """
    if name:
        return f"{name}.json"
    if trap is None or sign is None:
        raise ValueError("Provide either `name`, or both `trap` and `sign`.")
    stat = {+1: "bosons", -1: "fermions"}.get(sign)
    if stat is None:
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    return f"{trap.name}_{stat}.json"


# ---------------------------------------------------------------------------
# Public API: save / load
# ---------------------------------------------------------------------------
def save_run(
    results: dict,
    path: Union[str, Path],
    *,
    trap: Optional[Trap] = None,
    parameters: Optional[dict] = None,
    outcome: Optional[RunOutcome] = None,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Serialize an evaporation run to JSON.

    Safe to call on partial results (after early halt) — only what's
    present in `results` is written.

    Typical usage with session directories:

        session = make_session_dir()
        save_run(results_b, session / "box_bosons.json",
                 trap=trap, parameters={...}, outcome=outcome)
        save_run(results_f, session / "box_fermions.json",
                 trap=trap, parameters={...}, outcome=outcome)

    Parameters
    ----------
    results : dict
        Evaporation results dict (from `create_result_dict` /
        `create_mb_result_dict`, populated by a run).
    path : str or Path
        Output file path. Parent directory is created if missing.
    trap : Trap, optional
        If provided, `trap.describe()` is recorded under
        `metadata.trap` so the run can later be matched to the trap
        configuration that produced it.
    parameters : dict, optional
        Physical parameters of the run (N0, T0, mu0, E0, dT, dmu,
        Q-schedule parameters, statistics sign, etc.). Keep this small
        and structured — these are the inputs needed to reproduce.
    outcome : RunOutcome, optional
        Run-completion summary from `run_quantum_evaporation`. Recorded
        in metadata for diagnostic purposes.
    extra_metadata : dict, optional
        Free-form extra metadata (notes, git SHA, environment info).

    Returns
    -------
    Path
        The path that was written, for convenience in chaining.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if trap is not None:
        metadata["trap"] = trap.describe()
    if outcome is not None:
        metadata["outcome"] = {
            "n_completed": outcome.n_completed,
            "n_steps_requested": outcome.n_steps_requested,
            "halted_early": outcome.halted_early,
            "halt_reason": outcome.halt_reason,
        }
    if extra_metadata:
        metadata.update(extra_metadata)

    payload = {
        "parameters": parameters or {},
        "metadata": metadata,
        "results": _serialize_results(results),
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


def load_run(path: Union[str, Path]) -> dict:
    """Load a run saved by `save_run`.

    Returns
    -------
    dict
        The full payload with keys `parameters`, `metadata`, `results`.
        Schema version is checked against the current `SCHEMA_VERSION`;
        unknown versions raise rather than silently misinterpret.
    """
    path = Path(path)
    with open(path) as f:
        payload = json.load(f)

    schema = payload.get("metadata", {}).get("schema_version")
    if schema is None:
        # Pre-versioned files from early development. Tolerate but warn.
        import warnings
        warnings.warn(
            f"{path}: no schema_version in metadata; treating as legacy. "
            f"Re-save to upgrade.",
            stacklevel=2,
        )
    elif schema != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {schema} not supported by this "
            f"version of evap_cool (expected {SCHEMA_VERSION}). "
            f"Either downgrade evap_cool or migrate the file."
        )

    return payload


# ---------------------------------------------------------------------------
# Listing / discovery
# ---------------------------------------------------------------------------
def list_runs(
    directory: Union[str, Path],
    pattern: str = "*.json",
    *,
    recursive: bool = True,
    include_thermo: bool = True,
) -> list[Path]:
    """Return run files under `directory`, sorted by path.

    Because the session-folder layout is date-then-time, sorting by path
    is also chronological.

    Parameters
    ----------
    directory : str or Path
        Root to search. Pass the `runs/` base to find every run across
        every session, or a specific session folder to list just that
        session.
    pattern : str
        Glob pattern. Default `"*.json"`.
    recursive : bool
        If True (default), walk subdirectories. Required for the
        nested session layout. Set False only if your runs are flat.
    include_thermo : bool
        If False, filter out `*_thermo.json` files. Useful for picking
        out source runs without their post-processed siblings.

    Returns
    -------
    list of Path
        Sorted, with thermo files optionally filtered out.
    """
    directory = Path(directory)
    paths = directory.rglob(pattern) if recursive else directory.glob(pattern)
    if not include_thermo:
        paths = (p for p in paths if not p.stem.endswith("_thermo"))
    return sorted(paths)


def list_sessions(base: Union[str, Path] = "runs") -> list[Path]:
    """Return session directories under `base`, sorted chronologically.

    A session directory is one matching the layout
    `<base>/YYYY-MM-DD/HHhMMmSSs[_label]/` written by `make_session_dir`.
    Other subdirectories (e.g. ad-hoc folders the user created) are
    ignored.

    Parameters
    ----------
    base : str or Path
        Runs root. Default `"runs"`. Returns an empty list if missing.

    Returns
    -------
    list of Path
        Sorted oldest-first by date, then by time.
    """
    base = Path(base)
    if not base.exists():
        return []

    sessions: list[Path] = []
    for date_dir in sorted(base.iterdir()):
        if not (date_dir.is_dir() and _DATE_RE.match(date_dir.name)):
            continue
        for time_dir in sorted(date_dir.iterdir()):
            if time_dir.is_dir() and _TIME_RE.match(time_dir.name):
                sessions.append(time_dir)
    return sessions