"""Persistence for evaporation simulation results.

Writes a JSON file per run containing three top-level sections:
  - `parameters`: physical inputs (initial state, Q schedule, NR perturbations)
  - `metadata`:   bookkeeping (timestamp, trap descriptor, schema version, halt reason)
  - `results`:    the time-series arrays produced by the simulation loop

mpmath `mpf` values are cast to `float` on write. This is lossy at the
precision used in the polylog inner loop, but the saved time series is for
plotting and downstream analysis, not for resuming computation; callers
who need bit-exact reproducibility should rerun from `parameters`.

This module knows nothing about traps directly: it consumes whatever
`Trap.describe()` returns. That keeps storage decoupled from the
trap-class hierarchy — adding a new trap type doesn't require touching
storage code.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from .evaporation import RunOutcome
from .thermodynamics.base import Trap


SCHEMA_VERSION = 1


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
# Public API
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


def list_runs(directory: Union[str, Path], pattern: str = "*.json") -> list[Path]:
    """Return all JSON files in `directory` matching `pattern`, sorted by name."""
    return sorted(Path(directory).glob(pattern))