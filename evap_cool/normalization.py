"""Self-normalization of evaporation runs to the initial (pre-evaporation) state.

A saved run records the equilibrium trajectory during evaporation:
the source run JSON carries the trajectory state variables (`N`, `T`,
and, for quantum runs, `Mu`, `E`), while its `*_thermo.json` sibling
carries the post-processed state functions and coefficients
(`Omega`, `S`, `P`, `H`, `F`, `G`, `alpha`, `C_V`, `C_P`, `kappa_T`,
`B_P`; plus `Mu`, `E` for Maxwell-Boltzmann runs).

This module merges those two files and rescales every quantity by its
*first* element — the value of the gas in its initial state, before any
evaporation step:

    X_hat_i = X_i / X_0

Every series therefore starts at exactly 1 and is dimensionless (a ratio
of like-unit quantities). The independent variable is normalized the same
way, so plots are drawn against ``T_i / T_0``. This replaces the older
physically-motivated normalization (``X / (N k_B T)`` and friends): the
new dimensionless meaning is "fraction of the initial value", not
"departure from a classical limit".

The heat-capacity difference ``C_P - C_V`` is handled specially. To keep
its physical meaning it is normalized as the ratio of the *raw* differences

    (C_P - C_V)_i / (C_P - C_V)_0

rather than the (meaningless) difference of the two normalized series, and
is stored under the key ``CP_minus_CV``.

Output
------
For each source run a sibling ``<stem>_norm.json`` is written next to it,
with the same payload shape as a run/thermo file (`metadata` + `results`)
but its own schema version. The `metadata` block links back to both the
source run and the source thermo file and snapshots the raw initial values
under `metadata.initial_state` so the normalization can be undone.

Quick usage
-----------
    from evap_cool import process_and_save_normalized, normalize_session

    # One run at a time (thermo sibling auto-discovered):
    process_and_save_normalized(session / "box_bosons.json")
    # writes -> session / "box_bosons_norm.json"

    # Or the whole session in one call:
    normalize_session(session)

Notes
-----
* Missing / non-numeric entries propagate as ``None`` (JSON null) so
  partial or halted runs stay length-aligned with their source.
* If the first element of a series is missing, non-finite, or zero, the
  whole normalized series for that key is emitted as ``None`` (we cannot
  define a ratio). Quantities that legitimately cross zero mid-run
  (e.g. ``Mu``) are left noisy by design — only a bad *first* element
  suppresses a series.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from .storage import load_run, list_runs, _to_float
from .post_processing import load_thermodynamics


NORM_SCHEMA_VERSION = 2          # v2 adds the per-particle X/N quantities
_SUPPORTED_NORM_SCHEMAS = (1, 2)

# Trajectory state variables; always taken from the *source run*.
_TRAJECTORY_KEYS = ("N", "T")

# Full set of quantities normalized as X_i / X_0.
_NORMALIZE_KEYS = (
    "N", "T", "Mu", "E",
    "Omega", "S", "P", "H", "F", "G",
    "CV", "CP", "kappa_T", "B_P",
)

# Carried through and normalized as well when present in the source data.
_OPTIONAL_KEYS = ("alpha",)

# Derived quantity: normalized difference of the raw heat capacities.
_DERIVED_KEY = "CP_minus_CV"

# Per-particle quantities (schema v2). Each is the normalized X divided by the
# normalized N, elementwise: X_hat_i / N_hat_i = (X / N)_i / (X / N)_0.
# This is dimensionless, starts at 1, and is the per-particle quantity relative
# to its initial per-particle value. Stored under "<base>_over_N".
_PER_PARTICLE_BASES = ("Omega", "S", "F", "G", "CV", "CP", "CP_minus_CV")
_PER_PARTICLE_KEYS = tuple(f"{b}_over_N" for b in _PER_PARTICLE_BASES)


# ---------------------------------------------------------------------------
# Coercion / arithmetic helpers
# ---------------------------------------------------------------------------
def _finite_or_none(x: Optional[float]) -> Optional[float]:
    """Return a finite float unchanged, mapping None / NaN / inf to None."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if math.isfinite(xf) else None


def _merge_series(run_results: dict, thermo_results: dict, key: str):
    """Pick the source array for `key`.

    `N` and `T` always come from the source run (the evaporation
    trajectory). Everything else prefers the thermo file and falls back
    to the run — which is what routes `Mu`/`E` to the thermo file for
    Maxwell-Boltzmann runs (where the run lacks them) and to the run for
    quantum runs (where the thermo file lacks them). It also ensures
    `Omega` is taken from the post-processed equilibrium value, not the
    value tracked inside the run loop.
    """
    if key in _TRAJECTORY_KEYS:
        return run_results.get(key)
    if thermo_results and key in thermo_results:
        return thermo_results[key]
    return run_results.get(key)


def _normalize_by_first(values) -> Optional[list]:
    """Return ``[v_i / v_0]`` with non-finite entries mapped to None.

    Returns None if `values` is None. If the first element is missing,
    non-finite, or zero, the entire series is returned as ``[None, ...]``
    (a ratio is undefined).
    """
    if values is None:
        return None
    arr = [_finite_or_none(v) for v in values]
    if not arr:
        return []
    x0 = arr[0]
    if x0 is None or x0 == 0.0:
        return [None] * len(arr)
    out = []
    for v in arr:
        if v is None:
            out.append(None)
        else:
            r = v / x0
            out.append(r if math.isfinite(r) else None)
    return out


def _raw_difference(cp, cv) -> Optional[list]:
    """Elementwise raw ``C_P - C_V`` with None propagation."""
    if cp is None or cv is None:
        return None
    n = min(len(cp), len(cv))
    diff = []
    for i in range(n):
        a = _finite_or_none(cp[i])
        b = _finite_or_none(cv[i])
        diff.append(None if (a is None or b is None) else a - b)
    return diff


def _divide_by_normalized_N(values, n_hat) -> Optional[list]:
    """Elementwise ``values_i / n_hat_i`` with None / zero propagation.

    Both arguments are already self-normalized series (``X_i / X_0`` and
    ``N_i / N_0``), so the result is ``(X / N)_i / (X / N)_0``.
    """
    if values is None or n_hat is None:
        return None
    m = min(len(values), len(n_hat))
    out = []
    for i in range(m):
        xi = _finite_or_none(values[i])
        ni = _finite_or_none(n_hat[i])
        if xi is None or ni is None or ni == 0.0:
            out.append(None)
        else:
            r = xi / ni
            out.append(r if math.isfinite(r) else None)
    return out


def _add_per_particle(normalized: dict) -> None:
    """Inject the per-particle ``<base>_over_N`` series (schema v2), in place."""
    n_hat = normalized.get("N")
    for base, key in zip(_PER_PARTICLE_BASES, _PER_PARTICLE_KEYS):
        normalized[key] = _divide_by_normalized_N(normalized.get(base), n_hat)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def _build(run_payload: dict, thermo_payload: Optional[dict]) -> tuple[dict, dict]:
    """Return ``(normalized_results, initial_state)``.

    `normalized_results` maps each quantity key to its ``X_i / X_0``
    series (plus the derived ``CP_minus_CV``). `initial_state` records
    the raw ``X_0`` used as the divisor for each key, for metadata.
    """
    run_results = (run_payload or {}).get("results", {}) or {}
    thermo_results = (thermo_payload or {}).get("results", {}) or {}

    keys = list(_NORMALIZE_KEYS)
    for k in _OPTIONAL_KEYS:
        if k in thermo_results or k in run_results:
            keys.append(k)

    normalized: dict = {}
    initial: dict = {}
    raw: dict = {}

    for key in keys:
        series = _merge_series(run_results, thermo_results, key)
        raw[key] = series
        normalized[key] = _normalize_by_first(series)
        if series:
            initial[key] = _finite_or_none(series[0])

    # Derived C_P - C_V: normalize the raw difference (physical meaning).
    diff = _raw_difference(raw.get("CP"), raw.get("CV"))
    normalized[_DERIVED_KEY] = _normalize_by_first(diff)
    if diff:
        initial[_DERIVED_KEY] = _finite_or_none(diff[0])

    # Per-particle quantities (schema v2): X_hat / N_hat.
    _add_per_particle(normalized)

    return normalized, initial


def compute_normalized_run(
    run_payload: dict,
    thermo_payload: Optional[dict] = None,
) -> dict:
    """Build the normalized `results` dict from loaded run / thermo payloads.

    Parameters
    ----------
    run_payload : dict
        A payload from `storage.load_run` (`parameters`/`metadata`/`results`).
    thermo_payload : dict, optional
        A payload from `post_processing.load_thermodynamics`. If omitted,
        only the quantities present in the run (`N`, `T`, and `Mu`/`E`
        for quantum runs) are normalized; thermo-only quantities are
        emitted as None.

    Returns
    -------
    dict
        Keyed by the normalized quantity names plus ``T`` (the x-axis
        ``T_i / T_0``) and the derived ``CP_minus_CV``.
    """
    normalized, _ = _build(run_payload, thermo_payload)
    return normalized


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------
def _default_thermo_path(run_path: Path) -> Path:
    return run_path.with_name(run_path.stem + "_thermo.json")


def _default_norm_path(run_path: Path) -> Path:
    return run_path.with_name(run_path.stem + "_norm.json")


def save_normalized(
    normalized: dict,
    source_run_path: Union[str, Path],
    *,
    thermo_path: Optional[Union[str, Path]] = None,
    out_path: Optional[Union[str, Path]] = None,
    initial_state: Optional[dict] = None,
    run_payload: Optional[dict] = None,
    thermo_payload: Optional[dict] = None,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Write a normalized run to a sibling ``<stem>_norm.json`` file.

    Mirrors the shape of run / thermo files: a `metadata` block plus a
    `results` block holding the normalized arrays.
    """
    src = Path(source_run_path)
    out = Path(out_path) if out_path is not None else _default_norm_path(src)
    out.parent.mkdir(parents=True, exist_ok=True)

    run_meta = (run_payload or {}).get("metadata", {}) if run_payload else {}
    thermo_meta = (thermo_payload or {}).get("metadata", {}) if thermo_payload else {}

    metadata: dict = {
        "schema_version": NORM_SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "normalized_by": "first_element",  # X_i / X_0
        "x_axis": "T_over_T0",
        "source_run": {
            "path": str(src.resolve()),
            "filename": src.name,
            "saved_at": run_meta.get("saved_at"),
            "trap": run_meta.get("trap"),
            "outcome": run_meta.get("outcome"),
        },
        "source_thermo": (
            {
                "path": str(Path(thermo_path).resolve()),
                "filename": Path(thermo_path).name,
                "sign": thermo_meta.get("sign"),
            }
            if thermo_path is not None
            else None
        ),
    }
    if run_meta.get("trap") is not None:
        metadata["trap"] = run_meta["trap"]
    if thermo_meta.get("sign") is not None:
        metadata["sign"] = thermo_meta["sign"]
    if initial_state is not None:
        metadata["initial_state"] = initial_state
    if extra_metadata:
        metadata.update(extra_metadata)

    payload = {"metadata": metadata, "results": normalized}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def process_and_save_normalized(
    run_path: Union[str, Path],
    thermo_path: Optional[Union[str, Path]] = None,
    out_path: Optional[Union[str, Path]] = None,
    *,
    verbose: bool = False,
) -> Path:
    """Load, normalize, and save in one call.

    The thermo sibling is auto-discovered as ``<run_stem>_thermo.json``
    next to the run unless `thermo_path` is given. A run with no thermo
    sibling is still normalized (its thermo-only quantities come out as
    None).

    Returns the path of the written ``*_norm.json``.
    """
    run_path = Path(run_path)
    run_payload = load_run(run_path)

    thermo_payload = None
    resolved_thermo: Optional[Path] = None
    if thermo_path is not None:
        resolved_thermo = Path(thermo_path)
    else:
        candidate = _default_thermo_path(run_path)
        if candidate.exists():
            resolved_thermo = candidate
    if resolved_thermo is not None and resolved_thermo.exists():
        thermo_payload = load_thermodynamics(resolved_thermo)
    elif verbose:
        print(f"  no thermo sibling for {run_path.name}; "
              f"normalizing run-only quantities")

    normalized, initial = _build(run_payload, thermo_payload)
    out = save_normalized(
        normalized, run_path,
        thermo_path=resolved_thermo, out_path=out_path,
        initial_state=initial,
        run_payload=run_payload, thermo_payload=thermo_payload,
    )
    if verbose:
        print(f"  normalized -> {out.name}")
    return out


def normalize_session(
    session_dir: Union[str, Path],
    *,
    verbose: bool = False,
) -> list[Path]:
    """Normalize every source run in a session folder.

    Pairs each ``<trap>_<stat>.json`` with its ``*_thermo.json`` sibling
    (when present) and writes the corresponding ``*_norm.json``. Thermo
    files and previously-written norm files are skipped as sources.

    Returns the list of written norm-file paths.
    """
    session_dir = Path(session_dir)
    written: list[Path] = []
    for run_path in list_runs(session_dir, include_thermo=False):
        if run_path.stem.endswith("_norm"):
            continue
        try:
            written.append(
                process_and_save_normalized(run_path, verbose=verbose)
            )
        except Exception as exc:  # keep going across a mixed session
            if verbose:
                print(f"  skip {run_path.name}: {exc}")
    return written


def load_normalized(path: Union[str, Path]) -> dict:
    """Load a normalized run written by `save_normalized`.

    Returns the full payload (`metadata` + `results`). The schema version
    is checked against `NORM_SCHEMA_VERSION`.
    """
    path = Path(path)
    with open(path) as f:
        payload = json.load(f)
    schema = payload.get("metadata", {}).get("schema_version")
    if schema not in _SUPPORTED_NORM_SCHEMAS:
        raise ValueError(
            f"{path}: norm schema_version {schema} not supported "
            f"by this version of evap_cool (supported: {_SUPPORTED_NORM_SCHEMAS})."
        )
    return payload