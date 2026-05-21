"""Adaptive two-stage zoom near a degeneracy boundary.

Two public entry points share a common dynamic:

* `run_with_bec_zoom` — bosons.  Coarse pass with a loose `alpha_floor`
  locates the BEC cliff; the fine pass re-seeds a few steps before the
  halt with higher mpmath precision, a tighter `alpha_floor`, and a
  finer Q-schedule, allowing the trajectory to approach mu → 0 closely.

* `run_with_fermi_zoom` — fermions.  Coarse pass uses the standard
  loop; once the polylog evaluation crosses into the analytic-
  continuation region (alpha > 0, z = -e^alpha < -1) `mp.polylog`
  begins to return `mpc` values with tiny imaginary residues and
  `float(...)` of the truncation outputs raises `TypeError`.  The fine
  pass re-seeds a few steps before this halt with higher precision and
  `real_part_on_mpc=True`, which strips those imaginary residues via
  `mp.re` and lets the iteration continue into the degenerate regime.

Both functions return a stitched `(results, ZoomOutcome)` pair that is
drop-in compatible with `save_run` and the plotting helpers.

This module imports only public names from `evap_cool.evaporation` and
`evap_cool.thermodynamics.base`; it adds no new dependencies and does
not modify the existing simulation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mpmath as mp

from .evaporation import (
    RunOutcome,
    create_result_dict,
    build_cutoff_schedule,
    initialize_quantum_state,
    run_quantum_evaporation,
)
from .thermodynamics.base import Trap


# ---------------------------------------------------------------------------
# Outcome container
# ---------------------------------------------------------------------------
@dataclass
class ZoomOutcome:
    """Summary of a two-stage zoomed evaporation run.

    Duck-types as `evap_cool.evaporation.RunOutcome`: it exposes the
    attributes (`n_completed`, `n_steps_requested`, `halted_early`,
    `halt_reason`) that `save_run` reads, so a `ZoomOutcome` can be
    passed directly as `outcome=` to persist the merged trajectory.
    Per-stage detail lives in `to_metadata()`, intended for the
    `extra_metadata` argument of `save_run`.

    Attributes
    ----------
    coarse : RunOutcome
        Outcome of the stage-1 (coarse) pass.
    fine : RunOutcome or None
        Outcome of the stage-2 (fine) pass. None if stage 2 was skipped
        (coarse pass did not halt early, or halted too early to seed).
    seed_index : int
        Index into the coarse results dict from which stage 2 was seeded.
        Equals `coarse.n_completed - backoff` (clamped to >= 1).
    n_total_committed : int
        Total committed evaporation steps across both stages
        (= `len(combined["N"]) - 1`).
    """
    coarse: RunOutcome
    fine: Optional[RunOutcome]
    seed_index: int
    n_total_committed: int

    # ------------------------------------------------------------------
    # RunOutcome-compatible interface for save_run
    # ------------------------------------------------------------------
    @property
    def n_completed(self) -> int:
        """Total committed steps across both stages."""
        return self.n_total_committed

    @property
    def n_steps_requested(self) -> int:
        """Total steps requested across both stages."""
        n = self.coarse.n_steps_requested
        if self.fine is not None:
            n += self.fine.n_steps_requested
        return n

    @property
    def halted_early(self) -> bool:
        """True if the *last* stage to run halted early.

        Conventionally, a zoom run is considered to have halted early
        only if the fine stage did. If stage 2 was skipped, fall back
        to the coarse outcome.
        """
        if self.fine is not None:
            return self.fine.halted_early
        return self.coarse.halted_early

    @property
    def halt_reason(self) -> Optional[str]:
        """Halt reason of the last stage to run (None if completed)."""
        if self.fine is not None:
            return self.fine.halt_reason
        return self.coarse.halt_reason

    # ------------------------------------------------------------------
    # Serialization helper for save_run(extra_metadata=...)
    # ------------------------------------------------------------------
    def to_metadata(self) -> dict:
        """Return a JSON-serializable dict describing both stages.

        Intended use::

            save_run(combined, path, ..., outcome=zoom_outcome,
                     extra_metadata={"zoom": zoom_outcome.to_metadata()})
        """
        def _encode(o: Optional[RunOutcome]) -> Optional[dict]:
            if o is None:
                return None
            return {
                "n_completed": o.n_completed,
                "n_steps_requested": o.n_steps_requested,
                "halted_early": o.halted_early,
                "halt_reason": o.halt_reason,
            }

        return {
            "seed_index": self.seed_index,
            "n_total_committed": self.n_total_committed,
            "coarse": _encode(self.coarse),
            "fine": _encode(self.fine),
        }


# ---------------------------------------------------------------------------
# Helper: extract state vector at a given index
# ---------------------------------------------------------------------------
def _state_at(results: dict, i: int) -> Tuple[float, float, float, float, float]:
    """Return (N, T, Mu, E, Omega) at index i from a results dict."""
    return (
        results["N"][i],
        results["T"][i],
        results["Mu"][i],
        results["E"][i],
        results["Omega"][i],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_with_bec_zoom(
    trap: Trap,
    N0: float, T0: float, mu0: float, E0: float, Omega0: float,
    *,
    Q0: float,
    dQ_coarse: float,
    n_steps_coarse: int,
    dQ_fine: float,
    n_steps_fine: int,
    dT: float = 1e-20,
    dmu: float = 1e-30,
    sign: int = +1,
    alpha_floor_coarse: float = -1e-3,
    alpha_floor_fine: float = -1e-12,
    dps_coarse: int = 15,
    dps_fine: int = 60,
    backoff: int = 5,
    restore_dps: bool = True,
    verbose: bool = True,
) -> Tuple[dict, ZoomOutcome]:
    """Run a coarse evaporation pass, then zoom in on the BEC approach.

    Stage 1 runs with `dQ_coarse` and `alpha_floor_coarse` to locate the
    halt step. Stage 2 re-seeds `backoff` steps before the halt and runs
    at higher mpmath precision (`dps_fine`), tighter `alpha_floor_fine`,
    and finer cut spacing (`dQ_fine`). The two trajectories are stitched
    into a single results dict; the duplicated seed point is dropped and
    `Nf` / `Tf` are re-normalized against the original `(N0, T0)`.

    Parameters
    ----------
    trap : Trap
        Trap object. Must support the standard quantum evaporation API.
    N0, T0, mu0, E0, Omega0 : float
        Initial thermodynamic state for stage 1.
    Q0, dQ_coarse, n_steps_coarse : float, float, int
        Stage-1 Q-schedule (linear: Q0 - i*dQ_coarse for i = 1..n_steps).
    dQ_fine, n_steps_fine : float, int
        Stage-2 Q-schedule (anchored at the seed Q).
    dT, dmu : float
        NR initial offsets, passed through unchanged to both stages.
    sign : int
        +1 for bosons (zoom is meaningful), -1 for fermions (alpha_floor
        is ignored by the loop, but stage 2 still gives finer resolution).
    alpha_floor_coarse, alpha_floor_fine : float
        Boson safety thresholds for each stage. Stage 2 should be much
        closer to 0 (e.g. -1e-12) to allow approach to BEC.
    dps_coarse, dps_fine : int
        mpmath decimal precision for each stage. Stage 2 needs > ~50
        digits because polylog evaluation near the z=1 branch point
        loses precision rapidly.
    backoff : int
        Number of steps before the coarse halt to seed stage 2. Larger
        values give stage 2 a more comfortable equilibrium to restart
        from but waste a little overlap.
    restore_dps : bool
        If True (default), the previous `mp.mp.dps` is restored on exit
        so callers downstream don't inherit `dps_fine`. Set to False if
        you want the elevated precision to persist (e.g. for follow-up
        equilibrium calculations on the final state).
    verbose : bool
        Print stage banners and seed diagnostics.

    Returns
    -------
    combined : dict
        Stitched results dict, ready for `save_run` and the plotting
        helpers. `Q` is exactly the list of cuts that were applied
        (length = `len(N) - 1`), so plots align without padding.
    outcome : ZoomOutcome
        Per-stage outcomes plus the seed index.
    """
    # ------------------------------------------------------------------
    # Stage 1: coarse pass — locate the cliff
    # ------------------------------------------------------------------
    saved_dps = mp.mp.dps
    try:
        mp.mp.dps = dps_coarse
        if verbose:
            print(f"[zoom] Stage 1 (coarse): dQ={dQ_coarse:g}, "
                  f"alpha_floor={alpha_floor_coarse:g}, dps={dps_coarse}")

        coarse = create_result_dict()
        coarse["Q"] = build_cutoff_schedule(Q0, dQ_coarse, n_steps_coarse)
        initialize_quantum_state(coarse, N0, T0, mu0, E0, Omega0)

        out_coarse = run_quantum_evaporation(
            coarse, trap, N0,
            n_steps=n_steps_coarse, dT=dT, dmu=dmu, sign=sign,
            alpha_floor=alpha_floor_coarse, verbose=verbose,
        )

        # If coarse never halted, the schedule was too short or alpha_floor
        # too lax. Stage 2 has nothing to zoom into; bail out cleanly.
        if not out_coarse.halted_early:
            if verbose:
                print("[zoom] coarse pass completed without halting; "
                      "skipping fine stage.")
            return coarse, ZoomOutcome(
                coarse=out_coarse, fine=None,
                seed_index=out_coarse.n_completed,
                n_total_committed=out_coarse.n_completed,
            )

        # If the coarse pass died too early, there's nothing meaningful to
        # seed from. (We need at least one good step plus the initial state.)
        if out_coarse.n_completed < max(2, backoff):
            if verbose:
                print(f"[zoom] coarse halted too early "
                      f"(n_completed={out_coarse.n_completed}); "
                      f"skipping fine stage. Try a more conservative "
                      f"alpha_floor_coarse or smaller dQ_coarse.")
            return coarse, ZoomOutcome(
                coarse=out_coarse, fine=None,
                seed_index=out_coarse.n_completed,
                n_total_committed=out_coarse.n_completed,
            )

        # ------------------------------------------------------------------
        # Pick the seed point: a few steps before the halt
        # ------------------------------------------------------------------
        seed_index = max(1, out_coarse.n_completed - backoff)
        seed_N, seed_T, seed_Mu, seed_E, seed_Omega = _state_at(coarse, seed_index)
        # Q[i] is the cut applied during step i to produce N[i+1]. So the cut
        # that produced `seed_index` lives at coarse["Q"][seed_index - 1].
        seed_Q = coarse["Q"][seed_index - 1]

        if verbose:
            alpha_seed = seed_Mu / (trap.kB * seed_T)
            print(f"[zoom] coarse halt @ step {out_coarse.n_completed} "
                  f"({out_coarse.halt_reason})")
            print(f"[zoom] seed @ index {seed_index}: "
                  f"T={seed_T:.6e}, alpha={alpha_seed:.3e}, "
                  f"Q_seed={seed_Q:.6e}")

        # ------------------------------------------------------------------
        # Stage 2: fine pass — zoom in
        # ------------------------------------------------------------------
        mp.mp.dps = dps_fine
        if verbose:
            print(f"[zoom] Stage 2 (fine):   dQ={dQ_fine:g}, "
                  f"alpha_floor={alpha_floor_fine:g}, dps={dps_fine}")

        fine = create_result_dict()
        fine["Q"] = build_cutoff_schedule(seed_Q, dQ_fine, n_steps_fine)
        initialize_quantum_state(fine, seed_N, seed_T, seed_Mu, seed_E, seed_Omega)

        out_fine = run_quantum_evaporation(
            fine, trap, seed_N,                # local N0 for fine-stage Nf
            n_steps=n_steps_fine, dT=dT, dmu=dmu, sign=sign,
            alpha_floor=alpha_floor_fine, verbose=verbose,
        )

        # ------------------------------------------------------------------
        # Stitch:  coarse[:seed_index+1]  ++  fine[1:]   (drop dup'd seed)
        # ------------------------------------------------------------------
        combined = create_result_dict()

        for key in ("N", "T", "Mu", "E", "Omega"):
            combined[key] = list(coarse[key][:seed_index + 1]) + list(fine[key][1:])

        # Q for the merged trajectory: coarse cuts that produced states
        # 1..seed_index (= coarse["Q"][:seed_index]) followed by the fine
        # cuts that were *actually applied* (= fine["Q"][:n_fine_committed]).
        # This makes len(combined["Q"]) == len(combined["N"]) - 1, so plots
        # align without needing align_results().
        n_fine_committed = len(fine["N"]) - 1
        combined["Q"] = (
            list(coarse["Q"][:seed_index])
            + list(fine["Q"][:n_fine_committed])
        )

        # Re-normalize against the *original* (N0, T0) so the seam at the
        # seed point is invisible in normalized plots.
        combined["Nf"] = [n / N0 for n in combined["N"][:-1]]
        combined["Tf"] = [t / T0 for t in combined["T"][:-1]]

        n_total = len(combined["N"]) - 1

        if verbose:
            print(f"[zoom] stitched: {n_total} total committed steps "
                  f"({seed_index} coarse + {n_fine_committed} fine)")

        return combined, ZoomOutcome(
            coarse=out_coarse, fine=out_fine,
            seed_index=seed_index,
            n_total_committed=n_total,
        )

    finally:
        if restore_dps:
            mp.mp.dps = saved_dps


# ---------------------------------------------------------------------------
# Private stitching helper (shared by both public zoom entry points)
# ---------------------------------------------------------------------------
def _stitch(coarse: dict, fine: dict, seed_index: int,
            N0: float, T0: float) -> dict:
    """Splice  coarse[:seed_index+1]  ++  fine[1:]  (drop duplicated seed).

    The duplicated seed at index 0 of `fine` is the same state as
    `coarse[seed_index]`, so dropping `fine[0]` yields a contiguous
    trajectory with no doubled point.

    `Q` is set to *exactly* the cuts that produced the committed states,
    so `len(combined["Q"]) == len(combined["N"]) - 1` and plots align
    without `align_results`. `Nf` and `Tf` are renormalized against the
    original `(N0, T0)` so the seam at the seed point is invisible in
    normalized plots.
    """
    combined = create_result_dict()
    for key in ("N", "T", "Mu", "E", "Omega"):
        combined[key] = list(coarse[key][:seed_index + 1]) + list(fine[key][1:])

    n_fine_committed = len(fine["N"]) - 1
    combined["Q"] = (
        list(coarse["Q"][:seed_index])
        + list(fine["Q"][:n_fine_committed])
    )

    combined["Nf"] = [n / N0 for n in combined["N"][:-1]]
    combined["Tf"] = [t / T0 for t in combined["T"][:-1]]
    return combined


# ---------------------------------------------------------------------------
# Public API: Fermi zoom
# ---------------------------------------------------------------------------
def run_with_fermi_zoom(
    trap: Trap,
    N0: float, T0: float, mu0: float, E0: float, Omega0: float,
    *,
    Q0: float,
    dQ_coarse: float,
    n_steps_coarse: int,
    dQ_fine: float,
    n_steps_fine: int,
    dT: float = 1e-20,
    dmu: float = 1e-30,
    sign: int = -1,
    dps_coarse: int = 15,
    dps_fine: int = 60,
    backoff: int = 5,
    restore_dps: bool = True,
    verbose: bool = True,
) -> Tuple[dict, ZoomOutcome]:
    """Run a coarse fermion pass, then zoom in through the mpc-degeneracy halt.

    Same two-stage dynamic as `run_with_bec_zoom`, with two differences:

    * No `alpha_floor`.  Fermions have no BEC singularity, so the loop's
      alpha guard is ignored (`alpha_floor` is bosons-only inside
      `run_quantum_evaporation`).  The natural fermion halt comes from
      `float(mpc)` raising `TypeError` once the polylog argument crosses
      below z = -1 — i.e. once alpha = mu / (kB T) climbs above 0.

    * Stage 2 sets `real_part_on_mpc=True`, which strips imaginary
      residues from the truncation outputs (`N_new`, `E_new`,
      `Omega_new`) via `mp.re` before casting to float.  This lets the
      iteration continue into the degenerate regime that the strict
      coarse pass cannot reach.

    Stage 1 is intentionally left strict (`real_part_on_mpc=False`) so
    that the halt index marks the genuine onset of the degenerate
    regime rather than something further on.  Stage 2 then re-seeds
    `backoff` steps before that halt with higher mpmath precision
    (`dps_fine`) and a finer Q-schedule.

    Parameters
    ----------
    trap : Trap
        Trap object. Must support the standard quantum evaporation API.
    N0, T0, mu0, E0, Omega0 : float
        Initial thermodynamic state for stage 1.
    Q0, dQ_coarse, n_steps_coarse : float, float, int
        Stage-1 Q-schedule (linear: Q0 - i*dQ_coarse for i = 1..n_steps).
    dQ_fine, n_steps_fine : float, int
        Stage-2 Q-schedule (anchored at the seed Q).
    dT, dmu : float
        NR initial offsets, passed through unchanged to both stages.
    sign : int
        Statistics sign. Defaults to -1 (fermions) — the only case this
        function is designed for.  Passing +1 will work but does *not*
        emulate `run_with_bec_zoom`: there is no alpha_floor argument
        here, and bosons relying on alpha_floor for halt-detection
        should use `run_with_bec_zoom` instead.
    dps_coarse, dps_fine : int
        mpmath decimal precision for each stage.  Stage 2 typically
        wants > ~50 digits because polylog evaluation across the
        z = -1 branch loses precision rapidly.
    backoff : int
        Number of steps before the coarse halt to seed stage 2.  Larger
        values give stage 2 a more comfortable equilibrium to restart
        from but waste a little overlap.
    restore_dps : bool
        If True (default), the previous `mp.mp.dps` is restored on exit.
    verbose : bool
        Print stage banners and seed diagnostics.

    Returns
    -------
    combined : dict
        Stitched results dict, ready for `save_run` and the plotting
        helpers.  `Q` is exactly the list of cuts that were applied
        (`len(Q) = len(N) - 1`), so plots align without padding.
    outcome : ZoomOutcome
        Per-stage outcomes plus the seed index.
    """
    # ------------------------------------------------------------------
    # Stage 1: coarse pass — locate the mpc-degeneracy halt
    # ------------------------------------------------------------------
    saved_dps = mp.mp.dps
    try:
        mp.mp.dps = dps_coarse
        if verbose:
            print(f"[fermi-zoom] Stage 1 (coarse): dQ={dQ_coarse:g}, "
                  f"dps={dps_coarse}, real_part_on_mpc=False")

        coarse = create_result_dict()
        coarse["Q"] = build_cutoff_schedule(Q0, dQ_coarse, n_steps_coarse)
        initialize_quantum_state(coarse, N0, T0, mu0, E0, Omega0)

        out_coarse = run_quantum_evaporation(
            coarse, trap, N0,
            n_steps=n_steps_coarse, dT=dT, dmu=dmu, sign=sign,
            real_part_on_mpc=False, verbose=verbose,
        )

        # If coarse never halted, there's nothing for stage 2 to rescue.
        if not out_coarse.halted_early:
            if verbose:
                print("[fermi-zoom] coarse pass completed without halting; "
                      "skipping fine stage.")
            return coarse, ZoomOutcome(
                coarse=out_coarse, fine=None,
                seed_index=out_coarse.n_completed,
                n_total_committed=out_coarse.n_completed,
            )

        # If coarse died too early, we can't seed stage 2 from anything
        # meaningful.  Match the BEC-zoom guard: need at least one good
        # step in addition to the initial state.
        if out_coarse.n_completed < max(2, backoff):
            if verbose:
                print(f"[fermi-zoom] coarse halted too early "
                      f"(n_completed={out_coarse.n_completed}); "
                      f"skipping fine stage. Try a smaller dQ_coarse.")
            return coarse, ZoomOutcome(
                coarse=out_coarse, fine=None,
                seed_index=out_coarse.n_completed,
                n_total_committed=out_coarse.n_completed,
            )

        # ------------------------------------------------------------------
        # Pick the seed point: a few steps before the halt
        # ------------------------------------------------------------------
        seed_index = max(1, out_coarse.n_completed - backoff)
        seed_N, seed_T, seed_Mu, seed_E, seed_Omega = _state_at(
            coarse, seed_index,
        )
        # Q[i] is the cut applied during step i to produce N[i+1].  The
        # cut that produced `seed_index` lives at coarse["Q"][seed_index-1].
        seed_Q = coarse["Q"][seed_index - 1]

        if verbose:
            alpha_seed = seed_Mu / (trap.kB * seed_T)
            print(f"[fermi-zoom] coarse halt @ step {out_coarse.n_completed} "
                  f"({out_coarse.halt_reason})")
            print(f"[fermi-zoom] seed @ index {seed_index}: "
                  f"T={seed_T:.6e}, alpha={alpha_seed:+.3e}, "
                  f"Q_seed={seed_Q:.6e}")

        # ------------------------------------------------------------------
        # Stage 2: fine pass — mpc-tolerant continuation
        # ------------------------------------------------------------------
        mp.mp.dps = dps_fine
        if verbose:
            print(f"[fermi-zoom] Stage 2 (fine):   dQ={dQ_fine:g}, "
                  f"dps={dps_fine}, real_part_on_mpc=True")

        fine = create_result_dict()
        fine["Q"] = build_cutoff_schedule(seed_Q, dQ_fine, n_steps_fine)
        initialize_quantum_state(fine, seed_N, seed_T, seed_Mu, seed_E, seed_Omega)

        out_fine = run_quantum_evaporation(
            fine, trap, seed_N,                # local N0 for fine-stage Nf
            n_steps=n_steps_fine, dT=dT, dmu=dmu, sign=sign,
            real_part_on_mpc=True, verbose=verbose,
        )

        # ------------------------------------------------------------------
        # Stitch:  coarse[:seed_index+1]  ++  fine[1:]   (drop dup'd seed)
        # ------------------------------------------------------------------
        combined = _stitch(coarse, fine, seed_index, N0, T0)
        n_fine_committed = len(fine["N"]) - 1
        n_total = len(combined["N"]) - 1

        if verbose:
            print(f"[fermi-zoom] stitched: {n_total} total committed steps "
                  f"({seed_index} coarse + {n_fine_committed} fine)")

        return combined, ZoomOutcome(
            coarse=out_coarse, fine=out_fine,
            seed_index=seed_index,
            n_total_committed=n_total,
        )

    finally:
        if restore_dps:
            mp.mp.dps = saved_dps