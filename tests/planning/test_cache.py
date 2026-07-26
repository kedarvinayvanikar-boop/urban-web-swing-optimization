"""Tests for transfer-result caching and edge derivation.

`TransferCache` is tested against its own hit/miss bookkeeping directly.
`edge_cost` and `resulting_planning_state` are exercised against a real
`LocalTransferResult` produced by `solve_local_transfer` on the same
reachable scenario used in `test_local_transfer.py` (derived by simulating
the physics first, not by asserting arbitrary numbers), cross-checked
against an independent direct `attach_to_anchor` + `discretize_state` call.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.dynamics.attachment import attach_to_anchor
from webswing.optimization.local_transfer import LocalTransferProblem, solve_local_transfer
from webswing.planning.cache import TransferCache, edge_cost, resulting_planning_state
from webswing.planning.state import PlanningState, PlanningStateResolution, discretize_state

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)
RESOLUTION = PlanningStateResolution(
    theta_bin_width=0.1, omega_bin_width=0.2, ell_bin_width=0.5, nu_bin_width=0.1
)


def make_reachable_result():
    constraints = SwingConstraints(
        tension_max=200.0, load_factor_max=20.0, ell_min=1.0, ell_max=30.0, nu_max=5.0
    )
    domain = BallisticDomain(x_min=-200.0, x_max=200.0, y_max=200.0, max_duration=10.0)
    problem = LocalTransferProblem(
        initial_state=np.array([-0.3, 1.0, 10.0, 0.0]),
        current_anchor=(0.0, 50.0),
        target_anchor=(22.0, 27.0),
        params=PARAMS,
        constraints=constraints,
        u_min=-0.5,
        u_max=0.5,
        n_control_segments=3,
        t_release_min=0.1,
        t_release_max=3.0,
        capture_radius=4.0,
        ballistic_domain=domain,
    )
    return solve_local_transfer(problem)


# --- TransferCache -----------------------------------------------------------------


def test_cache_computes_once_and_reuses_on_repeated_key() -> None:
    cache = TransferCache()
    from_state = PlanningState(anchor_id="A", theta_bin=0, omega_bin=0, ell_bin=1, nu_bin=0)
    calls = []

    def compute():
        calls.append(1)
        return "sentinel-result"

    first = cache.get_or_compute(from_state, "B", compute)
    second = cache.get_or_compute(from_state, "B", compute)

    assert first == "sentinel-result"
    assert second == "sentinel-result"
    assert len(calls) == 1
    assert cache.hits == 1
    assert cache.misses == 1
    assert len(cache) == 1


def test_cache_distinguishes_different_anchor_targets() -> None:
    cache = TransferCache()
    from_state = PlanningState(anchor_id="A", theta_bin=0, omega_bin=0, ell_bin=1, nu_bin=0)
    cache.get_or_compute(from_state, "B", lambda: "to-B")
    cache.get_or_compute(from_state, "C", lambda: "to-C")
    assert len(cache) == 2
    assert cache.misses == 2


def test_cache_distinguishes_different_from_states() -> None:
    cache = TransferCache()
    state_a = PlanningState(anchor_id="A", theta_bin=0, omega_bin=0, ell_bin=1, nu_bin=0)
    state_b = PlanningState(anchor_id="A", theta_bin=5, omega_bin=0, ell_bin=1, nu_bin=0)
    cache.get_or_compute(state_a, "TARGET", lambda: "result-a")
    cache.get_or_compute(state_b, "TARGET", lambda: "result-b")
    assert len(cache) == 2


def test_cache_contains_reports_key_presence() -> None:
    cache = TransferCache()
    from_state = PlanningState(anchor_id="A", theta_bin=0, omega_bin=0, ell_bin=1, nu_bin=0)
    assert (from_state, "B") not in cache
    cache.get_or_compute(from_state, "B", lambda: "x")
    assert (from_state, "B") in cache


# --- edge_cost ---------------------------------------------------------------------


def test_edge_cost_of_successful_result_equals_release_plus_ballistic_time() -> None:
    result = make_reachable_result()
    assert result.success is True
    assert edge_cost(result) == pytest.approx(result.release_time + result.ballistic_time, rel=1e-12)


def test_edge_cost_of_failed_result_is_infinite() -> None:
    result = make_reachable_result()
    failed = replace(result, success=False)
    assert edge_cost(failed) == math.inf


# --- resulting_planning_state -------------------------------------------------------


def test_resulting_planning_state_matches_independent_attach_and_discretize() -> None:
    result = make_reachable_result()
    to_anchor = (22.0, 27.0)

    node = resulting_planning_state(result, "TARGET", to_anchor, PARAMS, RESOLUTION)

    x, y, vx, vy = result.final_state
    attachment = attach_to_anchor(x, y, vx, vy, to_anchor[0], to_anchor[1], PARAMS)
    expected = discretize_state("TARGET", attachment.z, RESOLUTION)

    assert node == expected
    assert node.anchor_id == "TARGET"


def test_resulting_planning_state_is_none_for_failed_transfer() -> None:
    result = make_reachable_result()
    failed = replace(result, success=False)
    assert resulting_planning_state(failed, "TARGET", (22.0, 27.0), PARAMS, RESOLUTION) is None


def test_resulting_planning_state_is_none_when_attachment_range_exceeded() -> None:
    result = make_reachable_result()
    to_anchor = (22.0, 27.0)
    x, y, _vx, _vy = result.final_state
    actual_capture_distance = math.hypot(x - to_anchor[0], y - to_anchor[1])

    node = resulting_planning_state(
        result, "TARGET", to_anchor, PARAMS, RESOLUTION, max_attachment_range=actual_capture_distance * 0.5
    )
    assert node is None
