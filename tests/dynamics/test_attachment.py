"""Tests for the ballistic-to-attached attachment transition.

Validates web-length/angle reconstruction against the coordinate mapping
already exercised in `test_release.py` (round trip through
`attached_position`), the radial/tangential velocity decomposition
(orthogonality and reconstruction of the original vector), the
nonnegativity and exact form of the energy-loss identity, the two boundary
capture cases (pure-radial and pure-tangential incoming velocity), and the
two implemented rejection conditions (degenerate geometry, excess range).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from webswing.config import PhysicalParameters
from webswing.dynamics.attachment import (
    AttachmentResult,
    attach_to_anchor,
    attachment_energy_loss,
    decompose_velocity,
    tangential_velocity_to_angular_rate,
    web_length_and_angle,
)
from webswing.dynamics.release import attached_position, attached_velocity
from webswing.exceptions import AttachmentRangeExceededError, NonPositiveWebLengthError

PARAMS = PhysicalParameters(mass=2.0, gravity=9.80665)
X_ANCHOR, Y_ANCHOR = 5.0, 12.0


def test_web_length_and_angle_round_trips_through_attached_position() -> None:
    theta_true, ell_true = -0.9, 4.0
    x, y = attached_position(theta_true, ell_true, X_ANCHOR, Y_ANCHOR)
    ell, theta = web_length_and_angle(x, y, X_ANCHOR, Y_ANCHOR)
    assert ell == pytest.approx(ell_true, rel=1e-12)
    assert theta == pytest.approx(theta_true, rel=1e-12)


def test_web_length_and_angle_rejects_coincident_anchor() -> None:
    with pytest.raises(NonPositiveWebLengthError):
        web_length_and_angle(X_ANCHOR, Y_ANCHOR, X_ANCHOR, Y_ANCHOR)


def test_decompose_velocity_reconstructs_original_vector() -> None:
    theta = 0.35
    vx, vy = 3.0, -2.0
    (vx_r, vy_r), (vx_t, vy_t) = decompose_velocity(vx, vy, theta)
    assert (vx_r + vx_t) == pytest.approx(vx, rel=1e-12)
    assert (vy_r + vy_t) == pytest.approx(vy, rel=1e-12)


def test_decompose_velocity_components_are_orthogonal() -> None:
    theta = 1.1
    vx, vy = -1.5, 4.2
    v_radial, v_tangential = decompose_velocity(vx, vy, theta)
    dot = v_radial[0] * v_tangential[0] + v_radial[1] * v_tangential[1]
    assert dot == pytest.approx(0.0, abs=1e-12)


def test_pure_radial_velocity_has_zero_tangential_component() -> None:
    theta = 0.7
    e_r = (math.sin(theta), -math.cos(theta))
    speed = 6.0
    vx, vy = speed * e_r[0], speed * e_r[1]
    v_radial, v_tangential = decompose_velocity(vx, vy, theta)
    assert v_tangential[0] == pytest.approx(0.0, abs=1e-10)
    assert v_tangential[1] == pytest.approx(0.0, abs=1e-10)
    assert v_radial[0] == pytest.approx(vx, rel=1e-10)
    assert v_radial[1] == pytest.approx(vy, rel=1e-10)


def test_pure_tangential_velocity_recovers_exact_omega() -> None:
    theta, ell, omega_true = 0.4, 3.0, 1.3
    vx, vy = attached_velocity(theta, omega_true, ell, nu=0.0)
    v_radial, v_tangential = decompose_velocity(vx, vy, theta)
    assert v_radial[0] == pytest.approx(0.0, abs=1e-10)
    assert v_radial[1] == pytest.approx(0.0, abs=1e-10)
    omega = tangential_velocity_to_angular_rate(v_tangential[0], v_tangential[1], theta, ell)
    assert omega == pytest.approx(omega_true, rel=1e-10)


def test_tangential_velocity_to_angular_rate_rejects_nonpositive_length() -> None:
    with pytest.raises(NonPositiveWebLengthError):
        tangential_velocity_to_angular_rate(1.0, 0.0, 0.0, ell=0.0)


def test_energy_loss_equals_half_mass_times_radial_speed_squared() -> None:
    rng = np.random.default_rng(20260720)
    theta = 0.2
    for _ in range(20):
        vx, vy = rng.uniform(-10.0, 10.0, size=2)
        v_radial, v_tangential = decompose_velocity(vx, vy, theta)
        loss = attachment_energy_loss(vx, vy, v_tangential[0], v_tangential[1], PARAMS.mass)
        v_r_sq = v_radial[0] ** 2 + v_radial[1] ** 2
        assert loss == pytest.approx(0.5 * PARAMS.mass * v_r_sq, rel=1e-10)


def test_energy_loss_is_nonnegative() -> None:
    rng = np.random.default_rng(42)
    for _ in range(50):
        theta = rng.uniform(-math.pi, math.pi)
        vx, vy = rng.uniform(-15.0, 15.0, size=2)
        _, v_tangential = decompose_velocity(vx, vy, theta)
        loss = attachment_energy_loss(vx, vy, v_tangential[0], v_tangential[1], PARAMS.mass)
        assert loss >= -1e-12


def test_energy_loss_zero_for_pure_tangential_incoming_velocity() -> None:
    theta, ell, omega = -0.6, 2.5, 0.9
    vx, vy = attached_velocity(theta, omega, ell, nu=0.0)
    _, v_tangential = decompose_velocity(vx, vy, theta)
    loss = attachment_energy_loss(vx, vy, v_tangential[0], v_tangential[1], PARAMS.mass)
    assert loss == pytest.approx(0.0, abs=1e-10)


def test_attach_to_anchor_matches_component_functions() -> None:
    theta_true, ell_true, omega_true = 0.5, 4.0, 0.6
    x, y = attached_position(theta_true, ell_true, X_ANCHOR, Y_ANCHOR)
    # nu=0 so the incoming velocity is purely tangential, giving a known omega.
    vx, vy = attached_velocity(theta_true, omega_true, ell_true, nu=0.0)

    result = attach_to_anchor(x, y, vx, vy, X_ANCHOR, Y_ANCHOR, PARAMS)

    assert isinstance(result, AttachmentResult)
    assert result.z.shape == (4,)
    theta, omega, ell, nu = result.z
    assert theta == pytest.approx(theta_true, rel=1e-10)
    assert ell == pytest.approx(ell_true, rel=1e-10)
    assert omega == pytest.approx(omega_true, rel=1e-10)
    assert nu == 0.0
    assert result.energy_loss == pytest.approx(0.0, abs=1e-8)
    assert result.x_anchor == X_ANCHOR
    assert result.y_anchor == Y_ANCHOR


def test_attach_to_anchor_reports_lossy_capture_for_radial_incoming_velocity() -> None:
    theta_true, ell_true = 0.2, 3.0
    x, y = attached_position(theta_true, ell_true, X_ANCHOR, Y_ANCHOR)
    e_r = (math.sin(theta_true), -math.cos(theta_true))
    speed = 5.0
    vx, vy = speed * e_r[0], speed * e_r[1]

    result = attach_to_anchor(x, y, vx, vy, X_ANCHOR, Y_ANCHOR, PARAMS)

    assert result.z[1] == pytest.approx(0.0, abs=1e-10)  # omega
    assert result.energy_loss == pytest.approx(0.5 * PARAMS.mass * speed**2, rel=1e-10)


def test_attach_to_anchor_rejects_degenerate_geometry() -> None:
    with pytest.raises(NonPositiveWebLengthError):
        attach_to_anchor(X_ANCHOR, Y_ANCHOR, 1.0, 0.0, X_ANCHOR, Y_ANCHOR, PARAMS)


def test_attach_to_anchor_rejects_excess_range() -> None:
    x, y = attached_position(0.3, 10.0, X_ANCHOR, Y_ANCHOR)
    with pytest.raises(AttachmentRangeExceededError):
        attach_to_anchor(x, y, 0.0, 0.0, X_ANCHOR, Y_ANCHOR, PARAMS, max_attachment_range=5.0)


def test_attach_to_anchor_accepts_within_range() -> None:
    x, y = attached_position(0.3, 4.0, X_ANCHOR, Y_ANCHOR)
    result = attach_to_anchor(x, y, 0.0, 0.0, X_ANCHOR, Y_ANCHOR, PARAMS, max_attachment_range=5.0)
    assert result.z[2] == pytest.approx(4.0, rel=1e-10)
