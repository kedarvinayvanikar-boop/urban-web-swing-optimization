r"""Attachment transition: ballistic Cartesian state to attached pendulum state.

A new web attachment is not automatically energy-preserving (documented in
CLAUDE.md). Given a pre-attachment Cartesian state (x, y, vx, vy) and a
candidate anchor a = (x_a, y_a), define the anchor-to-body vector

    r = (x - x_a, y - y_a),   l = |r|,   e_r = r / l.

Under the coordinate convention x = x_a + l*sin(theta), y = y_a - l*cos(theta)
(theta from the downward vertical, documented in CLAUDE.md and used
throughout this package), r = l*(sin(theta), -cos(theta)), so

    theta = atan2(x - x_a, y_a - y)

recovers the swing angle directly from position, and e_r = (sin(theta),
-cos(theta)) without a separate division.

The velocity decomposes as v = v_r + v_t, where v_r = (v . e_r) e_r is the
radial component and v_t = v - v_r is the tangential component. This module
implements the documented default capture model: the radial component is
removed instantaneously and the tangential component is retained,

    v^+ = v_t,

an idealized inelastic attachment. The corresponding kinetic-energy loss is

    Delta_E = (1/2) m (|v|^2 - |v_t|^2) = (1/2) m |v_r|^2 >= 0,

by the Pythagorean identity |v|^2 = |v_r|^2 + |v_t|^2 (v_r and v_t are
orthogonal by construction), so the loss is nonnegative for every input.

The retained tangential velocity is converted to an angular rate using the
forward mapping from `dynamics.release` with l_dot = 0 (the radial rate is
zero immediately after capture, since the radial velocity was just removed):

    vx = l*omega*cos(theta),   vy = l*omega*sin(theta).

Since v_t is orthogonal to e_r = (sin(theta), -cos(theta)), it is
necessarily parallel to e_theta = (cos(theta), sin(theta)) (the only other
direction in the plane orthogonal to e_r), so

    omega = (vx_t*cos(theta) + vy_t*sin(theta)) / l

recovers the angular rate exactly, regardless of how v_t arose.

This module implements only the kinematic and energetic core of the
attachment transition together with the maximum-attachment-range and
degenerate-geometry rejections, which require no interface beyond this
package. It deliberately does not implement: line-of-sight/building
intersection rejection (requires `geometry`, not yet implemented), web-
tension or load-factor rejection (requires `optimization.constraints`, not
yet implemented), or attachment-impulse-limit rejection. Callers composing a
full attachment feasibility check must apply those rejections separately
once the corresponding modules exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from webswing.config import PhysicalParameters
from webswing.exceptions import AttachmentRangeExceededError, NonPositiveWebLengthError

_MIN_WEB_LENGTH_M = 1.0e-9
"""Web lengths at or below this are treated as a degenerate (coincident-anchor) geometry."""


def web_length_and_angle(x: float, y: float, x_anchor: float, y_anchor: float) -> tuple[float, float]:
    r"""Recover web length and swing angle from a Cartesian position.

    Inverts x = x_a + l*sin(theta), y = y_a - l*cos(theta) for (l, theta).

    Parameters
    ----------
    x, y : float
        Body position, in meters.
    x_anchor, y_anchor : float
        Anchor position, in meters.

    Returns
    -------
    tuple[float, float]
        (ell, theta): web length in meters, swing angle in radians from the
        downward vertical, in (-pi, pi].

    Raises
    ------
    NonPositiveWebLengthError
        If the position coincides with the anchor (or is otherwise within
        `_MIN_WEB_LENGTH_M` of it), making theta numerically undefined.
    """
    dx = x - x_anchor
    dy = y - y_anchor
    ell = math.hypot(dx, dy)
    if not (math.isfinite(ell) and ell > _MIN_WEB_LENGTH_M):
        raise NonPositiveWebLengthError(ell)
    theta = math.atan2(dx, -dy)
    return ell, theta


def decompose_velocity(
    vx: float, vy: float, theta: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    r"""Decompose a Cartesian velocity into radial and tangential components.

    e_r = (sin(theta), -cos(theta)) is the unit vector from the anchor
    toward the body. The radial component is the projection of v onto e_r;
    the tangential component is the remainder, and is orthogonal to e_r by
    construction.

    Parameters
    ----------
    vx, vy : float
        Cartesian velocity, in m/s.
    theta : float
        Swing angle from the downward vertical, in radians.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        ((vx_r, vy_r), (vx_t, vy_t)): radial and tangential velocity
        components, in m/s. Their sum equals (vx, vy).
    """
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    e_r = (sin_t, -cos_t)
    v_dot_er = vx * e_r[0] + vy * e_r[1]
    v_radial = (v_dot_er * e_r[0], v_dot_er * e_r[1])
    v_tangential = (vx - v_radial[0], vy - v_radial[1])
    return v_radial, v_tangential


def tangential_velocity_to_angular_rate(vx_t: float, vy_t: float, theta: float, ell: float) -> float:
    r"""Convert a post-capture tangential velocity to angular rate omega.

    Parameters
    ----------
    vx_t, vy_t : float
        Tangential velocity component, in m/s (radial component already
        removed).
    theta : float
        Swing angle from the downward vertical, in radians.
    ell : float
        Web length, in meters. Must be finite and strictly positive.

    Returns
    -------
    float
        Angular rate omega = theta_dot, in rad/s.

    Raises
    ------
    NonPositiveWebLengthError
        If `ell` is non-positive or non-finite.
    """
    if not (math.isfinite(ell) and ell > _MIN_WEB_LENGTH_M):
        raise NonPositiveWebLengthError(ell)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    return (vx_t * cos_t + vy_t * sin_t) / ell


def attachment_energy_loss(vx: float, vy: float, vx_t: float, vy_t: float, mass: float) -> float:
    r"""Evaluate the kinetic-energy loss from removing the radial velocity.

    Delta_E = (1/2) m (|v|^2 - |v_t|^2), which equals (1/2) m |v_r|^2 by the
    Pythagorean identity and is therefore always nonnegative.

    Parameters
    ----------
    vx, vy : float
        Pre-attachment Cartesian velocity, in m/s.
    vx_t, vy_t : float
        Post-attachment (tangential-only) Cartesian velocity, in m/s.
    mass : float
        Point mass, in kilograms.

    Returns
    -------
    float
        Kinetic-energy loss, in joules. Always >= 0.
    """
    v_sq = vx * vx + vy * vy
    v_t_sq = vx_t * vx_t + vy_t * vy_t
    return 0.5 * mass * (v_sq - v_t_sq)


@dataclass(frozen=True)
class AttachmentResult:
    """Outcome of a successful attachment transition.

    Parameters
    ----------
    z : np.ndarray, shape (4,)
        Post-attachment attached state [theta, omega, ell, nu]. `nu` is
        always 0.0 immediately after capture, since the radial velocity was
        just removed.
    energy_loss : float
        Kinetic-energy loss from the capture, in joules. Always >= 0. This
        value must be surfaced by callers, not discarded, per the
        requirement to not conceal attachment energy loss.
    x_anchor, y_anchor : float
        Anchor position used for this attachment, in meters.
    """

    z: np.ndarray
    energy_loss: float
    x_anchor: float
    y_anchor: float


def attach_to_anchor(
    x: float,
    y: float,
    vx: float,
    vy: float,
    x_anchor: float,
    y_anchor: float,
    params: PhysicalParameters,
    max_attachment_range: float | None = None,
) -> AttachmentResult:
    r"""Attach a ballistic Cartesian state to a candidate anchor.

    Applies the documented default capture model (radial velocity removed,
    tangential velocity retained) and reports the resulting attached state
    together with the kinetic-energy loss.

    This function checks only the maximum-attachment-range and degenerate-
    geometry rejection conditions from CLAUDE.md's Attachment Transition
    section. Line-of-sight/building-intersection, impulse-limit, and
    resulting-state constraint (tension, acceleration) rejections are not
    implemented here; see the module docstring.

    Parameters
    ----------
    x, y : float
        Pre-attachment body position, in meters.
    vx, vy : float
        Pre-attachment Cartesian velocity, in m/s.
    x_anchor, y_anchor : float
        Candidate anchor position, in meters.
    params : PhysicalParameters
        Supplies the mass used in the energy-loss calculation.
    max_attachment_range : float or None, optional
        Maximum permitted web length, in meters. If None, no range check is
        performed.

    Returns
    -------
    AttachmentResult
        The post-attachment state, energy loss, and anchor position.

    Raises
    ------
    NonPositiveWebLengthError
        If the position coincides with the anchor (degenerate geometry).
    AttachmentRangeExceededError
        If `max_attachment_range` is given and the computed web length
        exceeds it.
    """
    ell, theta = web_length_and_angle(x, y, x_anchor, y_anchor)
    if max_attachment_range is not None and ell > max_attachment_range:
        raise AttachmentRangeExceededError(ell, max_attachment_range)

    _, (vx_t, vy_t) = decompose_velocity(vx, vy, theta)
    omega = tangential_velocity_to_angular_rate(vx_t, vy_t, theta, ell)
    energy_loss = attachment_energy_loss(vx, vy, vx_t, vy_t, params.mass)

    z = np.array([theta, omega, ell, 0.0], dtype=float)
    return AttachmentResult(z=z, energy_loss=energy_loss, x_anchor=x_anchor, y_anchor=y_anchor)
