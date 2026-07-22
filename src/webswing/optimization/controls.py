r"""Radial control profiles u(t) = l_ddot for the controlled variable-length pendulum.

Implements the control-interface requirement from CLAUDE.md's Attached
Swinging Dynamics section: "at least: constant radial acceleration,
piecewise-constant radial acceleration, and a bounded parameterized control
profile suitable for optimization." Each type here is a `RadialControl`
(the `Callable[[float], float]` alias from `dynamics.swing`) that can be
passed directly to `swing_state_derivative` or any `dynamics.events` swing
event, so a control constructed here integrates without adaptation.

`equal_interval_control` is the bridge to `optimization.local_transfer`: it
maps a flat decision vector (the optimizer's parameter array) directly into
a bounded piecewise-constant control over N equal-width intervals spanning
a fixed transfer duration, so the optimizer's box constraints
[u_min, u_max]^N are exactly this module's own bound validation.

Boundary evaluation policy
----------------------------
`PiecewiseConstantControl` evaluates for any real t, extrapolating by
holding the first segment's value for t before the first breakpoint and the
last segment's value for t at or after the last breakpoint, rather than
raising. This tolerates the small floating-point overshoot an integrator's
adaptive step or event refinement can produce at the ends of the transfer
interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from webswing.exceptions import InvalidControlProfileError


def _validate_bounds(u_min: float, u_max: float) -> None:
    if not (math.isfinite(u_min) and math.isfinite(u_max)):
        raise InvalidControlProfileError(
            f"u_min and u_max must be finite, got u_min={u_min!r}, u_max={u_max!r}"
        )
    if not (u_max > u_min):
        raise InvalidControlProfileError(
            f"u_max ({u_max!r}) must be strictly greater than u_min ({u_min!r})"
        )


@dataclass(frozen=True)
class ConstantControl:
    """A constant radial acceleration u(t) = value, for all t.

    Parameters
    ----------
    value : float
        Commanded radial acceleration, in m/s^2. Must be finite and lie
        within [u_min, u_max].
    u_min, u_max : float
        Control bounds, in m/s^2. Must be finite with `u_max > u_min`.

    Raises
    ------
    InvalidControlProfileError
        If the bounds are invalid or `value` lies outside them.
    """

    value: float
    u_min: float
    u_max: float

    def __post_init__(self) -> None:
        _validate_bounds(self.u_min, self.u_max)
        if not math.isfinite(self.value):
            raise InvalidControlProfileError(f"value must be finite, got {self.value!r}")
        if not (self.u_min <= self.value <= self.u_max):
            raise InvalidControlProfileError(
                f"value {self.value!r} is outside bounds [{self.u_min!r}, {self.u_max!r}]"
            )

    def __call__(self, t: float) -> float:
        return self.value


@dataclass(frozen=True)
class PiecewiseConstantControl:
    """A piecewise-constant radial acceleration profile over N segments.

    Segment i covers `[breakpoints[i], breakpoints[i + 1])` with value
    `values[i]`, for i = 0, ..., N - 1.

    Parameters
    ----------
    breakpoints : np.ndarray, shape (N + 1,)
        Strictly increasing segment boundary times, in seconds.
    values : np.ndarray, shape (N,)
        Constant radial acceleration for each segment, in m/s^2. Each must
        lie within [u_min, u_max].
    u_min, u_max : float
        Control bounds, in m/s^2. Must be finite with `u_max > u_min`.

    Raises
    ------
    InvalidControlProfileError
        If the bounds are invalid, `breakpoints` is not finite and strictly
        increasing, the segment/value counts are inconsistent, or any value
        lies outside [u_min, u_max].
    """

    breakpoints: np.ndarray
    values: np.ndarray
    u_min: float
    u_max: float

    def __post_init__(self) -> None:
        _validate_bounds(self.u_min, self.u_max)

        breakpoints = np.asarray(self.breakpoints, dtype=float)
        values = np.asarray(self.values, dtype=float)

        if breakpoints.ndim != 1 or breakpoints.size < 2:
            raise InvalidControlProfileError(
                f"breakpoints must be 1-D with at least 2 entries, got shape {breakpoints.shape}"
            )
        if not np.all(np.isfinite(breakpoints)):
            raise InvalidControlProfileError("breakpoints must be finite")
        if not np.all(np.diff(breakpoints) > 0.0):
            raise InvalidControlProfileError("breakpoints must be strictly increasing")

        if values.ndim != 1 or values.size != breakpoints.size - 1:
            raise InvalidControlProfileError(
                f"values must be 1-D with length len(breakpoints) - 1 "
                f"({breakpoints.size - 1}), got shape {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise InvalidControlProfileError("values must be finite")
        if not np.all((values >= self.u_min) & (values <= self.u_max)):
            raise InvalidControlProfileError(
                f"all values must lie within [{self.u_min!r}, {self.u_max!r}], got {values!r}"
            )

        breakpoints.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "breakpoints", breakpoints)
        object.__setattr__(self, "values", values)

    def __call__(self, t: float) -> float:
        idx = int(np.searchsorted(self.breakpoints, t, side="right")) - 1
        idx = min(max(idx, 0), self.values.size - 1)
        return float(self.values[idx])


def equal_interval_control(
    parameters: np.ndarray, duration: float, u_min: float, u_max: float
) -> PiecewiseConstantControl:
    """Build a piecewise-constant control over N equal-width intervals spanning [0, duration].

    Maps an optimizer's flat decision vector directly into a bounded
    `RadialControl`: `parameters[i]` is the constant radial acceleration on
    the i-th of N equal-width intervals.

    Parameters
    ----------
    parameters : np.ndarray, shape (N,)
        Per-interval radial acceleration values, in m/s^2. N >= 1.
    duration : float
        Total transfer duration, in seconds. Must be finite and strictly
        positive.
    u_min, u_max : float
        Control bounds, in m/s^2. Must be finite with `u_max > u_min`.

    Returns
    -------
    PiecewiseConstantControl
        Control profile with breakpoints `linspace(0, duration, N + 1)` and
        `values = parameters`.

    Raises
    ------
    InvalidControlProfileError
        If `duration` is not finite and strictly positive, `parameters` is
        not a non-empty 1-D array, or any bound/value check in
        `PiecewiseConstantControl` fails.
    """
    if not (math.isfinite(duration) and duration > 0.0):
        raise InvalidControlProfileError(
            f"duration must be finite and strictly positive, got {duration!r}"
        )
    params = np.asarray(parameters, dtype=float)
    if params.ndim != 1 or params.size < 1:
        raise InvalidControlProfileError(
            f"parameters must be a non-empty 1-D array, got shape {params.shape}"
        )
    breakpoints = np.linspace(0.0, duration, params.size + 1)
    return PiecewiseConstantControl(breakpoints=breakpoints, values=params, u_min=u_min, u_max=u_max)
