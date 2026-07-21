"""Domain-specific exception hierarchy for invalid physical states.

Library code raises these instead of letting invalid physical configurations
(e.g. a non-positive web length, which makes the 1/l terms in the swing
equations singular) propagate as NaNs or opaque ValueErrors.
"""

from __future__ import annotations


class WebSwingError(Exception):
    """Base class for all domain-specific errors raised by the webswing package."""


class InvalidPhysicalParameterError(WebSwingError):
    """Raised when a physical parameter (mass, gravity, ...) is outside its valid domain."""


class NonPositiveWebLengthError(WebSwingError):
    """Raised when a web length is non-positive or non-finite.

    The coordinate mapping x = x_a + l*sin(theta), y = y_a - l*cos(theta) and
    the 1/l terms in the swing equations of motion are undefined at l <= 0.
    """

    def __init__(self, ell: float) -> None:
        super().__init__(f"web length must be finite and strictly positive, got ell={ell!r}")
        self.ell = ell


class AttachmentRangeExceededError(WebSwingError):
    """Raised when a candidate anchor lies beyond the maximum attachment range."""

    def __init__(self, ell: float, max_range: float) -> None:
        super().__init__(
            f"web length {ell!r} exceeds maximum attachment range {max_range!r}"
        )
        self.ell = ell
        self.max_range = max_range


class NonFiniteStateError(WebSwingError):
    """Raised when an integrator state contains a NaN or infinite component.

    Zero-crossing event detection cannot observe non-finite values (any
    comparison involving NaN is False), so this check must be applied
    directly to sampled or output states rather than expressed as a
    `solve_ivp` event function.
    """

    def __init__(self, state: object, label: str = "state") -> None:
        super().__init__(f"{label} contains a non-finite component: {state!r}")
        self.state = state
        self.label = label


class InvalidGeometryError(WebSwingError):
    """Raised when urban geometry (a building polygon, region, or city) is invalid.

    Covers structurally invalid polygons (too few vertices, non-finite or
    degenerate), descriptive fields inconsistent with the polygon they
    describe (width, height, roof elevation), and invalid region or city
    definitions (non-positive extents, duplicate building identifiers).
    """
