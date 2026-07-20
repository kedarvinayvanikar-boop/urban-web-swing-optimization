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
