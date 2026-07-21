"""Tests for the typed constraint/domain configuration containers.

Validates successful construction and every individual rejection branch of
`SwingConstraints` and `BallisticDomain`, including the cross-field ordering
checks (ell_max > ell_min, x_max > x_min).
"""

from __future__ import annotations

import pytest

from webswing.config import BallisticDomain, SwingConstraints
from webswing.exceptions import InvalidPhysicalParameterError


def make_swing_constraints(**overrides: float) -> SwingConstraints:
    defaults = dict(tension_max=500.0, load_factor_max=4.0, ell_min=1.0, ell_max=20.0, nu_max=5.0)
    defaults.update(overrides)
    return SwingConstraints(**defaults)


def make_ballistic_domain(**overrides: float) -> BallisticDomain:
    defaults = dict(x_min=-100.0, x_max=100.0, y_max=200.0, max_duration=30.0)
    defaults.update(overrides)
    return BallisticDomain(**defaults)


def test_swing_constraints_accepts_valid_values() -> None:
    c = make_swing_constraints()
    assert c.tension_max == 500.0
    assert c.ell_max > c.ell_min


@pytest.mark.parametrize(
    "field", ["tension_max", "load_factor_max", "ell_min", "ell_max", "nu_max"]
)
@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_swing_constraints_rejects_nonpositive_or_nonfinite_fields(field: str, bad_value: float) -> None:
    with pytest.raises(InvalidPhysicalParameterError):
        make_swing_constraints(**{field: bad_value})


def test_swing_constraints_rejects_ell_max_not_greater_than_ell_min() -> None:
    with pytest.raises(InvalidPhysicalParameterError):
        make_swing_constraints(ell_min=5.0, ell_max=5.0)
    with pytest.raises(InvalidPhysicalParameterError):
        make_swing_constraints(ell_min=5.0, ell_max=4.0)


def test_ballistic_domain_accepts_valid_values() -> None:
    d = make_ballistic_domain()
    assert d.x_max > d.x_min
    assert d.max_duration == 30.0


@pytest.mark.parametrize("field", ["x_min", "x_max", "y_max", "max_duration"])
def test_ballistic_domain_rejects_nonfinite_fields(field: str) -> None:
    for bad_value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(InvalidPhysicalParameterError):
            make_ballistic_domain(**{field: bad_value})


def test_ballistic_domain_rejects_nonpositive_y_max() -> None:
    with pytest.raises(InvalidPhysicalParameterError):
        make_ballistic_domain(y_max=0.0)
    with pytest.raises(InvalidPhysicalParameterError):
        make_ballistic_domain(y_max=-10.0)


def test_ballistic_domain_rejects_nonpositive_max_duration() -> None:
    with pytest.raises(InvalidPhysicalParameterError):
        make_ballistic_domain(max_duration=0.0)


def test_ballistic_domain_rejects_x_max_not_greater_than_x_min() -> None:
    with pytest.raises(InvalidPhysicalParameterError):
        make_ballistic_domain(x_min=10.0, x_max=10.0)
    with pytest.raises(InvalidPhysicalParameterError):
        make_ballistic_domain(x_min=10.0, x_max=5.0)
