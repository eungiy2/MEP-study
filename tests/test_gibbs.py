"""Unit tests for gibbs -- Gibbs weights / free energy, Eq. (25)-(26)."""

import numpy as np
import pytest

from drying.optimization.gibbs import free_energy, gibbs_probs

COSTS = np.array([10.0, 20.0, 30.0, 40.0])
BEST = 0  # index of the minimum-cost path


def test_probs_normalize():
    for beta in (1e-6, 0.01, 1.0, 100.0):
        p = gibbs_probs(beta, COSTS)
        assert np.isclose(p.sum(), 1.0)
        assert (p >= 0).all()


def test_probs_uniform_as_beta_to_zero():
    p = gibbs_probs(1e-9, COSTS)
    assert np.allclose(p, 1.0 / len(COSTS), atol=1e-3)


def test_probs_concentrate_as_beta_large():
    p_small = gibbs_probs(0.01, COSTS)
    p_large = gibbs_probs(100.0, COSTS)
    assert p_large[BEST] > p_small[BEST]
    assert p_large[BEST] > 0.99       # threshold: effectively 1 for very large beta (Eq. 28)


def test_free_energy_bounds():
    # F* = min D - (1/beta) log( sum exp(-beta(D - minD)) ) <= min D
    for beta in (0.01, 1.0, 100.0):
        f = free_energy(beta, COSTS)
        assert f <= COSTS.min() + 1e-9
    # as beta -> inf, F* -> min D
    assert free_energy(1e6, COSTS) == pytest.approx(COSTS.min(), rel=1e-6)
    # as beta -> 0:  F* -> mean(D) - (1/beta)*ln N   (entropy term -(1/beta) ln N diverges)
    n = len(COSTS)
    assert free_energy(1e-6, COSTS) == pytest.approx(COSTS.mean() - np.log(n) / 1e-6, rel=1e-3)