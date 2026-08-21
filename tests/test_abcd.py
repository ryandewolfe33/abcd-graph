import numpy as np
import numpy.testing as npt
import pytest
from utils import assert_no_bad_edges

from abcd_graph import ABCD


@pytest.mark.parametrize("n", [100, 200])
def test_abcd(n):
    rng = np.random.default_rng(seed=1)
    abcd = ABCD(n, rng=rng)
    edges, memberships = abcd.sample()

    assert_no_bad_edges(edges)
    assert np.max(edges) == n - 1
    assert memberships.shape[1] == n


@pytest.mark.parametrize("n", [100, 200])
def test_seed(n):
    rng1 = np.random.default_rng(seed=1)
    rng2 = np.random.default_rng(seed=1)

    abcd = ABCD(n, rng=rng1)
    edges1, memberships1 = abcd.sample()

    # reset seed
    abcd.rng = rng2
    edges2, memberships2 = abcd.sample()

    assert_no_bad_edges(edges1)
    assert_no_bad_edges(edges2)
    npt.assert_array_equal(edges1, edges2)
    npt.assert_array_equal(memberships1.toarray(), memberships2.toarray())
