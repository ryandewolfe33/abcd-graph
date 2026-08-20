import numpy as np
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
