import numpy as np
import numpy.testing as npt
import pytest
from utils import assert_no_bad_edges

from abcd_graph import ABCD


@pytest.mark.parametrize("n", [100, 200])
def test_abcd(n):
    rng = np.random.default_rng(seed=1)
    abcd = ABCD(n, rng=rng)
    sample = abcd.sample()

    assert_no_bad_edges(sample.edges)
    assert np.max(sample.edges) == n - 1
    assert sample.membership_matrix.shape[1] == n


@pytest.mark.parametrize("n", [100, 200])
def test_seed(n):
    rng1 = np.random.default_rng(seed=1)
    rng2 = np.random.default_rng(seed=1)

    abcd = ABCD(n, rng=rng1)
    sample1 = abcd.sample()

    # reset seed
    abcd.rng = rng2
    sample2 = abcd.sample()

    assert_no_bad_edges(sample1.edges)
    assert_no_bad_edges(sample2.edges)
    npt.assert_array_equal(sample1.edges, sample2.edges)
    npt.assert_array_equal(sample2.to_dense(), sample2.to_dense())
