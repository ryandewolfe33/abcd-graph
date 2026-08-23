import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp
from utils import assert_no_bad_edges

from abcd_graph import ABCD
from abcd_graph.abcd_sample import ABCDSample


@pytest.mark.parametrize("n", [100, 200])
def test_abcd(n):
    rng = np.random.default_rng(seed=1)
    abcd = ABCD(n, rng=rng)
    sample = abcd.sample()

    assert_no_bad_edges(sample.edges)
    assert np.max(sample.edges) == n - 1
    assert sample.membership_matrix.shape[1] == n


def test_abcd_raises_large_n():
    abcd = ABCD(np.iinfo(np.uint32).max + 1)
    with pytest.raises(ValueError):
        abcd.sample()


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


@pytest.mark.filterwarnings("ignore::UserWarning:powerlaw*")
@pytest.mark.filterwarnings("ignore::RuntimeWarning:powerlaw*")
def test_fit():
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=np.uint32)
    coms = sp.csr_array([[1, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    sample = ABCDSample(edges, coms)
    abcd = ABCD(100)

    abcd.fit(sample)

    assert abcd.n == 5
    assert abcd.xi == 0.4
    assert abcd.eta == 1.25
    assert abcd.min_degree == 2
    assert abcd.max_degree == 2
    assert abcd.degree_exponent != 2.5
    assert abcd.min_community_size == 2
    assert abcd.max_community_size == 3
    assert abcd.community_size_exponent != 1.5
    assert abcd.degree_sequence is None
    assert abcd.community_size_sequence is None
