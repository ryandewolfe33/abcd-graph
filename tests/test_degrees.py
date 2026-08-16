import numpy as np
import pytest
import scipy.sparse as sp

from abcd_graph.degrees import split_degrees


@pytest.mark.parametrize("xi", [0.2, 0.4])
def test_split_degrees(xi):
    degrees = np.array([10, 5, 3])
    membership_matrix = sp.csr_array([[1, 0, 0], [1, 1, 0]])
    rng = np.random.default_rng(seed=1)

    community_degrees, background_degrees = split_degrees(
        degrees,
        membership_matrix,
        xi,
        rng,
    )

    assert community_degrees.dtype == np.uint32
    assert community_degrees.shape == membership_matrix.shape
    assert len(background_degrees) == len(degrees)
    assert background_degrees.dtype == np.uint32
    assert np.all(community_degrees.sum(axis=0) + background_degrees == degrees)
    is_outlier = membership_matrix.sum(axis=0) == 0
    assert np.all(np.abs(background_degrees - degrees * xi)[~is_outlier] < 1)
    assert np.all(
        np.abs(community_degrees.sum(axis=0) - degrees * (1 - xi))[~is_outlier] < 1
    )
    assert np.all(background_degrees[is_outlier] == degrees[is_outlier])
