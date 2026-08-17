import numpy as np
import pytest
import scipy.sparse as sp

from abcd_graph.degrees import (
    assign_degrees,
    split_degrees,
)


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
    assert np.all(background_degrees[is_outlier] == degrees[is_outlier])
    assert np.all(community_degrees.sum(axis=1) % 2 == 0)


def test_assign_degrees_no_overlap():
    degrees = np.concatenate(
        (np.full(16, 4, dtype=np.uint32), np.full(16, 3, dtype=np.uint32))
    )
    # Membership matrix with 4 communities size 7 and 4 outliers
    indptr = np.arange(5, dtype=np.uint32) * 7
    indices = np.arange(28, dtype=np.uint32)
    data = np.ones(28, dtype=np.bool)
    membership_matrix = sp.csr_array((data, indices, indptr), shape=(4, 32))
    xi = 0.2
    rng = np.random.default_rng(seed=1)

    assigned_degrees = assign_degrees(
        degrees,
        membership_matrix,
        xi,
        rng,
    )

    assert assigned_degrees.shape == degrees.shape
    assert assigned_degrees.dtype == np.uint32
    assert np.sum(assigned_degrees == 4) == 16
    assert np.sum(assigned_degrees == 3) == 16


def test_assign_degrees_overlap():
    degrees = np.concatenate(
        (np.full(16, 4, dtype=np.uint32), np.full(16, 3, dtype=np.uint32))
    )
    # Membership matrix with 4 communities size 7 and 4 outliers
    indptr = np.arange(6, dtype=np.uint32) * 7
    indices = np.concatenate((np.arange(28, dtype=np.uint32), np.arange(7) * 4))
    data = np.ones(35, dtype=np.bool)
    membership_matrix = sp.csr_array((data, indices, indptr), shape=(5, 32))
    xi = 0.2
    rng = np.random.default_rng(seed=1)

    assigned_degrees = assign_degrees(
        degrees,
        membership_matrix,
        xi,
        rng,
    )

    assert assigned_degrees.shape == degrees.shape
    assert assigned_degrees.dtype == np.uint32
    assert np.sum(assigned_degrees == 4) == 16
    assert np.sum(assigned_degrees == 3) == 16


@pytest.mark.parametrize("xi", [0.2, 0.4])
@pytest.mark.parametrize("rho", [-0.7, -0.2, 0.2, 0.7])
def test_assign_degrees_overlap_with_rho(xi, rho):
    degrees = np.concatenate(
        (np.full(16, 4, dtype=np.uint32), np.full(16, 3, dtype=np.uint32))
    )
    # Membership matrix with 4 communities size 7 and 4 outliers
    indptr = np.arange(6, dtype=np.uint32) * 7
    indices = np.concatenate((np.arange(28, dtype=np.uint32), np.arange(7) * 4))
    data = np.ones(35, dtype=np.bool)
    membership_matrix = sp.csr_array((data, indices, indptr), shape=(5, 32))
    rng = np.random.default_rng(seed=1)

    assigned_degrees = assign_degrees(
        degrees,
        membership_matrix,
        xi,
        rng,
        rho,
    )

    assert assigned_degrees.shape == degrees.shape
    assert assigned_degrees.dtype == np.uint32
    assert np.sum(assigned_degrees == 4) == 16
    assert np.sum(assigned_degrees == 3) == 16
    n_coms = membership_matrix.sum(axis=0)
    empirical_rho = np.corrcoef(assigned_degrees, n_coms)[0, 1]
    assert np.sign(empirical_rho) == np.sign(rho)
