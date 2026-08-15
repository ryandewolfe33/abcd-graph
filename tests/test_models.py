import numpy as np
import pytest
import scipy.sparse as sp
from numpy.typing import NDArray

from abcd_graph.models import configuration_model, rewire


def assert_no_bad_edges(edges: NDArray[np.uint32]):
    n = np.max(edges) + 1
    adjacency_matrix = sp.coo_array(
        (np.ones(edges.shape[0], dtype=np.int32), edges.T),
        shape=(n, n),
    )
    assert np.all(adjacency_matrix.diagonal() == 0)
    assert np.all(adjacency_matrix.data == 1)


def test_configuration_model():
    rng = np.random.default_rng(seed=1)
    node_ids = np.arange(32, dtype=np.uint32)
    degrees = np.full(32, 4, dtype=np.int64)

    edges = configuration_model(node_ids, degrees, rng)

    ids, counts = np.unique(edges, return_counts=True)
    assert set(ids) == set(node_ids)
    assert np.all(counts == 4)
    assert edges.dtype == np.uint32


def test_rewire_loops():
    rng = np.random.default_rng(seed=1)
    edges = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 4],
            [5, 6],
        ],
        dtype=np.uint32,
    )

    edges = rewire(edges, rng)

    assert edges.shape == (5, 2)
    assert edges.dtype == np.uint32
    assert_no_bad_edges(edges)


def test_rewire_multiedges():
    rng = np.random.default_rng(seed=1)
    edges = np.array(
        [
            [0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
        ],
        dtype=np.uint32,
    )

    edges = rewire(edges, rng)

    assert edges.shape == (4, 2)
    assert edges.dtype == np.uint32
    assert_no_bad_edges(edges)


def test_rewire_swap_two_bad_edges():
    rng = np.random.default_rng(seed=1)
    edges = np.array([[0, 0], [1, 2], [1, 2]], dtype=np.uint32)

    edges = rewire(edges, rng)

    assert edges.shape == (3, 2)
    assert edges.dtype == np.uint32
    assert_no_bad_edges(edges)


def test_rewire_many():
    rng = np.random.default_rng(seed=1)
    edges = np.array(
        [[0, 0], [0, 1], [0, 1], [2, 3], [2, 3], [3, 3], [4, 5], [5, 6]],
        dtype=np.uint32,
    )

    edges = rewire(edges, rng)

    assert edges.shape == (8, 2)
    assert edges.dtype == np.uint32
    assert_no_bad_edges(edges)


def test_rewire_failure():
    rng = np.random.default_rng(seed=1)
    edges = np.array([[0, 0], [0, 1]], dtype=np.uint32)

    edges = rewire(edges, rng)

    assert edges.shape == (2, 2)
    assert edges.dtype == np.uint32
    with pytest.raises(AssertionError):
        assert_no_bad_edges(edges)


def test_rewire_drop_collisions():
    rng = np.random.default_rng(seed=1)
    edges = np.array([[0, 0], [0, 1]], dtype=np.uint32)

    edges = rewire(edges, rng, drop_collisions=True)

    assert edges.shape == (1, 2)
    assert edges.dtype == np.uint32
    assert_no_bad_edges(edges)
