from typing import Any, Protocol

import numpy as np
from numba import from_dtype, njit
from numba.typed import List, Set
from numba.types import UniTuple
from numpy.random import Generator
from numpy.typing import NDArray


# Define the function interface contract structurally
class Model(Protocol):
    def __call__(
        self, node_ids: NDArray[np.uint32], degrees: NDArray[np.uint32], rng: Generator
    ) -> NDArray[np.uint32]: ...


def chunglu_model(
    node_ids: NDArray[np.uint32], degrees: NDArray[np.integer[Any]], rng: Generator
) -> NDArray[np.uint32]:
    """Sample a random graph with, on expectation, the given degree sequence.

    Parameters
    ----------
    node_ids: NDArray[np.uint32]
        List of node_ids for the graph. The returned edge list will contain each
        node_id equal to it's degree

    degrees: NDArray[np.integer[Any]]
        List of degrees. The value degrees[i] corresponds to the degree of node node_ids[i]

    rng: Generator
        numpy.random.Generator object used for randomness.
    """
    node_probs = degrees / degrees.sum()
    edges = rng.choice(node_ids, size=np.sum(degrees), p=node_probs).reshape(-1, 2)
    return edges


@njit(nogil=True)
def configuration_model(
    node_ids: NDArray[np.uint32], degrees: NDArray[np.uint32], rng: Generator
) -> NDArray[np.uint32]:
    """Sample a random graph with the given degree sequence.

    Parameters
    ----------
    node_ids: NDArray[np.uint32]
        List of node_ids for the graph. The returned edge list will contain each
        node_id equal to it's degree

    degrees: NDArray[np.integer[Any]]
        List of degrees. The value degrees[i] corresponds to the degree of node node_ids[i]

    rng: Generator
        numpy.random.Generator object used for randomness.
    """
    stubs = np.repeat(node_ids, degrees)
    rng.shuffle(stubs)
    edges = stubs.reshape(-1, 2)
    return edges


@njit(inline="always")
def make_edge_tuple(edge: NDArray[np.integer[Any]]) -> UniTuple(np.integer[Any], 2):
    high, low = edge[0], edge[1]
    if high < low:
        high, low = low, high
    return (low, high)


@njit(inline="always")
def swap(
    edge1: UniTuple(np.integer[Any], 2),
    edge2: UniTuple(np.integer[Any], 2),
    rng: Generator,
) -> (UniTuple(np.integer[Any], 2), UniTuple(np.integer[Any], 2)):
    if rng.uniform() > 0.5:
        return (edge1[0], edge2[0]), (edge2[0], edge1[0])
    return (edge1[0], edge2[1]), (edge2[1], edge1[0])


@njit(inline="always")
def is_bad_swap(
    edge1: NDArray[np.integer[Any]],
    edge2: NDArray[np.integer[Any]],
    good_edges: Set[UniTuple(np.integer[Any], 2)],
) -> bool:
    if edge1[0] == edge1[1] or edge2[0] == edge2[1]:
        return True
    if edge1 == edge2:
        return True
    if edge1 in good_edges or edge2 in good_edges:
        return True
    return False


def get_edge_type(edges: NDArray[np.integer[Any]]) -> UniTuple(np.integer[Any], 2):
    return UniTuple(from_dtype(edges.dtype), 2)


@njit(nogil=True)
def rewire(
    edges: NDArray[np.integer[Any]],
    edge_type: UniTuple(np.integer[Any], 2),
    rng: Generator,
    max_swap_attempts_per_bad_edge: int = 5,
) -> int:
    """Perform inplace edge swaps to resolve loops and multi-edges.

    Parameters
    ----------

    edges: NDArray[np.integer[Any]]
        Edge list with shape (n_edges, 2) to be rewired. Will be altered in place.

    node_dtype: np.integer[Any]
        dtype of nodes stored in the edges list. Required for numba pre-compilation
        and is assumed to be correct. Consistency checks should be made before
        calling this function.

    rng: Generator
        numpy.random.Generator object to use for randomness.

    max_swap_attempts_per_bad_edge: int, default=5
        Cap the attempted edge swaps to this values times the number of initial bad edges.

    """
    # Move good edges to the front, add their hashes to a set, and
    # make a List-backed-queue of bad edges
    good_edges = Set.empty(edge_type)
    bad_queue = List.empty_list(edge_type)
    n_good_edges = 0
    for i in range(edges.shape[0]):
        edge = make_edge_tuple(edges[i])
        if edge[0] != edge[1] and edge not in good_edges:
            good_edges.add(edge)
            edges[n_good_edges] = edge
            n_good_edges += 1
        else:
            bad_queue.append(edge)

    # Try to resolve the bad edge at the head of bad_queue by trying a swap
    # with a random edge (good or bad, but not the one we are trying to resolve)
    # If the swap would cause a collision, move bad edge to the back of the
    # queue. Repeat until the Queue is empty or we give up.

    # Store bad edges in the indices 0:queue_len and keep at the front
    # of the list
    next_index = len(bad_queue) - 1
    queue_len = len(bad_queue)

    for _ in range(len(bad_queue) * max_swap_attempts_per_bad_edge):
        if queue_len == 0:
            break

        next_index = (next_index + 1) % queue_len
        bad_edge = bad_queue[next_index]
        choose_from_good_edges = rng.uniform() < n_good_edges / (
            n_good_edges + queue_len - 1
        )  # Always True if queue_len == 1
        if choose_from_good_edges:
            swap_index = rng.integers(0, n_good_edges)
            swap_candidate_edge = make_edge_tuple(edges[swap_index])
            new_edge1, new_edge2 = swap(bad_edge, swap_candidate_edge, rng)
            if not is_bad_swap(new_edge1, new_edge2, good_edges):
                edges[swap_index] = new_edge1
                edges[n_good_edges] = new_edge2
                n_good_edges += 1
                good_edges.add(new_edge1)
                good_edges.add(new_edge2)
                good_edges.discard(swap_candidate_edge)
                bad_queue[next_index] = bad_queue[queue_len - 1]
                queue_len -= 1
        else:
            swap_offset = rng.integers(1, queue_len)  # don't choose current head
            swap_index = (next_index + swap_offset) % queue_len
            swap_candidate_edge = bad_queue[swap_index]
            new_edge1, new_edge2 = swap(bad_edge, swap_candidate_edge, rng)
            if not is_bad_swap(new_edge1, new_edge2, good_edges):
                edges[n_good_edges] = new_edge1
                edges[n_good_edges + 1] = new_edge2
                n_good_edges += 2
                good_edges.add(new_edge1)
                good_edges.add(new_edge2)
                bad_queue[next_index] = bad_queue[queue_len - 1]
                bad_queue[swap_index] = bad_queue[queue_len - 2]
                queue_len -= 2

    # Write bad edges that failed to swap back into the edge list
    for i in range(queue_len):
        edges[n_good_edges + i] = bad_queue[i]

    return n_good_edges
