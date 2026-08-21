from typing import Any, Protocol

import numpy as np
from numba import njit
from numba.typed import List, Set
from numba.types import uint64
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
    node_ids = np.arange(len(degrees), dtype=np.uint32)
    stubs = np.repeat(node_ids, degrees)
    rng.shuffle(stubs)
    edges = stubs.reshape(-1, 2)
    return edges


@njit(inline="always")
def make_edge_id(
    edge: NDArray[np.uint32],
) -> uint64:
    high, low = edge[0], edge[1]
    if high < low:
        high, low = low, high
    return (uint64(high) << 32) | uint64(low)


@njit(inline="always")
def edge_from_id(
    edge_id: uint64,
) -> NDArray[np.uint32]:
    edge = np.empty(2, dtype=np.uint32)
    edge[0] = np.uint32(edge_id >> 32)
    edge[1] = np.uint32(edge_id & 0xFFFFFFFF)
    return edge


@njit(inline="always")
def swap(
    edge1: NDArray[np.uint32],
    edge2: NDArray[np.uint32],
    rng: Generator,
):
    edge1 = edge1.copy()
    edge2 = edge2.copy()
    if rng.uniform() > 0.5:
        edge1[0], edge2[0] = edge2[0], edge1[0]
    else:
        edge1[0], edge2[1] = edge2[1], edge1[0]
    return edge1, edge2


@njit(inline="always")
def is_bad_swap(
    edge1: NDArray[np.uint32],
    edge2: NDArray[np.uint32],
    good_edges: Set[uint64],
) -> bool:
    if edge1[0] == edge1[1] or edge2[0] == edge2[1]:
        return True
    edge1_id = make_edge_id(edge1)
    edge2_id = make_edge_id(edge2)
    if edge1_id == edge2_id:
        return True
    if edge1_id in good_edges or edge2_id in good_edges:
        return True
    return False


@njit(nogil=True)
def rewire(
    edges: NDArray[np.uint32],
    rng: Generator,
    max_swap_attempts_per_bad_edge: int = 5,
) -> int:
    """Perform inplace edge swaps to resolve loops and multi-edges.

    Parameters
    ----------

    edges: NDArray[np.uint32]
        Edge list with shape (n_edges, 2) to be rewired. Will be altered in place.

    rng: Generator
        numpy.random.Generator object to use for randomness.

    max_swap_attempts_per_bad_edge: int, default=5
        Cap the attempted edge swaps to this values times the number of initial bad edges.

    """
    # Move good edges to the front, add their hashes to a set, and
    # make a List-backed-queue of bad edges
    good_edges = Set.empty(uint64)
    bad_queue = List.empty_list(uint64)
    n_good_edges = 0
    for i in range(edges.shape[0]):
        edge = edges[i]
        edge_id = make_edge_id(edge)
        if edge[0] != edge[1] and edge_id not in good_edges:
            good_edges.add(edge_id)
            edges[n_good_edges] = edge
            n_good_edges += 1
        else:
            bad_queue.append(edge_id)

    # Try to resolve the bad edge at the head of bad_queue by trying a swap
    # with a random edge (good or bad, but not the one we are trying to resolve)
    # If the swap would cause a collision, move bad edge to the back of the
    # queue. Repeat until the Queue is empty or we give up.

    queue_head = 0
    queue_tail = len(bad_queue)
    n_bad_edges = queue_tail
    for _ in range(queue_tail * max_swap_attempts_per_bad_edge):
        if n_bad_edges == 0:
            break
        bad_edge = edge_from_id(bad_queue[queue_head])
        choose_from_good_edges = rng.uniform() < n_good_edges / (
            n_good_edges + n_bad_edges - 1
        )
        if choose_from_good_edges:
            swap_index = rng.integers(0, n_good_edges)
            swap_candidate_edge = edges[swap_index]
            new_edge1, new_edge2 = swap(bad_edge, swap_candidate_edge, rng)
            if not is_bad_swap(new_edge1, new_edge2, good_edges):
                edges[swap_index] = new_edge1
                edges[n_good_edges] = new_edge2
                n_good_edges += 1
                good_edges.add(make_edge_id(new_edge1))
                good_edges.add(make_edge_id(new_edge2))
                good_edges.discard(make_edge_id(swap_candidate_edge))
                queue_head = (queue_head + 1) % len(bad_queue)
                n_bad_edges -= 1
            else:
                queue_head, queue_tail = (queue_head + 1) % len(bad_queue), queue_tail
        else:
            swap_offset = rng.integers(1, n_bad_edges)  # don't choose current head
            swap_index = (queue_head + swap_offset) % len(bad_queue)
            swap_candidate_edge_id = bad_queue[swap_index]
            swap_candidate_edge = edge_from_id(swap_candidate_edge_id)
            new_edge1, new_edge2 = swap(bad_edge, swap_candidate_edge, rng)
            if not is_bad_swap(new_edge1, new_edge2, good_edges):
                edges[n_good_edges] = new_edge1
                n_good_edges += 1
                edges[n_good_edges] = new_edge2
                good_edges.add(make_edge_id(new_edge1))
                good_edges.add(make_edge_id(new_edge2))
                n_good_edges += 1
                bad_queue.pop(swap_index)
                queue_head = (queue_head + 1) % len(bad_queue)
                n_bad_edges -= 2
            else:
                queue_head, queue_tail = (queue_head + 1) % len(bad_queue), queue_tail

    # Write bad edges that failed to swap back into the edge list
    for i in range(n_bad_edges):
        bad_edge_id = bad_queue[(queue_head + 1) % len(bad_queue)]
        bad_edge = edge_from_id(bad_edge_id)
        edges[n_good_edges + i] = bad_edge

    return n_good_edges
