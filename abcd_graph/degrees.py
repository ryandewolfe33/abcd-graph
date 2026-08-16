import numpy as np
import scipy.sparse as sp
from numba import njit
from numpy.random import Generator
from numpy.typing import NDArray


@njit
def _split_community_degree(
    community_degrees: NDArray[np.uint32],
    background_degrees: NDArray[np.uint32],
    indptr: NDArray[np.uint64],
    data: NDArray[np.uint32],
    rng: Generator,
) -> None:
    for i in range(len(community_degrees)):
        n_coms = indptr[i + 1] - indptr[i]
        if n_coms == 0:
            background_degrees[i] += community_degrees[i]
            continue
        min_degree = int(community_degrees[i] / n_coms)
        data[indptr[i] : indptr[i + 1]] = min_degree
        n_to_add = community_degrees[i] - min_degree * n_coms
        if n_to_add > 0:
            random_numbers = rng.uniform(size=n_coms)
            add_indices = np.argsort(random_numbers)[:n_to_add]
            add_indices += indptr[i]
            for j in add_indices:
                data[j] += 1


def split_degrees(
    degrees: NDArray[np.uint32],
    membership_matrix: sp.csr_array,
    xi: float,
    rng: Generator,
) -> (sp.csr_array, NDArray[np.uint32]):
    """Split degrees into community degrees and background degrees. The fraction of
    degree in the background is, on expectation, xi. Community degrees will be split
    evenly among communities if the nodes belongs to more than one.

    Parameters
    ----------
    degrees: NDArray
        Array of degrees.

    membership_matrix: sp.csr_array
        n_communities x n_nodes sparse matrix of memberships. A one at index i,j means
        node j is in community i. Outlier nodes have no membership.

    xi: float
        Proportion of degrees assigned to the background graph.

    rng: Generator
        numpy.random.Generator for a source of randomness.

    Returns
    -------
    community_degrees: sp.csr_array
        n_communities x n_nodes sparse matrix of community degrees. The value at index i,j
        is the degree of node j in community i. Has the same non-zero entries as the
        membership_matrix.

    background_degrees: NDArray[np.uint32]
        Array for the degree of each node in the background graph.
    """
    background_degrees = degrees * xi
    background_degrees += rng.uniform(size=len(degrees))
    background_degrees = background_degrees.astype(np.uint32)

    community_degrees = membership_matrix.copy().astype(np.uint32)
    community_degrees = community_degrees.tocsc()
    _split_community_degree(
        degrees - background_degrees,
        background_degrees,
        community_degrees.indptr,
        community_degrees.data,
        rng,
    )

    return community_degrees.tocsr(), background_degrees
