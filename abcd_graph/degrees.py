import numpy as np
import scipy.sparse as sp
from numba import njit
from numpy.random import Generator
from numpy.typing import NDArray


@njit(cache=True)
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


@njit(cache=True)
def make_community_degree_sums_even(
    community_degrees_indptr: NDArray,
    community_degrees_indices: NDArray,
    community_degrees_data: NDArray,
    background_degrees: NDArray[np.uint32],
    rng: Generator,
):
    for com in range(len(community_degrees_indptr) - 1):
        com_members = community_degrees_indices[
            community_degrees_indptr[com] : community_degrees_indptr[com + 1]
        ]
        com_degrees = community_degrees_data[
            community_degrees_indptr[com] : community_degrees_indptr[com + 1]
        ]
        if np.sum(com_degrees.astype(np.uint64)) % 2 == 0:
            continue
        indices_of_max_degree = np.where(com_degrees == np.max(com_degrees))[0]
        decrease_index = indices_of_max_degree[
            rng.integers(0, len(indices_of_max_degree))
        ]
        community_degrees_data[community_degrees_indptr[com] + decrease_index] -= 1
        background_degrees[com_members[decrease_index]] += 1


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
    community_degrees = community_degrees.tocsr()
    make_community_degree_sums_even(
        community_degrees.indptr,
        community_degrees.indices,
        community_degrees.data,
        background_degrees,
        rng,
    )
    return community_degrees.tocsr(), background_degrees


def _assign_outlier_degrees(
    degrees: NDArray[np.uint32],
    outlier_threshold: float,
    n_outliers: int,
    rng: Generator,
) -> (NDArray[np.uint32], NDArray[np.uint32]):
    available_indices = np.where(degrees < outlier_threshold)[0]
    if len(available_indices) > n_outliers:
        chosen_indices = rng.choice(available_indices, size=n_outliers, replace=False)
    else:
        chosen_indices = np.argsort(degrees)[:n_outliers]
    outlier_degrees = degrees[chosen_indices]

    remaining_mask = np.ones_like(degrees, dtype=np.bool)
    remaining_mask[chosen_indices] = False
    remaining_degrees = degrees[remaining_mask]

    return outlier_degrees, remaining_degrees


@njit(cache=True)
def _assign_degrees(
    degrees: NDArray[np.uint32],
    n_coms: NDArray,
    thresholds: NDArray[np.floating],
    rng: Generator,
    alpha: float = 0.0,
) -> NDArray[np.uint32]:
    assigned_degrees = np.empty_like(degrees)
    open_nodes = np.arange(len(n_coms), dtype=np.uint32)[n_coms > 0]
    n_coms_exp_alpha = n_coms.astype(np.float64) ** alpha
    for d in degrees:
        allowed_indices = np.where(d <= thresholds[open_nodes])[0]
        if len(allowed_indices) == 0:
            allowed_indices = np.where(
                thresholds[open_nodes] == np.min(thresholds[open_nodes])
            )[0]
        allowed_nodes = open_nodes[allowed_indices]

        # Choose an available node proportional to n_coms ** alpha
        if len(allowed_nodes) > 1:
            probs = n_coms_exp_alpha[allowed_nodes]
            probs /= np.sum(probs)
            cum_prob = np.cumsum(probs)
            random_value = rng.uniform()
            chosen_index = np.searchsorted(cum_prob, random_value)
        else:
            chosen_index = 0
        assigned_degrees[allowed_nodes[chosen_index]] = d

        # Remove assigned node and shorten list
        open_nodes[allowed_indices[chosen_index]] = open_nodes[-1]
        open_nodes = open_nodes[:-1]

    return assigned_degrees


@njit(cache=True)
def _assign_degrees_with_alpha_search(
    degrees: NDArray[np.uint32],
    n_coms: NDArray[np.uint32],
    thresholds: NDArray[np.floating],
    rng: Generator,
    rho: float,
    rho_tol: float,
    alpha_min: float,
    alpha_max: float,
    alpha_iters: int,
):
    # We known the sign of alpha will match the sign of rho
    if rho > 0:
        max_rho_degrees = _assign_degrees(
            degrees,
            n_coms,
            thresholds,
            rng,
            alpha_max,
        )
        empirical_rho = np.corrcoef(max_rho_degrees, n_coms)[0, 1]
        if empirical_rho < rho:  # Return best try
            return max_rho_degrees
        else:  # Prep search for positive alpha
            alpha_min = 0

    elif rho < 0:
        min_rho_degrees = _assign_degrees(
            degrees,
            n_coms,
            thresholds,
            rng,
            alpha_min,
        )
        empirical_rho = np.corrcoef(min_rho_degrees, n_coms)[0, 1]
        if empirical_rho > rho:  # Return best try
            return min_rho_degrees
        else:  # Prep search for negative alpha
            alpha_max = 0

    # Default to alpha=0
    assigned_degrees = _assign_degrees(
        degrees,
        n_coms,
        thresholds,
        rng,
    )

    # Binary search for best alpha
    for _ in range(alpha_iters):
        alpha = (alpha_max + alpha_min) / 2
        assigned_degrees = _assign_degrees(
            degrees,
            n_coms,
            thresholds,
            rng,
            alpha,
        )
        empirical_rho = np.corrcoef(assigned_degrees, n_coms)[0, 1]
        if np.abs(empirical_rho - rho) < rho_tol:
            return assigned_degrees
        elif empirical_rho > rho:
            alpha_max = alpha
        else:
            alpha_min = alpha

    return assigned_degrees


def assign_degrees(
    degrees: NDArray[np.uint32],
    membership_matrix: sp.csr_array,
    xi: float,
    rng: Generator,
    rho: float = 0.0,
    rho_tol: float = 0.01,
    alpha_min: float = -60,
    alpha_max: float = 60,
    alpha_iters: int = 10,
) -> NDArray[np.uint32]:
    """Assign degrees to nodes.

    Parameters
    ----------
    degrees: NDArray[np.uint32]
        Array of available degrees in any ordering.

    membership_matrix: sp.csr_array
        n_communities x n_nodes sparse matrix for community membership.

    xi: float
        Proportion of each degree that will be assigned to the global
        background graph.

    rho: float
        Target pearson correlation between node degree and number of
        communities the node is in. Only used if there is overlap.

    rho_tol: float
        Tolerance on empirical rho to stop alpha search early.

    alpha_min: float
        Internal parameter used for correlation between degree and number of
        communities.

    alpha_max: float
        Internal parameter used for correlation between degree and number of
        communities.

    alpha_iters: int
        Number of times to try alphas in a binary search.

    rng: Generator
        Source of randomness.

    Returns
    -------
    assigned_degrees: NDArray[np.uint32]
        Array of degrees that is aligned with the membership matrix.

    """
    degrees = np.sort(degrees)[::-1]  # sort degrees descending
    community_sizes = membership_matrix.sum(axis=1).astype(
        np.uint32
    )  # size of each community
    membership_matrix = membership_matrix.tocsc()
    n_coms = membership_matrix.sum(axis=0)  # number of communities per node
    community_size_matrix = (
        sp.diags_array(community_sizes, dtype=np.uint32) @ membership_matrix
    )
    min_com_sizes = community_size_matrix.min(axis=0, explicit=True).todense()

    n = membership_matrix.shape[1]
    n_outliers = np.sum(n_coms == 0)
    n_inliers = n - n_outliers
    outlier_threshold = (
        (n_inliers / n) * np.sum(np.minimum(xi * degrees, 1)) + n_outliers - 1
    )

    assigned_degrees = np.empty_like(degrees)
    outlier_mask = n_coms == 0
    outlier_degrees, remaining_degrees = _assign_outlier_degrees(
        degrees,
        outlier_threshold,
        n_outliers,
        rng,
    )
    assigned_degrees[outlier_mask] = outlier_degrees

    eta = np.sum(community_sizes) / n_inliers
    expected_primary_community_sizes = community_sizes / eta
    phi = 1 - n_inliers * xi / (n_inliers * xi + n_outliers) * np.sum(
        np.power(expected_primary_community_sizes / n_inliers, 2)
    )
    thresholds = min_com_sizes[~outlier_mask] * n_coms[~outlier_mask] / (1 - xi * phi)

    if rho == 0:
        inlier_degrees = _assign_degrees(
            remaining_degrees,
            n_coms[~outlier_mask],
            thresholds,
            rng,
        )
    else:
        inlier_degrees = _assign_degrees_with_alpha_search(
            remaining_degrees,
            n_coms[~outlier_mask],
            thresholds,
            rng,
            rho,
            rho_tol,
            alpha_min,
            alpha_max,
            alpha_iters,
        )
    assigned_degrees[~outlier_mask] = inlier_degrees
    return assigned_degrees
