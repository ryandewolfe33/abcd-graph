import numpy as np
import powerlaw
import scipy.sparse as sp
from numba import njit
from numpy.typing import ArrayLike, NDArray


@njit
def count_intra_community_edges(
    edges: NDArray[np.uint32],
    indptr: NDArray[np.uint64],
    indices: NDArray[np.uint32],
):
    m_intra_community = 0
    for i in range(edges.shape[0]):
        u, v = edges[i]
        u_coms = indices[indptr[u] : indptr[u + 1]]
        v_coms = indices[indptr[v] : indptr[v + 1]]
        m_intra_community += int(np.any(np.isin(u_coms, v_coms)))
    return m_intra_community


def icdf(points: ArrayLike, sequence: ArrayLike):
    points = np.asarray(points)
    points = np.insert(points, 0, 0)
    hist, bin_edges = np.histogram(sequence, bins=points)
    cdf = np.cumsum(hist)
    cdf /= cdf[-1]
    icdf = 1 - cdf
    return icdf


class ABCDSample:
    """A sample from the ABCD model.

    Consists of an edge list and a community-node membership matrix.
    Has functions for measuring empirical properties of the graph, and
    functions for converting the edge list to other common graph types.
    """

    def __init__(
        self,
        edges: NDArray[np.uint32],
        membership_matrix: sp.csr_array,
    ):
        self.edges = edges
        self.membership_matrix = membership_matrix

    @property
    def n(self) -> int:
        return self.membership_matrix.shape[1]

    @property
    def xi(self) -> float:
        membership_csc = self.membership_matrix.tocsc()
        m_intra_community = count_intra_community_edges(
            self.edges,
            membership_csc.indptr,
            membership_csc.indices,
        )
        return 1 - (m_intra_community / self.m)

    @property
    def m(self) -> int:
        return self.edges.shape[0]

    @property
    def outliers(self) -> int:
        return np.sum(self.membership_matrix.count_nonzero(axis=0) == 0)

    @property
    def eta(self) -> float:
        return self.membership_matrix.sum() / (self.n - self.outliers)

    @property
    def rho(self) -> float:
        degrees = self.to_sparse().sum(axis=1) // 2
        coms_per_node = self.membership_matrix.sum(axis=0)
        inlier_mask = coms_per_node > 0
        rho = np.corrcoef(degrees[inlier_mask], coms_per_node[inlier_mask])[0, 1]
        return rho

    @property
    def degree_sequence(self) -> NDArray:
        degrees = np.zeros(self.n, dtype=np.uint32)
        node, degree = np.unique_counts(self.edges)
        degrees[node] = degree
        return degrees

    @property
    def min_degree(self) -> int:
        return np.min(self.degree_sequence)

    @property
    def max_degree(self) -> int:
        return np.max(self.degree_sequence)

    @property
    def degree_exponent(self) -> float:
        degree_sequence = self.degree_sequence
        min_degree = np.min(degree_sequence)
        exponent = powerlaw.Fit(
            degree_sequence,
            discrete=True,
            verbose=False,
            xmin=min_degree,
        ).power_law.alpha
        return exponent

    @property
    def community_size_sequence(self) -> NDArray:
        return self.membership_matrix.sum(axis=1)

    @property
    def min_community_size(self) -> int:
        return np.min(self.community_size_sequence)

    @property
    def max_community_size(self) -> int:
        return np.max(self.community_size_sequence)

    @property
    def community_size_exponent(self) -> float:
        size_sequence = self.community_size_sequence
        min_size = np.min(size_sequence)
        exponent = powerlaw.Fit(
            size_sequence,
            discrete=True,
            verbose=False,
            xmin=min_size,
        ).power_law.alpha
        return exponent

    def to_sparse(self, matrix: bool = False) -> sp.csr_array | sp.csr_matrix:
        adjacency = sp.coo_array(
            (np.ones(self.edges.shape[0], dtype=np.bool), self.edges.T),
            shape=(self.n, self.n),
        )
        adjacency = adjacency + adjacency.transpose()
        if matrix:
            adjacency = sp.coo_matrix(adjacency)
        return adjacency.tocsr()

    def to_dense(self) -> NDArray[np.bool]:
        return self.to_sparse().todense()

    def to_networkx(self):
        try:
            import networkx as nx

            g = nx.from_edgelist(self.edges)
            return g
        except ImportError as e:
            raise ImportError(
                "Package 'networkx' is not installed. "
                "Run `pip install networkx` to install it."
            ) from e

    def to_igraph(self):
        try:
            import igraph as ig

            g = ig.Graph(n=self.n, edges=self.edges, directed=False)
            return g
        except ImportError as e:
            raise ImportError(
                "Package 'igraph' is not installed. "
                "Run `pip install igraph` to install it."
            ) from e

    def community_array(self):
        if self.eta > 1:
            raise ValueError(
                "Overlapping communities cannot be represent with an array"
            )
        result = np.full(self.n, -1)
        coms, nodes = self.membership_matrix.nonzero()
        result[nodes] = coms
        return result

    def degree_icdf(self, points: ArrayLike):
        return icdf(self.degree_sequence)

    def community_size_icdf(self, points: ArrayLike):
        return icdf(self.community_size_sequence)
