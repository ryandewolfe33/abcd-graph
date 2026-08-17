from collections.abc import Callable
from warnings import warn

import numpy as np
import scipy.sparse as sp
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from abcd_graph.degrees import assign_degrees, split_degrees
from abcd_graph.membership import build_membership_matrix
from abcd_graph.models import chunglu_model, configuration_model, rewire
from abcd_graph.samplers import sample_community_sizes, sample_degrees


class ABCD:
    """Artificial Benchmark for Community Detection

    This class combines the ABCDGraph and ABCDParams class, essentially adding a
    sample method to ABCDParams.

    Parameters
    ----------

    vcount : int
        The number of vertices in the graph.

    gamma : float, default=2.5
        Powerlaw exponent for the degree distribution. Not used if
        degree_sequence is passed.

    beta: float, default=1.5
        Powerlaw exponent for the community size distribution. Not used if a
        custom community_size_sequence is passed.

    xi: float, default=0.25
        Proportion of edges in the global background graph. Setting xi=0 gives
        disjoint communities while xi=1 gives a random graph with no community
        structure.

    min_degree : int,  default=5
        Minimum degree in the graph. Not used if degree_sequence is passed.

    max_degree : int, default=30
        Maximum degree in the graph. Not used if degree_sequence is passed.

    min_community_size : int, default=20
        Minimum community size. Not used if a custom community_size_sequence
        is passed.

    max_community_size: int, default=250
        Maximum community size. Not used if a custom community_size_sequence
        is passed.

    degree_sequence : Sequence[int] | NDArray[np.int64] | None, default=None
        Used to pass a custom degree sequence that overrides the default
        powerlaw distribution.

    community_size_sequence : Sequence[int] | NDArray[np.int64] | None, default=None
        Used to pass a custom community size sequence that overrides the default
        powerlaw distribution. The sum of the community sizes must equal the number
        of vertices minus the number of outliers.

    num_outliers : int, default=0
        The number of outliers. These vertices have their entire degree in the
        global background graph so do not appear in any community.

    model : Model | None, default=None
        Random graph model used to sample the community and background graphs.

    verbose : bool, default=False
        Flag to log runtime infomation.
    """

    def __init__(
        self,
        n: int,
        xi: float = 0.25,
        outliers: int | float = 0,
        eta: float = 1.0,
        dimension: int = 8,
        rho: float = 0.0,
        degree_exponent: float = 2.5,
        min_degree: int = 5,
        max_degree: int | Callable[[int], int] = lambda n: int(n**0.5),
        community_size_exponent: float = 1.5,
        min_community_size: int = 20,
        max_community_size: int | Callable[[int], int] = lambda n: int(n**0.75),
        degree_sequence: ArrayLike | None = None,
        community_size_sequence: ArrayLike | None = None,
        rho_tol: float = 0.05,
        alpha_min: float = -60.0,
        alpha_max: float = 60.0,
        alpha_iters: int = 10,
        model: str = "configuration",
        max_swap_attempts_per_bad_edge: int = 5,
        drop_collisions: bool = False,
        rng: Generator = np.random.default_rng(),
        verbose: bool = False,
    ):
        self.n = n
        self.xi = xi
        self.outliers = outliers
        self.eta = eta
        self.dimension = dimension
        self.rho = rho
        self.degree_exponent = degree_exponent
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.community_size_exponent = community_size_exponent
        self.min_community_size = min_community_size
        self.max_community_size = max_community_size
        self.degree_sequence = degree_sequence
        self.community_size_sequence = community_size_sequence
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.alpha_iters = alpha_iters
        self.rho_tol = rho_tol
        self.model = model
        self.max_swap_attempts_per_bad_edge = max_swap_attempts_per_bad_edge
        self.drop_collisions = drop_collisions
        self.rng = rng
        self.verbose = verbose

    def _validate_params(self):
        if not isinstance(self.n, (int, np.integer)) or self.n < 1:
            raise ValueError("n must be a positive integer")

        if not isinstance(self.xi, (float, np.floating)) or self.xi < 0 or self.xi > 1:
            raise ValueError("xi must be a float between 0 and 1")

        if isinstance(self.outliers, (int, np.integer)):
            if self.outliers < 0:
                raise ValueError("integer outliers must be positive")
        elif isinstance(self.outliers, (float, np.floating)):
            if self.outliers < 0 or self.outliers > 1:
                raise ValueError("float outliers must be between 0 and 1")
        else:
            raise ValueError(
                "outliers must be a positive integer or a float between 0 and 1"
            )

        if not isinstance(self.eta, (float, np.floating)) or self.eta < 1:
            raise ValueError("eta must be at least 1")

        if not isinstance(self.dimension, (int, np.integer)) or self.dimension < 1:
            raise ValueError("dimensions must be a positive integer")

        if (
            not isinstance(self.rho, (float, np.floating))
            or self.rho < -1
            or self.rho > 1
        ):
            raise ValueError("rho must be between -1 and 1")

        if (
            not isinstance(self.degree_exponent, (float, np.floating))
            or self.degree_exponent < 0
        ):
            raise ValueError("rho must be positive")
        elif self.degree_exponent < 2 or self.degree_exponent > 3:
            warn("Typical degree exponents are between 2 and 3", stacklevel=2)

        if (
            not isinstance(self.min_degree, (int, np.integer))
            or self.min_degree < 1
            or self.min_degree >= self.n
        ):
            raise ValueError("min degree must be a positive int at most n")

        if not isinstance(self.max_degree, Callable) and (
            not isinstance(self.max_degree, (int, np.integer))
            or self.max_degree >= self.n
        ):
            raise ValueError(
                "max degree must be greater than min degree and less than n"
            )

        if (
            not isinstance(self.community_size_exponent, (float, np.floating))
            or self.community_size_exponent < 0
        ):
            raise ValueError("rho must be positive")
        elif self.community_size_exponent < 1 or self.community_size_exponent > 2:
            warn("Typical degree exponents are between 1 and 2", stacklevel=2)

        if (
            not isinstance(self.min_community_size, (int, np.integer))
            or self.min_community_size < 2
            or self.min_community_size >= self.n
        ):
            raise ValueError("min community size must be an integer between 2 and n")

        if not isinstance(self.max_community_size, Callable) and (
            not isinstance(self.max_community_size, (int, np.integer))
            or self.max_community_size >= self.n
        ):
            raise ValueError(
                "max community size must be greater than min community size and less than n"
            )

        if self.degree_sequence is not None:
            try:
                self.degree_sequence_ = np.asarray(
                    self.degree_sequence, dtype=np.uint32
                )
            except TypeError as e:
                raise ValueError(
                    "degree sequence must be able to cast to a numpy array of uint32"
                ) from e
            if self.degree_sequence_.shape != (self.n,):
                raise ValueError("degree sequence must be 1d array of length n")
            if np.sum(self.degree_sequence_) % 2 != 0:
                raise ValueError("sum of degree sequence must be even")

        if self.community_size_sequence is not None:
            try:
                self.community_size_sequence_ = np.asarray(
                    self.community_size_sequence, dtype=np.uint32
                )
            except TypeError as e:
                raise ValueError(
                    "community size sequence must be able to cast to a numpy array of uint32"
                ) from e
            if np.any(self.community_size_sequence >= self.n):
                raise ValueError("community sizes must be less than n")
            if np.any(self.community_size_sequence < 1):
                raise ValueError("community sizes must be at least 2")

        if not isinstance(self.alpha_max, (float, np.floating)) or self.alpha_max <= 0:
            raise ValueError("alpha max must be positive")

        if not isinstance(self.alpha_min, (float, np.floating)) or self.alpha_min >= 0:
            raise ValueError("alpha min must be negative")

        if not isinstance(self.alpha_iters, (int, np.integer)) or self.alpha_iters < 1:
            raise ValueError("alpha iters must be positive integer")

        if not isinstance(self.rho_tol, (float, np.floating)) or self.rho_tol < 0:
            raise ValueError("rho tol must be at least 0")

        # TODO models
        if self.model not in ["configuration", "chung-lu"]:
            raise ValueError("model must be one of 'configuration' or 'chung-lu'")

        if (
            not isinstance(self.max_swap_attempts_per_bad_edge, (int, np.integer))
            or self.max_swap_attempts_per_bad_edge < 1
        ):
            raise ValueError(
                "max swap attempts per bad edge must a be positive integer"
            )

        if not isinstance(self.drop_collisions, (bool, np.bool)):
            raise ValueError("drop collisions must be True or False")

        if not isinstance(self.rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator object")

    def sample(self) -> (NDArray[np.uint32], sp.csr_array):
        self._validate_params()

        if self.outliers < 1:
            n_outliers = int(self.n * self.outliers)
        else:
            n_outliers = self.outliers

        if self.degree_sequence is None:
            self.max_degree_ = (
                self.max_degree(self.n)
                if callable(self.max_degree)
                else self.max_degree
            )
            self.degree_sequence_ = sample_degrees(
                self.n,
                self.degree_exponent,
                self.min_degree,
                self.max_degree_,
                self.rng,
            )

        if self.community_size_sequence is None:
            self.max_community_size_ = (
                self.max_community_size(self.n)
                if callable(self.max_community_size)
                else self.max_community_size
            )
            self.community_size_sequence_ = sample_community_sizes(
                self.n - n_outliers,
                self.community_size_exponent,
                self.min_community_size,
                self.max_community_size_,
                self.eta,
                self.rng,
            )

        self.membership_matrix_ = build_membership_matrix(
            self.n,
            self.community_size_sequence_,
            n_outliers,
            self.dimension,
            self.rng,
        )

        assigned_degrees = assign_degrees(
            self.degree_sequence_,
            self.membership_matrix_,
            self.xi,
            self.rng,
            self.rho,
            self.rho_tol,
            self.alpha_min,
            self.alpha_max,
            self.alpha_iters,
        )

        community_degrees, background_degrees = split_degrees(
            assigned_degrees,
            self.membership_matrix_,
            self.xi,
            self.rng,
        )
        
        # TODO Parallel
        if self.model == "configuration":
            model_func = configuration_model
        elif self.model == "chung-lu":
            model_func = chunglu_model
        else:
            raise ValueError("model should be one of 'configuration' or 'chung-lu")

        graphs = [
            model_func(
                community_degrees.indices[
                    community_degrees.indptr[i] : community_degrees.indptr[i + 1]
                ],
                community_degrees.data[
                    community_degrees.indptr[i] : community_degrees.indptr[i + 1]
                ],
                self.rng,
            )
            for i in range(community_degrees.shape[0])
        ]
        graphs.append(
            model_func(
                np.arange(self.n, dtype=np.uint32),
                background_degrees,
                self.rng,
            )
        )
        graphs = [
            rewire(
                g, self.rng, self.max_swap_attempts_per_bad_edge, self.drop_collisions
            )
            for g in graphs
        ]

        self.graph_ = np.vstack(graphs)
        self.graph_ = rewire(
            self.graph_,
            self.rng,
            self.max_swap_attempts_per_bad_edge,
            self.drop_collisions,
        )

        return self.graph_, self.membership_matrix_
