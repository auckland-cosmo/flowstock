"""
Estimate the movement of people from snapshots of counts
"""

import os
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import numba  # type: ignore
import numpy as np  # type: ignore
import scipy.optimize as opt  # type: ignore

# FUDGE is used in a few places to avoid overflows and underflows
FUDGE = 1e-9


class SaveOptions:
    """
    Options specifying how and where to save output

    Parameters
    ----------

    path : path, optional
        Path to which output should be saved, relative to the current working directory

    period : int, optional
        Period with which to save output.
        1 saves every step.

    append_time : bool, optional
        Whether to append the current time to the path.
        This helps to avoid conflicts between different runs.
    """

    def __init__(self, path="output", period: int = 1, append_time: bool = True):

        self.output_dir = os.path.join(os.getcwd(), path)
        self.period = period
        self.append_time = append_time

    def make_dir(self):
        """
        Create the directory for output

        If `self.append_times` is `True`, append the current time to the path.
        This disambiguates the current simulation.
        """
        if self.append_time:
            self.output_dir = os.path.join(
                os.getcwd(), "output", str(datetime.now()).replace(" ", "_")
            )
        else:
            self.output_dir = os.path.join(os.getcwd(), "output")

        os.makedirs(self.output_dir, exist_ok=True)


class Akagi:
    """
    Use the method of Akagi et al to estimate movement

    Parameters
    ----------

    N : np.ndarray
        Array of data to fit to.
        `N.shape == (T, N)`, where `T` is the number of times with data and `N` is the
        number of regions.

    d : np.ndarray
        Array of distances between regions.
        `d.shape == (N, N)`, where `N` is the number of regions.

    K : float
        Distance cutoff.
        Regions with distance greater than `K` are assumed not to have people flow
        between them.

    Attributes
    ----------

    M : np.ndarray
        The current guess of the movement array.
        `M.shape == (T - 1, N, N)`

    pi : np.ndarray
        The current guess of the departure probabilities.

    s : np.ndarray
        The current guess of the gathering scores.

    beta : float
        The current guess of the distance dependence.

    lambda : float
        Weighting of the cost function.

    N : np.ndarray

    d : np.ndarray

    K : float

    T : int

    num_cells : int

    gamma : np.ndarray
        Boolean array of neighbouring cells, including self as a neighbour.
        `gamma[i, j] is True` if and only if `d[i, j] <= K`.

    gamma_exc : np.ndarray
        Boolean array of neighbouring cells, excluding self as a neighbour.
        `gamma[i, j] is True` if and only if `d[i, j] <= K` and `i is not j`.

    gamma_indices : np.ndarray
        List of indices of neighbors of regions, including the originating region.
        `gamma_indices[i]` contains indices of all regions within distance `K` of region
        `i`.

    gamma_exc_indices : np.ndarray
        List of indices of neighbors of regions, excluding the originating region.

    Notes
    -----

    Every region must have at least one possible destination.

    References
    ----------

    .. [Akagi2018] Y. Akagi, T. Nishimura, T. Kurashima, and H. Toda, “A Fast and
    Accurate Method for Estimating People Flow from Spatiotemporal Population Data,”
    in Proceedings of the Twenty-Seventh International Joint Conference on Artificial
    Intelligence, Stockholm, Sweden, Jul. 2018, pp. 3293–3300,
    doi: 10.24963/ijcai.2018/457.
    """

    def __init__(
        self,
        N: np.ndarray,
        d: np.ndarray,
        K: float,
        save_options: SaveOptions = SaveOptions(),
        nonlinear_beta: bool = False,
    ):

        # array of populations in regions at times
        self.N: np.ndarray = N.copy(order="C")

        # array of distances
        self.d: np.ndarray = d.copy(order="C")

        try:
            assert self.d.shape[0] == self.d.shape[1]
            assert self.N.shape[1] == self.d.shape[0]
        except AssertionError as err:
            print("Shapes of N and d are inconsistent")
            raise err

        self.T: int = N.shape[0]
        num_cells = N.shape[1]
        self.num_cells: int = num_cells

        # Distance threshold
        self.K: float = K

        self.gamma: np.ndarray = self.gamma_calc()
        # Gamma excluding self
        self.gamma_exc: np.ndarray = self.gamma.copy()
        np.fill_diagonal(self.gamma_exc, False)

        # List of indices of neighbors of respective cells
        self.gamma_indices = np.where(self.gamma)
        self.gamma_exc_indices = np.where(self.gamma_exc)

        # self.M is the main output of the algorithm
        self.M: np.ndarray = np.zeros((self.T - 1, num_cells, num_cells), dtype=float)
        self.init_M_static()

        # Initial guesses for parameters
        self.pi: np.ndarray = np.ones(num_cells) / 50
        self.s: np.ndarray = np.ones(num_cells) / 50

        # Set bounds  on linear term in exponential as if we expect beta_0 * d ~ O(1)
        min_beta_0 = -1 / d.max() * 100
        max_beta_0 = +1 / d.max() * 100

        # Initialise beta to be within bounds but prefer positive
        self.beta: float = (0 + max_beta_0) / 2
        self.beta_bounds = [(min_beta_0, max_beta_0)]

        self.lamda = 1000

        self.save_options = save_options

    def exact_inference(
        self, eps: float, use_derivative: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Estimate the movement of people from data `self.N`.

        Parameters
        ----------

        eps : float
            Threshold used in optimizations.

        use_derivative : bool, optional
            Whether to use the analytic derivative of the likelihood.
            In normal usage this should be true.

        Returns
        -------
        (M, pi, s, beta) : Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """

        self.save_options.make_dir()
        self.save_config()

        step = 0

        L = self.likelihood(self.M, self.pi, self.s, self.beta)

        while True:
            print("step # ", step, ", L = ", L)
            print("pi[:10]=", self.pi[:10])
            print("s[:10]=", self.s[:10])
            print("beta=", self.beta)

            success_M = self.update_M(eps, use_derivative)
            print("M done")

            self.update_pi()
            print("pi done")

            success_s_beta = self.update_s_beta(eps)
            print("beta done")

            self.save_checkpoint(step)

            L_old, L = L, self.likelihood(self.M, self.pi, self.s, self.beta)

            step += 1

            if abs((L_old - L) / L) < eps and success_M and success_s_beta:
                break

        self.save_state(step)

        return self.M, self.pi, self.s, self.beta

    def neg_likelihood_flat(
        self,
        M: np.ndarray,
        pi: np.ndarray,
        s: np.ndarray,
        beta: float,
        term_0_log: Optional[np.ndarray] = None,
        term_1_braces: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate likelihood with `M` flattened

        This function is needed to interface with the API of
        `scipy.optimize.minimize`, which expects a function of a vector.

        Parameters
        ----------

        M: np.ndarray
            Flattened `M`.  It should have shape `(T-1) * N * N`, NOT `(T-1, N, N)`.

        pi: np.ndarray

        s: np.ndarray

        beta: np.ndarray

        term_0_log: Optional[np.ndnarry]
        term_1_braces: Optional[np.ndnarry]
            There is an option to pass in precomputed values for some terms in
            the likelihood.  This can speed the algorithm up significantly if
            used when minimizing with respect to `M`.

        Returns
        -------

        float
        """

        M_reshaped = np.reshape(M, self.M.shape)

        return -self.likelihood(
            M_reshaped, pi, s, beta, term_0_log=term_0_log, term_1_braces=term_1_braces
        )

    def likelihood(
        self,
        M: np.ndarray,
        pi: np.ndarray,
        s: np.ndarray,
        beta: np.ndarray,
        term_0_log: Optional[np.ndarray] = None,
        term_1_braces: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate  likelihood

        Parameters
        ----------

        M: np.ndarray

        pi: np.ndarray

        s: np.ndarray

        beta: np.ndarray

        term_0_log: Optional[np.ndnarry]
        term_1_braces: Optional[np.ndnarry]
            There is an option to pass in precomputed values for some terms in
            the likelihood.  This can speed the algorithm up significantly if
            used when minimizing with respect to `M`.

        Returns
        -------

        float
        """

        if term_0_log is None:
            term_0_log = self.term_0_log(pi)

        if term_1_braces is None:
            term_1_braces = self.term_1_braces(pi, s, beta, self.d)

        term_0 = _term_0_summed(term_0_log, M)

        term_1 = _term_1(term_1_braces, M)

        term_2 = _term_2(M)

        term_3 = -self.lamda / 2.0 * _cost(M, self.N)
        assert type(term_3) == float

        out = 0.0
        out += term_0
        out += term_1[:, self.gamma_exc_indices[0], self.gamma_exc_indices[1]].sum()
        out += term_2[:, self.gamma_indices[0], self.gamma_indices[1]].sum()
        out += term_3

        # added a print statement to return this each time we call it
        print("L ", term_3, out)
        return out

    def neg_dLdMlmn_flat(
        self, M, pi, s, beta, term_0_log=None, term_1_braces=None
    ) -> np.ndarray:
        """
        The Jacobian of the likelihood as a function of a flattened `M`

        Parameters
        ----------

        M: np.ndarray
            Flattened `M`.  It should have shape `(T-1) * N * N`, NOT `(T-1, N, N)`.

        pi: np.ndarray

        s: np.ndarray

        beta: np.ndarray

        term_0_log: Optional[np.ndnarry]
        term_1_braces: Optional[np.ndnarry]
            There is an option to pass in precomputed values for some terms in
            the likelihood.  This can speed the algorithm up significantly if
            used when minimizing with respect to `M`.

        Returns
        -------

        np.ndarray
        """

        if term_0_log is None:
            term_0_log = self.term_0_log(pi)

        if term_1_braces is None:
            term_1_braces = self.term_1_braces(pi, s, beta, self.d)

        M_reshaped = np.reshape(M, self.M.shape)

        Mder = -self.dLdMlmn(M_reshaped, pi, s, beta, term_0_log, term_1_braces)

        return np.reshape(Mder, (self.T - 1) * self.num_cells**2)

    def dLdMlmn(
        self,
        M: np.ndarray,
        pi: np.ndarray,
        s: np.ndarray,
        beta: np.ndarray,
        term_0_log: np.array,
        term_1_braces: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the Jacobian of the likelihood as a function of `M`

        Parameters
        ----------

        M: np.ndarray

        pi: np.ndarray

        s: np.ndarray

        beta: np.ndarray

        term_0_log: Optional[np.ndnarry]
        term_1_braces: Optional[np.ndnarry]
            There is an option to pass in precomputed values for some terms in
            the likelihood.  This can speed the algorithm up significantly if
            used when minimizing with respect to `M`.

        Returns
        -------

        np.ndarray
        """

        term_1 = term_0_log

        term_2 = term_1_braces

        # We need to be careful with this logarithm if Mtij is zero
        # This fudge applies to entries in M that should be identically 0, but
        # they aren't used in the select
        term_3 = -np.log(M + np.isclose(M, 0) * FUDGE)

        term_4 = self.lamda * (
            (self.N[:-1] - M.sum(axis=2))[..., np.newaxis]
            + (self.N[1:] - M.sum(axis=1))[:, np.newaxis, :]
        )

        term_3_4 = term_3 + term_4

        out = np.select(
            [
                self.gamma_exc,  # in gamma but not m==n; short circuits to avoid m==n
                self.gamma,  # m == n
            ],
            [
                term_2 + term_3_4,  # value when m is in gamma but not m==n
                term_1 + term_3_4,  # m==n
            ],
            default=0.0,
        )

        return out

    def cost(self, M: np.ndarray, N: np.ndarray) -> float:
        r"""
        Cost function

        Parameters
        ----------

        M : np.ndarray
        N : np.ndarray

        Notes
        -----

        .. math::

            \mathcal{C}({\bf M})}
            =
            \sum_{t=0}^{T-2} \sum_{i} {\left( N_{ti} - \sum_{j} \Mtij\right)}^2
            +
            \sum_{t=0}^{T-2} \sum_{i} {\left( N_{t+1,i} - \sum_{j} \Mtji\right)}^2
        """

        return _cost(M, N)

    def term_0_log(self, pi: np.ndarray) -> np.ndarray:

        # Fudge at pi == 1
        out = np.log(1 - pi + np.isclose(pi, 1) * FUDGE)[np.newaxis, ...]

        assert out.shape == (1, self.num_cells)

        return out

    def term_1_braces(
        self, pi: np.ndarray, s: np.ndarray, beta: float, d: np.ndarray
    ) -> np.ndarray:

        sexp = s[np.newaxis, ...] * np.exp(self.exponent(beta))

        out = (
            # TODO: Handle 0 in log
            np.log((pi + (pi == 0))[np.newaxis, ..., np.newaxis])
            + np.log(s[np.newaxis, np.newaxis, ...])
            + self.exponent(beta)[np.newaxis, ...]
            - np.log(sexp.sum(axis=1, where=self.gamma_exc))[
                np.newaxis, ..., np.newaxis
            ]
        )

        assert out.shape == (1, self.num_cells, self.num_cells)

        return out

    def update_M(self, eps: float, use_derivative: bool) -> bool:
        """
        Minimize likelihood with respect to `M`

        Search for an M that minimizes the likelihood, while leaving `pi`, `s`, `beta`
        fixed.
        Use the limited-memory BFGS optimization method.

        Parameters
        ----------

        eps : float

        use_derivative : bool, optional

        Returns
        -------
        bool : True if there were no errors
               False if there were errors

        Notes
        -----

        Constraints are put on `M` to obey the maximum distance `K`.
        """

        __import__("pudb").set_trace()

        bounds = self.M_bound()

        term_0_log = self.term_0_log(self.pi)
        term_1_braces = self.term_1_braces(self.pi, self.s, self.beta, self.d)

        jac: Optional[Callable]
        if use_derivative:
            jac = self.neg_dLdMlmn_flat
        else:
            jac = None

        result = opt.minimize(
            self.neg_likelihood_flat,
            # Use current M as initial guess
            x0=self.M,
            args=(self.pi, self.s, self.beta, term_0_log, term_1_braces),
            method="L-BFGS-B",
            jac=jac,
            bounds=bounds,
            options={
                "ftol": eps * 1e-1,
                # asking for a tighter limit here than in the solver as a whole.
                # "maxfun": 15_000_000,
            },
        )

        try:
            assert result.success
        except AssertionError:
            print("Error minimizing M", result.message)

        self.M = np.reshape(result.x, self.M.shape)

        return result.success

    def update_pi(self):
        """
        Minimize likelihood with respect to `pi`

        Parameters
        ----------

        Returns
        -------

        None
        """

        numer = self.M.sum(where=self.gamma_exc, axis=2).sum(axis=0)
        denom = self.M.sum(axis=2).sum(axis=0)

        assert numer.shape == denom.shape == self.pi.shape

        self.pi = numer / denom

    def update_s_beta(self, eps: float) -> bool:
        """
        Minimize likelihood with respect to `s` and `beta`

        Parameters
        ----------
        eps : float

        Returns
        -------
        bool : True if there were no errors
               False if there were errors
        """

        s = self.s
        beta = self.beta

        step = 0
        eps *= 1e-4

        f_0 = self.f(s, beta)
        f_1 = self.f(s, beta)

        while True:

            # Update s
            # The paper says to use s_u and beta_u, I didn't
            s = self.A() / (
                self.C_u(s, beta)[..., np.newaxis] * np.exp(self.exponent(beta))
            ).sum(where=self.gamma_exc, axis=1)

            # Fudge to force s != 0
            # Avoids problems with logs of 0 in calculation of f
            s = s + np.isclose(s, 0.0) * FUDGE

            # Renormalize s
            s /= s.max()

            # Update beta
            # The paper says to use f_u, s_u and beta_u, I didn't
            beta_res = opt.minimize_scalar(
                lambda beta_: -self.f(self.s, beta_),
                bounds=self.beta_bounds[0],
                method="bounded",
            )
            try:
                assert beta_res.success
                beta = beta_res.x
                success = beta_res.success
            except AssertionError as err:
                print("Error maximizing wrt beta")
                print(beta_res.message)
                print("Bashing on regardless")
                print(err)
                beta = beta_res.x
                success = beta_res.success

            f_2, f_1, f_0 = f_1, f_0, self.f(s, beta)

            if abs(f_1 - f_0) <= eps * abs(f_0):
                break

            step += 1

            # Check for cycles
            if step > 1_000 and abs(f_0 - f_1) > abs(f_0 - f_2):
                success = False
                print("s, beta: optimization caught in cycle; terminated")
                break

        self.s = s
        self.beta = beta

        assert np.all(np.isfinite(s))
        assert np.all(np.isfinite(beta))

        return success

    def update_s_beta_u(self, eps: float) -> bool:
        """
        Update s and beta iteratively

        Returns
        -------
        bool : True if there were no errors
               False if there were errors
        """

        s = self.s
        beta = self.beta

        s_u = self.s
        beta_u = self.beta

        step = 0
        eps *= 1e-4

        f_new = self.f(s, beta)

        while True:

            # Update s
            # Trying to use s_u and beta_u
            s = self.A() / (
                self.C_u(s_u, beta_u)[..., np.newaxis] * np.exp(self.exponent(beta_u))
            ).sum(where=self.gamma_exc, axis=1)

            # Fudge to force s != 0
            # Avoids problems with logs of 0 in calculation of f
            s = s + (s == 0.0) * FUDGE

            # Renormalize s
            s /= s.max()
            s_u = s

            # Update beta
            # Trying to use f_u, s_u and beta_u
            beta_res = opt.minimize(
                lambda beta_: -self.f_u(s, beta_, s_u, beta_u),
                x0=beta_u,
                method="SLSQP",
                bounds=self.beta_bounds,
            )
            try:
                assert beta_res.success
                beta = beta_res.x
                beta_u = beta
            except AssertionError as err:
                print("Error maximizing wrt beta")
                print(err)
                print(beta_res.message)
                print("Bashing on regardless")
                beta = beta_res.x
                beta_u = beta

            f_old, f_new = f_new, self.f_u(s, beta, s_u, beta_u)

            if abs((f_old - f_new) / f_new) < eps:
                break

            step += 1

        self.s = s
        self.beta = beta

        return beta_res.success

    def f(self, s, beta) -> float:
        """
        Objective function for Minorization-Maximization Algorith

        Parameters
        ----------
        s : np.ndarray

        beta : float

        Returns
        -------

        float
        """

        A = self.A()
        B = self.B()
        beta_D = (
            (self.exponent(beta)[np.newaxis, ...] * self.M)
            .sum(where=self.gamma_exc, axis=2)
            .sum(axis=(0, 1))
        )
        sexp_term = self.sexp_term(s, beta)

        out = (A * np.log(s) - B * np.log(sexp_term)).sum(axis=0) + beta_D

        return out

    def f_u(self, s, beta, s_u, beta_u) -> float:
        A = self.A()
        C = self.C_u(s_u, beta_u)
        beta_D = (
            (self.exponent(beta)[np.newaxis, ...] * self.M)
            .sum(where=self.gamma_exc, axis=2)
            .sum(axis=(0, 1))
        )
        sexp_term = self.sexp_term(s, beta)

        out = (A * np.log(s) - C * sexp_term).sum(axis=0) + beta_D
        return out

    def A(self):
        out = self.M.sum(where=self.gamma_exc.T, axis=(0, 1))
        assert out.shape == (self.num_cells,)

        return out

    def B(self):
        out = self.M.sum(where=self.gamma_exc, axis=(0, 2))
        assert out.shape == (self.num_cells,)

        return out

    def C_u(self, s_u, beta_u):
        out = self.B() / self.sexp_term(s_u, beta_u)
        assert out.shape == (self.num_cells,)

        # Fudge: C_u = 0 breaks s = A / C_u
        out += (out == 0) * FUDGE

        return out

    def sexp_term(self, s, beta):
        out = (s[np.newaxis, ...] * np.exp(self.exponent(beta))).sum(
            where=self.gamma_exc, axis=1
        )
        assert out.shape == (self.num_cells,)

        return out

    def exponent(self, beta):
        """
        Calculate the exponent in the distance-based probability
        """

        out = -beta * self.d
        # out = -beta[0] * self.d
        # out = -(beta[0] * self.d + beta[1] * self.d ** 2)

        return out

    def gamma_calc(self):

        gamma = self.d <= self.K

        return gamma

    def M_bound(self) -> List[Tuple[float, float]]:

        N = self.N
        gamma = self.gamma

        # Can't have more people move out of a region than are in it
        # Allow for 10% more people flowing from a cell - there is noise in the data
        upper_col = N[:-1][..., np.newaxis].astype(float) * 1.1
        upper = np.repeat(upper_col, self.num_cells, axis=2)

        # Can't have people move to disallowed regions
        upper_dist = np.where(gamma[np.newaxis, ...], upper, 0)

        assert upper_dist.shape == self.M.shape

        # Can't have negative people flowing into a region
        lower = np.zeros_like(upper_dist)

        bounds = list(zip(lower.flatten(), upper_dist.flatten()))

        return bounds

    def save_config(self):
        """
        Save configuration details
        """

        output_dir = self.save_options.output_dir

        np.save(os.path.join(output_dir, "N"), self.N)
        np.save(os.path.join(output_dir, "d"), self.d)
        np.save(os.path.join(output_dir, "K"), self.K)
        np.save(os.path.join(output_dir, "lambda"), self.lamda)
        np.save(os.path.join(output_dir, "gamma"), self.gamma)

    def save_checkpoint(self, step):
        """
        Save checkpoint data if at an appropriate step
        """

        if step % self.save_options.period == 0:
            self.save_state(step)

    def save_state(self, step):
        """
        Save state regardless of step number
        """

        output_dir = self.save_options.output_dir

        step_fmt = "_{:05d}".format(step)

        np.save(os.path.join(output_dir, "M" + step_fmt), self.M)
        np.save(os.path.join(output_dir, "pi" + step_fmt), self.pi)
        np.save(os.path.join(output_dir, "s" + step_fmt), self.s)
        np.save(os.path.join(output_dir, "beta" + step_fmt), self.beta)

    def init_M_static(self):
        """
        Initialise `M` such that people don't move
        """

        for i in range(self.M.shape[0]):
            np.fill_diagonal(self.M[i], self.N[i])  # Default to no movement

    def init_M_moving(self):
        """
        Initialise `M` to reflect a local decrease in the count
        """

        self.M = np.zeros_like(self.M)
        self.M += (
            self.gamma_exc
            * (np.abs(self.N[:-1] - self.N[1:]) / self.gamma_exc.sum(axis=1))[
                ..., np.newaxis
            ]
        )
        for i in range(self.M.shape[0]):
            np.fill_diagonal(self.M[i], self.N[i] - self.M.sum(axis=2))

    def M_jitter(self, amplitude):
        """
        Add some perturbations to `M`

        The amplitude is as a fraction of the current population.
        """

        self.M += (
            self.gamma
            * np.random.random(size=self.M.shape)
            * amplitude
            * self.N[:-1, ..., np.newaxis]
        )


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _term_0_summed(term_0_log: np.ndarray, M: np.ndarray):

    T = M.shape[0] + 1
    num_cells = M.shape[1]

    M_diag = np.zeros((T - 1, num_cells))
    for t in numba.prange(T - 1):
        for i in numba.prange(num_cells):
            M_diag[t, i] = M[t, i, i]

    mult = term_0_log[0] * M_diag
    assert mult.shape == (T - 1, num_cells)

    out = mult.sum()

    return out


@numba.jit(nopython=True, fastmath=True)
def _term_1(term_1_braces: np.ndarray, M: np.ndarray) -> np.ndarray:

    T = M.shape[0] + 1
    num_cells = M.shape[1]

    out = term_1_braces * M
    assert out.shape == (T - 1, num_cells, num_cells)

    return out


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _term_2(M: np.ndarray) -> np.ndarray:

    T = M.shape[0] + 1
    num_cells = M.shape[1]

    # TODO: Is this the best way to handle zeros in log?
    out = M * (1 - np.log(M + (M == 0)))

    assert out.shape == (T - 1, num_cells, num_cells)

    return out


@numba.jit(nopython=True, fastmath=True, parallel=True)
def _cost(M: np.ndarray, N: np.ndarray) -> float:
    term_0 = (np.abs(N[:-1] - M.sum(axis=2)) ** 2).sum()
    term_1 = (np.abs(N[1:] - M.sum(axis=1)) ** 2).sum()

    out = term_0 + term_1

    return out
