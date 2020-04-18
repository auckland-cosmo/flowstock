from typing import List, Optional, Tuple

import numba  # type: ignore
import numpy as np  # type: ignore
import scipy.optimize as opt  # type: ignore


class Akagi:

    """
    Use the method of Akagi et al to estimate movement
    """

    def __init__(self, N: np.ndarray, d: np.ndarray, K: float):

        # array of populations in regions at times
        self.N: np.ndarray = N

        # array of distances
        self.d: np.ndarray = d

        self.T: int = N.shape[0]
        num_cells = N.shape[1]
        self.num_cells: int = num_cells

        # Distance threshold
        self.K: float = K

        # List of indices of neighbors of respective cells
        self.gamma: np.ndarray = self.gamma_calc()
        # Gamma excluding self
        self.gamma_exc: np.ndarray = self.gamma.copy()
        np.fill_diagonal(self.gamma_exc, False)

        self.gamma_indices = np.where(self.gamma)
        self.gamma_exc_indices = np.where(self.gamma_exc)

        # self.M is the main output of the algorithm
        self.M: np.ndarray = np.zeros((self.T - 1, num_cells, num_cells), dtype=float)
        for i in range(self.M.shape[0]):
            np.fill_diagonal(self.M[i], N[i])  # Default to no movement

        # Initial guesses for parameters
        self.pi: np.ndarray = np.ones(num_cells) / 2
        self.s: np.ndarray = np.ones(num_cells) / 2
        self.beta: float = 1.0

        self.lamda = 1000

    def exact_inference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Estimate the movement of people in N and internal parameters
        """

        step = 0
        eps = 1e-5

        L = self.likelihood(self.M, self.pi, self.s, self.beta)
        L_old = L * (1 + 0.5)

        while abs((L_old - L) / L) > eps:
            print("step # ", step, ", L = ", L)

            self.update_M()

            self.update_pi()

            self.update_s_beta()

            L_old, L = L, self.likelihood(self.M, self.pi, self.s, self.beta)
            step += 1

        return self.M, self.pi, self.s, self.beta

    def neg_likelihood_flat(
        self, M, pi, s, beta, term_0_log=None, term_1_braces=None
    ) -> float:
        """
        Calculate likelihood with flattened M

        Needed for `scipy.optimize.minimize` to work right
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
        """

        d = self.d

        if term_0_log is None:
            term_0_log = self.term_0_log(pi)

        term_0 = term_0_log * M.diagonal(axis1=1, axis2=2)
        assert term_0.shape == (self.T - 1, self.num_cells)

        if term_1_braces is None:
            term_1_braces = self.term_1_braces(pi, s, beta, d)

        term_1 = term_1_braces * M
        assert term_1.shape == (self.T - 1, self.num_cells, self.num_cells)

        # TODO: Is this the best way to handle zeros in log?
        term_2 = M * (1 - np.log(M + (M == 0)))
        assert term_2.shape == (self.T - 1, self.num_cells, self.num_cells)

        term_3 = -self.lamda / 2.0 * self.cost(M, self.N)
        assert type(term_3) == float

        out = 0.0
        out += term_0.sum(axis=(0, 1))
        out += term_1[:, self.gamma_exc_indices[0], self.gamma_exc_indices[1]].sum()
        out += term_2[:, self.gamma_indices[0], self.gamma_indices[1]].sum()
        out += term_3

        return out

    def cost(self, M: np.ndarray, N: np.ndarray) -> float:
        """
        Cost function
        """

        return _cost(M, N)

    def term_0_log(self, pi: np.ndarray) -> np.ndarray:

        out = np.log(1 - pi)[np.newaxis, ...]

        return out

    def term_1_braces(
        self, pi: np.ndarray, s: np.ndarray, beta: float, d: np.ndarray
    ) -> np.ndarray:

        sexp = s[np.newaxis, ...] * np.exp(-beta * d)

        out = (
            # TODO: Handle 0 in log
            np.log((pi + (pi == 0))[np.newaxis, ..., np.newaxis])
            + np.log(s[np.newaxis, np.newaxis, ...])
            - beta * d[np.newaxis, ...]
            - np.log(sexp.sum(axis=1, where=self.gamma_exc))[
                np.newaxis, ..., np.newaxis
            ]
        )

        return out

    def update_M(self) -> bool:
        """
        Update M

        Search for an M that minimizes the likelihood

        Returns
        -------
        bool : True if there were no errors
               False if there were errors
        """

        bounds = self.M_bound()

        term_0_log = self.term_0_log(self.pi)
        term_1_braces = self.term_1_braces(self.pi, self.s, self.beta, self.d)

        result = opt.minimize(
            self.neg_likelihood_flat,
            # Use current M as initial guess
            x0=self.M,
            args=(self.pi, self.s, self.beta, term_0_log, term_1_braces),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": 1e-5,
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

        numer = self.M.sum(where=self.gamma_exc, axis=2).sum(axis=0)
        denom = self.M.sum(axis=2).sum(axis=0)

        assert numer.shape == denom.shape == self.pi.shape

        self.pi = numer / denom

    def update_s_beta(self) -> bool:
        """
        Update s and beta iteratively

        Returns
        -------
        bool : True if there were no errors
               False if there were errors
        """

        s = self.s
        beta = self.beta

        step = 0
        eps = 1e-8

        f_new = self.f(s, beta)
        f_old = f_new * (1 + 0.5)

        while abs((f_old - f_new) / f_new) > eps:

            # Update s
            # The paper says to use s_u and beta_u, I didn't
            s = self.A() / (
                self.C_u(s, beta)[..., np.newaxis] * np.exp(-beta * self.d)
            ).sum(where=self.gamma_exc, axis=1)

            # Fudge to force s != 0
            # Avoids problems with logs of 0 in calculation of f
            s = s + (s == 0.0) * 1e-5

            # Renormalize s
            s /= s.max()

            # Update beta
            # The paper says to use f_u, s_u and beta_u, I didn't
            beta_res = opt.minimize(
                lambda beta_: -self.f(self.s, beta_),
                x0=beta,
                method="SLSQP",
                bounds=[(0, 10)],
            )
            try:
                assert beta_res.success
                beta = beta_res.x[0]
            except AssertionError as err:
                print("Error maximizing wrt beta")
                print(err)
                print(beta_res.message)
                print("Bashing on regardless")
                beta = beta_res.x

            f_old, f_new = f_new, self.f(s, beta)
            step += 1

        self.s = s
        self.beta = beta

        assert np.all(np.isfinite(s))
        assert np.isfinite(beta)

        return beta_res.success

    def f(self, s, beta):
        """
        Objective function for Minorization-Maximization Algorith
        """

        A = self.A()
        B = self.B()
        D = self.D()
        sexp_term = self.sexp_term(s, beta)

        out = (A * np.log(s) - B * np.log(sexp_term)).sum(axis=0) - beta * D

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
        out += (out == 0) * 1e-5

        return out

    def D(self):
        out = (
            (self.M * self.d[np.newaxis, ...])
            .sum(where=self.gamma_exc, axis=2)
            .sum(axis=(0, 1))
        )
        assert out.shape == ()

        return out

    def sexp_term(self, s, beta):
        out = (s[np.newaxis, ...] * np.exp(-beta * self.d)).sum(
            where=self.gamma_exc, axis=1
        )
        assert out.shape == (self.num_cells,)

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


@numba.jit(nopython=True, fastmath=True)
def _cost(M: np.ndarray, N: np.ndarray) -> float:
    term_0 = (np.abs(N[:-1] - M.sum(axis=2)) ** 2).sum()
    term_1 = (np.abs(N[1:] - M.sum(axis=1)) ** 2).sum()

    out = term_0 + term_1

    return out
