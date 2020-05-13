import os
from datetime import datetime
from typing import List, Optional, Tuple

import numba  # type: ignore
import numpy as np  # type: ignore
import scipy.optimize as opt  # type: ignore


class SaveOptions:
    def __init__(self, path="output", period=1, append_time=True):

        self.output_dir = os.path.join(os.getcwd(), "output")
        self.period = period
        self.append_time = append_time

    def make_dir(self):
        if self.append_time:
            output_dir = os.path.join(os.getcwd(), "output", str(datetime.now()))
        else:
            output_dir = os.path.join(os.getcwd(), "output")

        os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir


class Akagi:

    """
    Use the method of Akagi et al to estimate movement
    """

    def __init__(
        self,
        N: np.ndarray,
        d: np.ndarray,
        K: float,
        save_options: SaveOptions = SaveOptions(),
    ):

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
        self.pi: np.ndarray = np.ones(num_cells) / 50
        self.s: np.ndarray = np.ones(num_cells) / 50
        self.beta: float = 0.1

        self.lamda = 1000

        self.save_options = save_options

    def exact_inference(
        self, eps: float, der: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Estimate the movement of people in N and internal parameters
        """

        self.save_options.make_dir()
        self.save_config()

        step = 0

        L = self.likelihood(self.M, self.pi, self.s, self.beta)
        L_old = L * (1 + 0.5)

        while abs((L_old - L) / L) > eps:
            print("step # ", step, ", L = ", L)
            print("beta ", self.beta)
            print("pi ", self.pi)

            self.update_M(eps, der)

            print("M done")

            self.update_pi()

            print("pi done")

            self.update_s_beta_u(eps)

            print("beta done")

            self.save_checkpoint(step)

            L_old, L = L, self.likelihood(self.M, self.pi, self.s, self.beta)

            step += 1

        self.save_state(step)

        return self.M, self.pi, self.s, self.beta

    def neg_likelihood_flat(
        self, M, pi, s, beta, term_0_log=None, term_1_braces=None
    ) -> np.ndarray:
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

        Parameters
        ----------

        term_0_log: Optional[np.ndnarry]
        term_1_braces: Optional[np.ndnarry]
            There is the option of caching some terms in the likelihood.  This
            can speed the algorithm up significantly if used when minimizing
            with respect to `M`.
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

    def dLdMlmn_flat(
        self, M, pi, s, beta, term_0_log=None, term_1_braces=None
    ) -> np.ndarray:
        """
        Flattening the matix M for derivative.

        Also, populate the output matrix dL/dMtij for each t,i,j.
        Needs to be fixed so that the loops areplaced with something pythonic.
        """

        M_reshaped = np.reshape(M, self.M.shape)

        # This looping here needs to be fixed.
        Mder = np.zeros((self.T - 1, self.num_cells, self.num_cells))
        for t in range(0, self.T - 1):
            for i in range(0, self.num_cells):
                for j in range(0, self.num_cells):
                    Mder[t, i, j] = -self.dLdMlmn(
                        M_reshaped,
                        pi,
                        s,
                        beta,
                        t,
                        i,
                        j,
                        term_0_log=term_0_log,
                        term_1_braces=term_1_braces,
                    )
        # return np.array(Mder) # to ouptut a matrix of gradients
        return np.reshape(Mder, (self.T - 1) * self.num_cells ** 2)

    def dLdMlmn(
        self,
        M: np.ndarray,
        pi: np.ndarray,
        s: np.ndarray,
        beta: np.ndarray,
        l: int,
        m: int,
        n: int,
        term_0_log: Optional[np.ndarray] = None,
        term_1_braces: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculates the derivative of the likelihood function w.r.t to
        one of the elements Mlmn.
        """

        out = 0.0

        term_4 = self.lamda * (
            self.N[l, m] + self.N[l + 1, n] - M[l, :, n].sum() - M[l, m, :].sum()
        )

        # we need to be careful with this logarithm if Mtij is zero
        term_3 = -np.log(M[l, m, n] + (M[l, m, n] == 0))

        sexp = s[np.newaxis, ...] * np.exp(self.exponent(beta)[m, n])

        # check if these terms are neighbours:
        if self.gamma[m, n] is True:
            if m == n:
                term_1 = np.log(1.0 - pi[m])
                out = term_1 + term_3 + term_4
            else:
                # m and n different, but neighbours:
                term_2 = (
                    np.log(pi[m])
                    + np.log(s[n])
                    + self.exponent(beta)[m, n]
                    - np.log(sexp.sum(axis=1, where=self.gamma_exc))
                )
                out = term_2 + term_3 + term_4
        else:
            out = 0.0

        return out

    def cost(self, M: np.ndarray, N: np.ndarray) -> float:
        """
        Cost function
        """

        return _cost(M, N)

    def term_0_log(self, pi: np.ndarray) -> np.ndarray:

        out = np.log(1 - pi)[np.newaxis, ...]

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

    def update_M(self, eps: float, der: bool) -> bool:
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

        if der is False:
            result = opt.minimize(
                self.neg_likelihood_flat,
                # Use current M as initial guess
                x0=self.M,
                args=(self.pi, self.s, self.beta, term_0_log, term_1_braces),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "ftol": eps,
                    # "maxfun": 15_000_000,
                },
            )
        else:
            result = opt.minimize(
                self.neg_likelihood_flat,
                # Use current M as initial guess
                x0=self.M,
                args=(self.pi, self.s, self.beta, term_0_log, term_1_braces),
                method="L-BFGS-B",
                jac=self.dLdMlmn_flat,
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

        # print(self.M[0][:10, :10])
        self.M = np.reshape(result.x, self.M.shape)
        # print(self.M[0][:10, :10])

        return result.success

    def update_pi(self):

        numer = self.M.sum(where=self.gamma_exc, axis=2).sum(axis=0)
        denom = self.M.sum(axis=2).sum(axis=0)

        assert numer.shape == denom.shape == self.pi.shape

        self.pi = numer / denom

    def update_s_beta(self, eps: float) -> bool:
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
        eps *= 1e-4

        f_new = self.f(s, beta)

        while True:

            # Update s
            # The paper says to use s_u and beta_u, I didn't
            s = self.A() / (
                self.C_u(s, beta)[..., np.newaxis] * np.exp(self.exponent(beta))
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
                bounds=[(-1, 10)],
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

            if not abs((f_old - f_new) / f_new) > eps:
                break

            step += 1

        self.s = s
        self.beta = beta

        assert np.all(np.isfinite(s))
        assert np.isfinite(beta)

        return beta_res.success

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
            s = s + (s == 0.0) * 1e-5

            # Renormalize s
            s /= s.max()
            s_u = s

            # Update beta
            # Trying to use f_u, s_u and beta_u
            beta_res = opt.minimize(
                lambda beta_: -self.f_u(s, beta_, s_u, beta_u),
                x0=beta_u,
                method="SLSQP",
                bounds=[(-1, 10)],
            )
            try:
                assert beta_res.success
                beta = beta_res.x[0]
                beta_u = beta
            except AssertionError as err:
                print("Error maximizing wrt beta")
                print(err)
                print(beta_res.message)
                print("Bashing on regardless")
                beta = beta_res.x
                beta_u = beta

            f_old, f_new = f_new, self.f_u(s, beta, s_u, beta_u)

            if not abs((f_old - f_new) / f_new) > eps:
                break

            step += 1

        self.s = s
        self.beta = beta

        return beta_res.success

    def f(self, s, beta) -> float:
        """
        Objective function for Minorization-Maximization Algorith
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
        out += (out == 0) * 1e-5

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
        # out = -beta[0] * self.d + beta[1] * self.d**2

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
