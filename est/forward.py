import numpy as np  # type: ignore


class ForwardSimulator:
    def __init__(
        self, pi, s, beta, d, K,
    ):

        self.s = s
        self.pi = pi
        self.beta = beta
        self.d = d

        num_cells = d.shape[0]

        gamma = [np.where(d[i] <= K)[0] for i in range(num_cells)]
        # gamma_array_indices = np.where(d <= K)

        sexp = s * np.exp(-beta * d)

        theta = np.zeros((num_cells, num_cells))
        for i in range(num_cells):
            for j in range(num_cells):
                if i == j:
                    theta[i, j] = 1 - pi[i]
                elif i in gamma[i]:
                    theta[i, j] = (
                        pi[i] * sexp[i, j] / np.delete(sexp[i], i, axis=0).sum()
                    )
                else:
                    theta[i, j] = 0
        self.theta_cum = np.cumsum(theta, axis=1)
        assert np.allclose(self.theta_cum[..., -1], 1)

    def simulate(self, N_init, num_steps):

        num_cells = N_init.shape[0]

        N = np.zeros((num_steps, num_cells), int)
        N[0] = N_init

        M = np.zeros((num_steps - 1, num_cells, num_cells), int)

        for t in range(num_steps - 1):

            for i in range(num_cells):
                for p in range(N[t, i]):
                    rand = np.random.uniform(0, 1)

                    to_index = np.searchsorted(self.theta_cum[i], rand)

                    M[t, i, to_index] += 1

            N[t + 1] = M[t].sum(axis=0)

        return N, M
