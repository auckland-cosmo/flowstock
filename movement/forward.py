import numpy as np  # type: ignore


class ForwardSimulator:
    def __init__(
        self, pi, s, beta, d, K,
    ):

        self.s = s
        self.pi = pi
        self.beta = beta
        self.d = d

        self.noise_amplitude = 0.0

        num_cells = d.shape[0]

        gamma = d <= K

        sexp = s[..., np.newaxis] * np.exp(-beta * d) * gamma

        self.theta = np.zeros((num_cells, num_cells))
        for i in range(num_cells):
            for j in range(num_cells):
                if i == j:
                    self.theta[i, j] = 1 - pi[i]
                elif d[i, j] <= K:
                    self.theta[i, j] = (
                        pi[i] * sexp[i, j] / np.delete(sexp[i], i, axis=0).sum()
                    )
                else:
                    self.theta[i, j] = 0
        self.theta_cum = np.cumsum(self.theta, axis=1)

        assert np.allclose(self.theta_cum[..., -1], 1)

    def simulate(self, N_init, num_steps):

        num_cells = N_init.shape[0]

        N = np.zeros((num_steps, num_cells), int)
        N[0] = N_init

        M = np.zeros((num_steps - 1, num_cells, num_cells), int)

        for t in range(num_steps - 1):

            # Loop over cells
            for i in range(num_cells):

                amp = int(self.noise_amplitude * N[t, i])

                if amp == 0:
                    people = N[t, i]
                else:
                    noise = max(np.random.randint(-amp, amp), -N[t, i])
                    people = N[t, i] + noise

                # Loop over people in cell after noise added
                for p in range(people):
                    rand = np.random.uniform(0, 1)

                    to_index = np.searchsorted(self.theta_cum[i], rand)

                    M[t, i, to_index] += 1

            N[t + 1] = M[t].sum(axis=0)

        return N, M
