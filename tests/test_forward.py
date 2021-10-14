"""
Tests of the generation of test data
"""

import numpy as np  # type: ignore
import pytest  # type: ignore

from flowstock import forward  # NOQA


def test_never_leave():

    np.random.seed(0)

    num_cells = 5

    pi = np.zeros(num_cells)
    s = np.ones(num_cells)
    beta = 1

    pos = np.arange(num_cells)
    d = np.abs(pos[np.newaxis, :] - pos[:, np.newaxis])

    K = 1

    sim = forward.ForwardSimulator(pi, s, beta, d, K)

    N_init = np.array([10, 0, 5, 6, 7])

    N_sim, M_sim = sim.simulate(N_init, 4)
    N_true = np.array([N_init, N_init, N_init, N_init])

    assert np.array_equal(N_sim, N_true)


def test_always_leave():

    np.random.seed(0)

    num_cells = 2

    pi = np.zeros(num_cells)
    # Always leave first region
    pi[0] = 1
    s = np.ones(num_cells)
    beta = 1

    pos = np.arange(num_cells)
    d = np.abs(pos[np.newaxis, :] - pos[:, np.newaxis])

    K = 1

    sim = forward.ForwardSimulator(pi, s, beta, d, K)

    N_init = np.array([10, 0])

    N_sim, M_sim = sim.simulate(N_init, 4)
    N_true = np.array(
        [
            [N_init[0], N_init[1]],
            [0, N_init.sum()],
            [0, N_init.sum()],
            [0, N_init.sum()],
        ]
    )

    assert np.array_equal(N_sim, N_true)


def test_forbiden_regions():
    """
    Test that people don't move further than they are allowed
    """

    np.random.seed(0)

    num_cells = 10

    pi = np.random.uniform(size=num_cells)
    s = np.random.uniform(size=num_cells)
    beta = 1

    pos = np.arange(num_cells)
    d = np.abs(pos[np.newaxis, :] - pos[:, np.newaxis])

    K = 2

    sim = forward.ForwardSimulator(pi, s, beta, d, K)

    N_init = np.random.randint(10, 100, num_cells)

    N_sim, M_sim = sim.simulate(N_init, 20)

    assert np.array_equal(M_sim, M_sim * (d <= K))


def test_gathering_scores():

    np.random.seed(0)

    num_cells = 3

    pi = np.zeros(num_cells)
    # Always leave middle region
    pi[1] = 1

    # region 2 is more attractive
    s = np.array([1, 1, 2])
    beta = 1

    pos = np.arange(num_cells)
    d = np.abs(pos[np.newaxis, :] - pos[:, np.newaxis])

    K = 2

    sim = forward.ForwardSimulator(pi, s, beta, d, K)

    N_init = np.array([0, 1000, 0])

    N_sim, M_sim = sim.simulate(N_init, 2)
    N_end = N_sim[-1]

    # Should have approximately twice as many people in region 2 as region 0
    assert N_end[0] / N_end[2] < 0.60
    assert N_end[0] / N_end[2] > 0.40


def test_no_destinations():
    """
    An exception should be raised if there is no possible destination for a cell
    """

    num_cells = 3

    pi = np.zeros(num_cells)
    # Always leave middle region
    pi[1] = 1

    s = np.ones_like(pi)
    beta = 1

    pos = np.arange(num_cells)
    d = np.abs(pos[np.newaxis, :] - pos[:, np.newaxis])

    # Distance cutoff is too short to move anywhere
    K = np.diff(pos).min() / 2

    with pytest.raises(ValueError) as _:
        _ = forward.ForwardSimulator(pi, s, beta, d, K)


def test_no_destinations_no_leave():
    """
    An exception should NOT be raised if there is no possible destination for a
    cell but leaving probability is zero.
    """

    num_cells = 3

    pi = np.zeros(num_cells)

    s = np.ones_like(pi)
    beta = 1

    pos = np.arange(num_cells)
    d = np.abs(pos[np.newaxis, :] - pos[:, np.newaxis])

    # Distance cutoff is too short to move anywhere
    K = np.diff(pos).min() / 2

    _ = forward.ForwardSimulator(pi, s, beta, d, K)
