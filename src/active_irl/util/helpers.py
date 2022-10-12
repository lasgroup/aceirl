import numpy as np

from typing import Tuple


def random_argmax(array, axis=None):
    """Argmax with random tie breaking."""
    return np.argmax(
        np.random.random(array.shape)
        * (array == np.amax(array, axis=axis, keepdims=True)),
        axis=axis,
    )


def check_transitions_rewards(
    transitions: np.ndarray, rewards: np.ndarray
) -> Tuple[int, int]:
    assert len(rewards.shape) == 3, "Rewards need to have shape [H, S, A]"

    horizon, n_states, n_actions = rewards.shape

    assert transitions.shape == (
        horizon,
        n_states,
        n_actions,
        n_states,
    ), "Transitions need to have shape [H, S, A, S]"

    assert np.allclose(transitions.sum(axis=3), 1)

    return horizon, n_states, n_actions


def ensure_policy_stochastic(policy, horizon, n_states, n_actions):
    if policy.shape == (horizon, n_states):
        policy = np.eye(n_actions)[policy]
    assert policy.shape == (horizon, n_states, n_actions)
    return policy


def get_hoeffding_ci(
    n_states: int, n_actions: int, horizon: int, sample_count: np.ndarray, delta: float
) -> np.ndarray:
    sample_count = np.maximum(sample_count, 1)
    ci = 2 * np.sqrt(
        2
        * np.log(24 * n_states * n_actions * horizon * np.square(sample_count) / delta)
        / sample_count
    )
    ci *= horizon - np.arange(horizon).reshape((horizon, 1, 1))
    return ci


def fixed_n_rounding(allocation, N):
    """Rounding procedure to ensure the sum of allocation is N."""
    allocation_shape = allocation.shape
    allocation = allocation.reshape((-1,))
    allocation = np.ceil(allocation).astype(int)
    support = allocation > 1e-2
    where_support = np.where(support)[0]
    while np.sum(allocation) != N:
        if np.sum(allocation) > N:
            j = random_argmax(allocation[support])
            allocation[where_support[j]] -= 1
        else:  # np.sum(allocation) < N
            j = random_argmax(-allocation[support])
            allocation[where_support[j]] += 1
    allocation = allocation.reshape(allocation_shape)
    return allocation
