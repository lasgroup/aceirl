import numpy as np

from typing import Tuple


def random_mdp(
    n_states: int, n_actions: int, horizon: int, non_reachable_states: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Transition model
    if non_reachable_states > 0:
        transitions = np.random.random((horizon, n_states, n_actions, n_states))
        nr_states = np.random.choice(
            range(n_states), non_reachable_states, replace=False
        )
        eps = 1e-9
        for s in nr_states:
            transitions[:, :, s] = eps
    else:
        transitions = np.random.random((horizon, n_states, n_actions, n_states))

    transitions /= transitions.sum(axis=3, keepdims=True)

    reward = np.random.random((horizon, n_states, n_actions))

    init_state_dist = np.random.random((n_states,))
    init_state_dist /= init_state_dist.sum()

    return transitions, reward, init_state_dist
