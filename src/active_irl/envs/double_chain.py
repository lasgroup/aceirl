import numpy as np

from typing import Tuple


def double_chain_mdp() -> Tuple[np.ndarray, np.ndarray]:
    n_states = 31
    n_actions = 2
    horizon = 20
    p_fail = 0.1
    goal = n_states - 1
    start = int(n_states / 2)

    left, right = 0, 1

    transitions = np.zeros((horizon, n_states, n_actions, n_states))

    for state in range(n_states):
        l_state, r_state = max(0, state - 1), min(n_states - 1, state + 1)
        transitions[:, state, left, l_state] = 1 - p_fail
        transitions[:, state, left, r_state] = p_fail
        transitions[:, state, right, r_state] = 1 - p_fail
        transitions[:, state, right, l_state] = p_fail

    reward = np.zeros((horizon, n_states, n_actions))
    reward[:, goal, :] = 1

    init_state_dist = np.zeros(n_states)
    init_state_dist[start] = 1

    return transitions, reward, init_state_dist
