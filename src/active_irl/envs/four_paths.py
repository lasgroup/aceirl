import numpy as np

from typing import Tuple


def four_paths_mdp(n_each_dir=10, horizon=20) -> Tuple[np.ndarray, np.ndarray]:
    n_actions = 4
    n_states = n_actions * n_each_dir + 1

    p_fail = 0.3 * np.random.random((4,))

    transitions = np.zeros((horizon, n_states, n_actions, n_states))

    for state in range(n_states):
        for action in range(n_actions):
            transitions[:, state, action, state] = 1

    for d in range(4):
        s_start = 1 + n_each_dir * d
        transitions[:, 0, d, s_start] = p_fail[d]
        transitions[:, 0, d, 0] = 1 - p_fail[d]

        for i in range(n_each_dir - 1):
            s = s_start + i
            if i == 0:
                prev_s = s_start
            else:
                prev_s = s - 1
            if i == n_each_dir - 2:
                next_s = s
            else:
                next_s = s + 1

            transitions[:, s, d, :] = 0
            transitions[:, s, d, prev_s] += p_fail[d]
            transitions[:, s, d, next_s] += 1 - p_fail[d]

    goal = (np.random.randint(4) + 1) * n_each_dir - 1
    reward = np.zeros((horizon, n_states, n_actions))
    reward[:, goal, :] = 1

    init_state_dist = np.zeros(n_states)
    init_state_dist[0] = 1

    return transitions, reward, init_state_dist
