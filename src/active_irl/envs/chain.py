import numpy as np

from typing import Tuple


def chain_mdp(
    n_states: int, n_actions: int, horizon: int, p_fail: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    transitions = np.zeros((horizon, n_states, n_actions, n_states))

    # States in the chain
    states_in_chain = list(range(n_states - 1))
    negative_state = n_states - 1
    good_action = n_actions - 1
    bad_actions = list(range(0, n_actions - 1))

    for s in states_in_chain:
        s_new = min(s + 1, states_in_chain[-1])
        transitions[:, s, good_action, s_new] = 1 - p_fail
        transitions[:, s, good_action, negative_state] = p_fail

        for a in bad_actions:
            transitions[:, s, a, negative_state] = 1 - p_fail
            transitions[:, s, a, s_new] = p_fail

    # If the agent perfoms good_action in the negative state can come back to the chain with a small probability
    transitions[:, negative_state, good_action, states_in_chain[0]] = 0.05
    transitions[:, negative_state, good_action, negative_state] = 1 - 0.05

    # If the agent perfoms a bad action in the negative state remains in the negative states
    for bad_a in bad_actions:
        transitions[:, negative_state, bad_a, negative_state] = 1 - 0.01
        transitions[:, negative_state, bad_a, states_in_chain[0]] = 0.01

    reward = np.ones((horizon, n_states, n_actions))
    reward[:, negative_state, :] = 0

    init_state_dist = np.ones(n_states) / n_states

    return transitions, reward, init_state_dist
