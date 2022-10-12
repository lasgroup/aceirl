import numpy as np

from typing import Tuple, Optional

from active_irl.util.helpers import (
    check_transitions_rewards,
    ensure_policy_stochastic,
    random_argmax,
)


def value_iteration(
    transitions: np.ndarray,
    rewards: np.ndarray,
    policy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform value iteration"""
    horizon, n_states, n_actions = check_transitions_rewards(transitions, rewards)

    if policy is not None:
        policy = ensure_policy_stochastic(policy, horizon, n_states, n_actions)

    q_opt = np.zeros((horizon, n_states, n_actions))
    q_opt[horizon - 1] = rewards[horizon - 1]

    for h in range(horizon - 1, 0, -1):
        if policy is not None:
            vh = np.sum(q_opt[h] * policy[h], axis=1)
        else:
            vh = np.max(q_opt[h], axis=1)
        q_opt[h - 1] = rewards[h - 1] + transitions[h - 1] @ vh

    if policy is not None:
        v_opt = np.sum(q_opt * policy, axis=2)
        pi_opt = policy
    else:
        v_opt = np.max(q_opt, axis=2)
        pi_opt = random_argmax(q_opt, axis=2)

    return q_opt, v_opt, pi_opt


def policy_evaluation(
    transitions: np.ndarray,
    rewards: np.ndarray,
    policy: np.ndarray,
):
    _, v, _ = value_iteration(transitions, rewards, policy=policy)
    return v
