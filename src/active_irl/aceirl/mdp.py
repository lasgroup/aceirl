import numpy as np

from typing import Tuple

from active_irl.rl import value_iteration
from active_irl.util.helpers import check_transitions_rewards, ensure_policy_stochastic


class MDP:
    """Represents an environment with optional access to a generative model."""

    def __init__(
        self,
        transitions: np.ndarray,
        rewards: np.ndarray,
        init_state_dist: np.ndarray,
    ):
        self.horizon, self.n_states, self.n_actions = check_transitions_rewards(
            transitions, rewards
        )
        assert init_state_dist.shape == (self.n_states,), (
            init_state_dist.shape,
            self.n_states,
        )
        assert np.allclose(init_state_dist.sum(), 1)

        self.transitions = transitions
        self.rewards = rewards
        self.init_state_dist = init_state_dist
        _, _, self.opt_pi = value_iteration(self.transitions, self.rewards)
        self.opt_pi_stoch = ensure_policy_stochastic(
            self.opt_pi, self.horizon, self.n_states, self.n_actions
        )

    def get_initial_state(self):
        return int(np.random.choice(self.n_states, p=self.init_state_dist))

    def get_next_state(self, timestep: int, state: int, action: int) -> int:
        return int(
            np.random.choice(self.n_states, p=self.transitions[timestep, state, action])
        )

    def query_generative_model(
        self, timestep: int, state: int, action: int
    ) -> Tuple[int, int]:
        next_state = np.random.choice(
            self.n_states, p=self.transitions[timestep, state, action]
        )
        optimal_action = self.opt_pi[timestep, state]
        return next_state, optimal_action
