import numpy as np

from typing import Tuple

from active_irl.util.helpers import check_transitions_rewards

P_TYPES = ["normal", "nrs", "random"]


class Gridworld:
    def __init__(
        self,
        p_obst: float = 0.8,
        horizon: int = 10,
        reward: np.ndarray = None,
        p_type: str = P_TYPES[0],
        p_fail: float = 0.1,
    ):

        self.size = 3
        self.n_states = int(self.size**2)
        self.n_actions = 4
        self.init_state = 1  # corresponding to (0, 1)
        self.goal_state = 7  # corresponding to (2, 1)
        self.horizon = horizon
        self.fail_prob = p_fail

        # Obstacle
        self.s_obst = 4  # Central cell
        self.a_obst = 1  # Right
        self.p_obst = p_obst

        self.reward = reward

        self.p_type = p_type

    def solve_mdp(self) -> np.ndarray:
        """Return the optimal policy

        Returns:
            narray: numpy array (S,)
        """
        transitions, reward, _ = self.get_mdp()
        n_states, n_actions, horizon = check_transitions_rewards(P, R)
        _, _, pi_opt = value_iteration(transitions, reward, horizon)
        return pi_opt

    def get_mdp(self) -> Tuple[int, int, np.ndarray, np.ndarray, int, np.ndarray]:
        transitions = self.get_transition_model()
        reward = self.get_reward()
        horizon = self.horizon
        init_state_dist = np.ones(self.n_states)
        init_state_dist[self.goal_state] = 0
        init_state_dist /= init_state_dist.sum()
        return transitions, reward, init_state_dist

    def get_reward(self) -> np.ndarray:
        if self.reward is not None:
            return self.reward
        reward = np.zeros((1, self.n_states, self.n_actions))
        # R[self.s_obst, :] = -1 * np.ones((self.A,))
        reward[0, self.goal_state, :] = np.ones((self.n_actions,))
        reward = np.repeat(reward, self.horizon, axis=0)
        return reward

    def get_transition_model(self) -> np.ndarray:
        models = {
            "normal": self.get_transition_model_normal,
            "nrs": self.get_transition_model_nra,
            "random": self.get_transition_model_random,
        }

        return models[self.p_type]()

    def get_transition_model_normal(self) -> np.ndarray:
        # Compute the transition probability matrix
        transitions = np.zeros(
            (self.horizon, self.n_states, self.n_actions, self.n_states)
        )

        for s in range(self.n_states):
            for a in range(self.n_actions):
                s_new = self._compute_next_state(s, a)
                if s != self.s_obst or a != self.a_obst:
                    # The agent goes in s_new if the action doesn't fail
                    transitions[:, s, a, s_new] += 1 - self.fail_prob
                else:
                    transitions[:, s, a, s_new] += (1 - self.p_obst) * (
                        1 - self.fail_prob
                    )
                    transitions[:, s, a, s] += self.p_obst * (1 - self.fail_prob)

                # Suppose now that action a fails and try all actions (including the right action)
                for a_fail in range(self.n_actions):
                    s_new = self._compute_next_state(s, a_fail)
                    transitions[:, s, a, s_new] += (
                        self.fail_prob / 4
                    )  # a_fail is taken with prob. p/4

        # The goal state is terminal -> only self-loop transitions
        transitions[:, self.goal_state, :, :] = 0
        transitions[:, self.goal_state, :, self.goal_state] = 1

        return transitions

    def get_transition_model_nra(self) -> np.ndarray:
        transitions = self.get_transition_model_normal()

        # State 2, 5 and 8 must be non reachable
        transitions[:, :, :, 2] = 0
        transitions[:, :, :, 5] = 0
        transitions[:, :, :, 8] = 0

        transitions /= transitions.sum(axis=3, keepdims=True)
        assert np.all(np.abs(transitions.sum(axis=3) - 1) < 1e-5), np.abs(
            transitions.sum(axis=3) - 1
        )
        return transitions

    def get_transition_model_random(self) -> np.ndarray:
        p = (1 / self.n_states) * np.ones(
            (self.horizon, self.n_states, self.n_actions, self.n_states)
        )
        return p

    def _compute_next_state(self, s: int, a: int) -> int:
        delta_x = [0, 1, 0, -1]  # Change in x for each action [UP, RIGHT, DOWN, LEFT]
        delta_y = [1, 0, -1, 0]  # Change in y for each action [UP, RIGHT, DOWN, LEFT]
        x, y = self._int_to_couple(s)  # Get the coordinates of s
        x_new = max(min(x + delta_x[a], self.size - 1), 0)  # Correct next-state for a
        y_new = max(min(y + delta_y[a], self.size - 1), 0)  # Correct next-state for a
        s_new = self._couple_to_int(x_new, y_new)
        return int(s_new)

    def _couple_to_int(self, x: int, y: int) -> int:
        """
        Mapping from N^2 to N.
        :param x: horizontal coordinate
        :param y: vertical coordinate
        :return: the integer identifier of the state
        """
        #     A    -----------------
        #     |    |_4_|_9_|_14|_19|
        #     |    |_3_|_8_|_13|_18|
        #  y axis  |_2_|_7_|_12|_17|
        #     |    |_1_|_6_|_11|_16|
        #     |    |_0_|_5_|_10|_15|
        #     |    -----------------
        #      -  -  - x axis - - - >

        return y + x * self.size

    def _int_to_couple(self, n: int) -> int:
        """
        Mapping from N to N^2
        :param x: horizontal coordinate
        :param y: vertical coordinate
        :return: the integer identifier of the state
        """
        #     A    -----------------
        #     |    |_4_|_9_|_14|_19|
        #     |    |_3_|_8_|_13|_18|
        #  y axis  |_2_|_7_|_12|_17|
        #     |    |_1_|_6_|_11|_16|
        #     |    |_0_|_5_|_10|_15|
        #     |    -----------------
        #      -  -  - x axis - - - >
        return np.floor(n / self.size), n % self.size
