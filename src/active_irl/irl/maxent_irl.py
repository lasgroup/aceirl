import numpy.linalg as la
import numpy as np

from active_irl.util.helpers import check_transitions_rewards, ensure_policy_stochastic


class MaximumEntropyIRL(object):
    """Implementation of MaxEnt IRL [1].

    [1] Ziebart, Brian D., et al. "Maximum Entropy Inverse Reinforcement Learning." AAAI 2008.
    """

    eps = 1e-24

    def __init__(
        self,
        optimal_policy: np.ndarray,
        transitions: np.ndarray,
        init_state_dist: np.ndarray,
        horizon: int,
        learning_rate: float = 1,
        max_iter: int = 100,
        gradient_method: str = "linear",
        beta: float = 1,
        regularizer: float = 0.01,
        time_dependent_reward: bool = True,
    ):
        self.horizon = horizon
        self.transitions = transitions
        self.n_states = transitions.shape[1]
        self.n_actions = transitions.shape[2]

        self.optimal_policy = ensure_policy_stochastic(
            optimal_policy, self.horizon, self.n_states, self.n_actions
        )

        self.init_state_dist = init_state_dist
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.beta = beta
        self.regularizer = regularizer
        self.time_dependent_reward = time_dependent_reward

        if not gradient_method in ["linear", "exponentiated"]:
            raise ValueError()

        self.gradient_method = gradient_method

        self.n_states_actions = self.n_states * self.n_actions

    def _compute_feature_expectations(self, policy: np.ndarray) -> np.ndarray:
        D_st = np.zeros((self.horizon, self.n_states, self.n_actions))
        D_st[0] = np.einsum("i,ij->ij", self.init_state_dist, policy[0])
        for h in np.arange(self.horizon - 1):
            D_st[h + 1] = np.einsum(
                "ik,ikj,jl->jl", D_st[h], self.transitions[h], policy[h + 1]
            )
        return D_st

    def _soft_value_iteration(self, reward: np.ndarray) -> np.ndarray:
        pi_ = np.zeros((self.horizon, self.n_states, self.n_actions))
        Q = np.zeros((self.horizon, self.n_states, self.n_actions))
        Q[self.horizon - 1] = reward[self.horizon - 1]
        pi_[self.horizon - 1] = np.exp(self.beta * Q[self.horizon - 1])
        pi_[self.horizon - 1] /= pi_[self.horizon - 1].sum(axis=1, keepdims=True)
        for h in np.arange(self.horizon - 1, 0, -1):
            Q[h - 1] = reward[h - 1]
            Q[h - 1] += np.einsum(
                "ikj,jl,jl->ik", self.transitions[h - 1], pi_[h], Q[h]
            )
            pi_[h - 1] = np.exp(self.beta * Q[h - 1])
            pi_[h - 1] /= pi_[h - 1].sum(axis=1, keepdims=True)
        return pi_

    def compute_expected_state_visitation_frequency(
        self, reward: np.ndarray
    ) -> np.ndarray:
        # To avoid bad arithmetic approximations
        reward -= reward.max()
        reward_exp = np.exp(reward)
        reward_exp /= reward_exp.sum()
        reward_exp = reward_exp.reshape((self.horizon, self.n_states, self.n_actions))

        reward_exp = reward.reshape((self.horizon, self.n_states, self.n_actions))

        pi_ = self._soft_value_iteration(reward_exp)

        return self._compute_feature_expectations(pi_)

    def compute_feature_expectations(self) -> np.ndarray:
        return self._compute_feature_expectations(self.optimal_policy)

    def run(self, verbose: bool = False) -> np.ndarray:

        # Compute features expectations
        feature_expectations = self.compute_feature_expectations()  # n_features

        # Weights initialization
        if self.time_dependent_reward:
            n_features = self.horizon * self.n_states * self.n_actions
        else:
            n_features = self.n_states * self.n_actions
        reward = np.ones(n_features) / n_features

        # Gradient descent
        for i in range(self.max_iter):
            if verbose:
                print("Iteration %s/%s" % (i + 1, self.max_iter))

            if self.time_dependent_reward:
                reward_ = reward
            else:
                reward_ = reward.reshape((1, self.n_states, self.n_actions))
                reward_ = np.repeat(reward_, self.horizon, axis=0)
                reward_ = reward_.reshape((-1,))

            expected_vf = self.compute_expected_state_visitation_frequency(reward_)

            if self.time_dependent_reward:
                gradient = feature_expectations - expected_vf
            else:
                gradient = feature_expectations.mean(axis=0) - expected_vf.mean(axis=0)

            if verbose:
                print("Iteration", i)
                print(
                    "Observed feature expectations:",
                    feature_expectations.sum(axis=(0, 2)),
                )
                print("Expected feature expectations:", expected_vf.sum(axis=(0, 2)))
                print("Gradient magnitude:", np.sum(np.square(gradient)))

            if self.gradient_method == "linear":
                reward += self.learning_rate * (
                    gradient.reshape(-1) - self.regularizer * reward
                )
            else:
                exponential = np.exp(self.learning_rate * gradient)
                reward = reward * exponential / np.dot(reward, exponential).sum()

        if self.time_dependent_reward:
            reward = reward.reshape((self.horizon, self.n_states, self.n_actions))
        else:
            reward = reward.reshape((1, self.n_states, self.n_actions))
            reward = np.repeat(reward, self.horizon, axis=0)

        reward -= np.min(reward)
        m = np.max(np.abs(reward))
        if m > 0:
            reward /= m

        return reward


if __name__ == "__main__":
    from active_irl.rl import value_iteration, policy_evaluation
    from active_irl.envs import chain_mdp

    n_states = 20
    horizon = 10
    n_actions = 10

    np.random.seed(3)

    transitions = np.random.random((horizon, n_states, n_actions, n_states))
    transitions = transitions / transitions.sum(axis=3, keepdims=True)
    init_state_dist = np.random.random((n_states,))
    init_state_dist = init_state_dist / init_state_dist.sum()

    reward = np.random.random((1, n_states, n_actions))
    reward = np.repeat(reward, horizon, axis=0)

    _, v_opt, optimal_policy = value_iteration(transitions, reward)

    irl = MaximumEntropyIRL(
        optimal_policy,
        transitions,
        init_state_dist,
        horizon,
        learning_rate=1,
        max_iter=100,
        beta=1,
    )
    inferred_reward = irl.run(verbose=True)

    _, _, inferred_policy = value_iteration(transitions, inferred_reward)
    v_inferred = policy_evaluation(transitions, reward, inferred_policy)

    print("Optimal policy:", v_opt[0] @ init_state_dist)
    print("Inferred policy:", v_inferred[0] @ init_state_dist)
    print("Reward error", np.sum(np.square(reward, inferred_reward)))
    print("Policy error:", np.sum(optimal_policy != inferred_policy))
