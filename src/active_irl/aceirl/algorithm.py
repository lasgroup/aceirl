import numpy as np

from typing import Tuple, Dict, List, Optional

from active_irl.util.helpers import get_hoeffding_ci, fixed_n_rounding

from active_irl.rl import policy_evaluation, value_iteration
from active_irl.irl import MaximumEntropyIRL

from .optimization import Optimizer
from .mdp import MDP


class ActiveIRL:
    def __init__(
        self,
        delta: float,
        epsilon_stop: float,
        mdp: MDP,
        irl_model: str = "maxent",
        target_mdp: Optional[MDP] = None,
        run_irl: bool = True,
        time_dependent_reward: bool = True,
        deterministic_policy: bool = True,
    ):
        # Environment
        self.mdp = mdp
        self.target_mdp = target_mdp
        self.n_states = mdp.n_states
        self.n_actions = mdp.n_actions
        self.horizon = mdp.horizon

        # Significance
        self.delta = delta

        # Epsilon for stopping condition
        self.epsilon_stop = epsilon_stop

        # Epsilon initilization
        self.epsilon = self.horizon / 10

        # Transition model ~ estimated [H, S, A, S]
        # initialize with ones as prior
        self.P_hat = (
            np.ones((self.horizon, self.n_states, self.n_actions, self.n_states))
            / self.n_states
        )

        # Reward function ~ estimated
        self.R_hat = np.zeros((self.horizon, self.n_states, self.n_actions))

        # Value function ~ estimated
        self.V_hat = None

        # Optimal Policy ~ estimated
        self.pi_count = np.ones((self.horizon, self.n_states, self.n_actions))
        self.pi_hat = self.pi_count / self.pi_count.sum(axis=2, keepdims=True)

        # Transition sample counter
        self.P_count = np.ones(
            (self.horizon, self.n_states, self.n_actions, self.n_states)
        )
        self.P_hat = self.P_count / self.P_count.sum(axis=3, keepdims=True)

        # State-action sample counter
        self.sample_count = np.zeros((self.horizon, self.n_states, self.n_actions))

        # Which irl model to use (currently only MaxEnt implemented)
        self.irl_model = irl_model

        # Series of loss during the episode
        self.loss_data = {"loss": [], "sample": []}

        # Optimal value function of MDP target, used for loss evaluation
        if self.target_mdp is None:
            self.p_target = self.mdp.transitions
            self.r_target = self.mdp.rewards
            self.init_state_dist_target = self.mdp.init_state_dist
        else:
            self.p_target = self.target_mdp.transitions
            self.r_target = self.target_mdp.rewards
            self.init_state_dist_target = self.target_mdp.init_state_dist

        _, self.V_star, _ = value_iteration(self.p_target, self.r_target)

        # Worst policy value function
        _, _, pi_worst = value_iteration(self.p_target, -self.r_target)
        V_worst = policy_evaluation(self.p_target, self.r_target, pi_worst)
        self.V_range = (self.V_star - V_worst)[0, :] @ self.init_state_dist_target

        # If false, running reward-free exploration
        self.run_irl = run_irl

        # If set to False the reward is enforced to be the same at all timesteps
        self.time_dependent_reward = time_dependent_reward

        # If set to True, we assume we know that the policy is deterministic
        self.deterministic_policy = deterministic_policy

        self.optimizer = Optimizer(delta=self.delta)

    def get_p_target_hat(self):
        if self.target_mdp is None:
            return self.P_hat
        return self.target_mdp.transitions

    def run(
        self,
        method: str = "travel",
        n_max: int = 1000,
        n_ep_per_iter: int = 1,
        use_policyci: bool = True,
        use_eps_const: bool = True,
        record_loss: bool = False,
        verbose: bool = False,
    ) -> Dict[str, List[float]]:
        """Main Active IRL loop."""
        ep = 0
        while True:
            ep += 1

            if verbose:
                print("Ep.", ep, end=" - ")

            if method == "travel":
                samples, stop = self._iteration_travel(
                    ep, n_ep_per_iter=n_ep_per_iter, verbose=verbose
                )
            elif method == "uniform_sampling":
                samples, stop = self._iteration_uniform_sampling(
                    ep, n_ep_per_iter=n_ep_per_iter, verbose=verbose
                )
            elif method == "aceirl":
                samples, stop = self._iteration_ace(
                    ep,
                    n_ep_per_iter=n_ep_per_iter,
                    use_policyci=use_policyci,
                    verbose=verbose,
                )
            elif method == "random_exploration":
                samples, stop = self._iteration_random_expl(
                    ep,
                    n_ep_per_iter=n_ep_per_iter,
                    use_policyci=use_policyci,
                    verbose=verbose,
                )
            else:
                raise NotImplementedError()

            current_sample = self.sample_count.sum()

            if record_loss:
                loss = self._compute_loss(verbose=verbose)
                self.loss_data["loss"].append(loss)
                self.loss_data["sample"].append(current_sample)

            if verbose:
                print(f"{current_sample} samples  (n_max {n_max})")
                print()
                print("state counts:", self.sample_count.sum(axis=(0, 2)))

            if stop or current_sample >= n_max:
                return self.loss_data

        return self.loss_data

    def _compute_loss(self, verbose=False):
        """Compute the normalized regret."""

        if self.run_irl:
            r_hat = self.R_hat
        else:
            r_hat = self.r_target
        p_hat = self.get_p_target_hat()
        _, _, pi_hat = value_iteration(p_hat, r_hat)

        # Evaluate the policy found using the real reward
        V_hat = policy_evaluation(self.p_target, self.r_target, pi_hat)

        # error = np.linalg.norm((self.V_star - V_hat))
        error = (self.V_star - V_hat)[0, :] @ self.init_state_dist_target
        error /= self.V_range

        if verbose:
            print("policy error", error)
            print("r_hat", self.R_hat[0])
            print("pi_hat", pi_hat[0])
            p_error = np.mean(np.square(self.mdp.opt_pi_stoch - self.pi_hat))
            print("Policy est. error:", p_error)
            tm_error = np.mean(np.square(self.mdp.transitions - self.P_hat))
            print("Transition model eror:", tm_error)
            print(self.mdp.transitions[0, 0, 0])
            print(self.P_hat[0, 0, 0])

        return error

    def _solve_mdp(
        self, transitions: np.ndarray, rewards: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        q_opt, _, pi_opt = value_iteration(transitions, rewards)
        return pi_opt, q_opt

    def get_sample_complexity(self):
        return self.sample_count.sum()

    def _iteration_travel(
        self,
        iter: int,
        n_ep_per_iter: int = 1,
        use_eps_const: bool = True,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """Iteration of TRAVEL with generative model. [1]

        [1] Metelli, Alberto Maria, et al. "Provably efficient learning of transferable rewards." ICML 2021.
        """
        n_max = n_ep_per_iter * self.horizon

        # Compute sampling strategy
        sampling_strategy = self.optimizer.compute_allocation_travel(
            self.P_hat,
            self.R_hat,
            self.mdp.init_state_dist,
            self.epsilon,
            self.sample_count,
            n_max,
            init_state_dist_target=self.init_state_dist_target,
            p_target=self.get_p_target_hat(),
            use_eps_const=use_eps_const,
            verbose=verbose,
        )
        sampling_strategy = fixed_n_rounding(sampling_strategy, n_max)

        # Actuate sampling strategy
        samples = self._actuate_sampling_strategy(sampling_strategy)

        self._update_mdp(samples)

        epsilon = self._compute_ace_new_eps(
            n_ep_per_iter,
            use_policyci=True,
            use_eps_const=use_eps_const,
            verbose=verbose,
        )

        if verbose:
            print(f"n: {np.sum(sampling_strategy)}", end=" - ")
            print(f"epsilon: {epsilon}", end=" - ")

            if self.epsilon is not None and epsilon > self.epsilon:
                print()
                print("Old epsilon", self.epsilon)
                print("New epsilon", epsilon)
                print("ERROR")

        self.epsilon = epsilon

        if self.epsilon * 10 <= self.epsilon_stop:
            return samples, True
        return samples, False

    def _iteration_uniform_sampling(
        self, iter: int, n_ep_per_iter: int = 1, verbose: bool = False
    ) -> Tuple[np.ndarray, bool]:
        """Iteration of uniform sampling with generative model."""

        n_samples = n_ep_per_iter * self.horizon

        # Sample uniformly
        n_opt = self.horizon * self.n_states * self.n_actions
        p = np.ones(n_opt) / n_opt
        sample_allocation = np.random.multinomial(n_samples, p)
        sample_allocation = sample_allocation.reshape(
            (self.horizon, self.n_states, self.n_actions)
        ).astype(int)

        # Actuate sampling strategy
        samples = self._actuate_sampling_strategy(sample_allocation)
        self._update_mdp(samples)

        return samples, False

    def _compute_ace_exploration_policy(
        self, n_ep_per_iter, use_policyci, use_eps_const: bool, verbose=False
    ):

        if use_policyci:
            pi_expl = self.optimizer.compute_pi_expl_aceirl(
                self.P_hat,
                self.R_hat,
                self.mdp.init_state_dist,
                self.epsilon,
                self.sample_count,
                n_ep_per_iter,
                init_state_dist_target=self.init_state_dist_target,
                p_target=self.get_p_target_hat(),
                use_eps_const=use_eps_const,
                verbose=verbose,
            )
        else:
            ci = get_hoeffding_ci(
                self.n_states,
                self.n_actions,
                self.horizon,
                self.sample_count,
                self.delta,
            )
            if verbose:
                print(
                    f"CI:  min {ci.min()}, med {np.median(ci)}, mean {ci.mean()}, max {ci.max()}"
                )

            qfun_expl, vfun_expl, pi_expl = value_iteration(self.P_hat, ci)

        return pi_expl

    def _compute_ace_new_eps(
        self, n_ep_per_iter: int, use_policyci: bool, use_eps_const: bool, verbose=False
    ):

        if use_policyci:
            epsilon = self.optimizer.compute_new_eps_aceirl(
                self.P_hat,
                self.R_hat,
                self.mdp.init_state_dist,
                self.epsilon,
                self.sample_count,
                n_ep_per_iter,
                init_state_dist_target=self.init_state_dist_target,
                p_target=self.get_p_target_hat(),
                use_eps_const=use_eps_const,
                verbose=verbose,
            )
        else:
            ci = get_hoeffding_ci(
                self.n_states,
                self.n_actions,
                self.horizon,
                self.sample_count,
                self.delta,
            )
            if verbose:
                print(
                    f"CI:  min {ci.min()}, med {np.median(ci)}, mean {ci.mean()}, max {ci.max()}"
                )

            qfun_expl, vfun_expl, pi_expl = value_iteration(self.get_p_target_hat(), ci)
            epsilon = vfun_expl[0, :] @ self.init_state_dist_target

        epsilon = min(epsilon, self.horizon / 10)

        return epsilon

    def _iteration_ace(
        self,
        iter: int,
        n_ep_per_iter: int = 1,
        use_policyci: bool = False,
        use_eps_const: bool = True,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """Iteration of AceIRL."""

        # Compute exploration policy
        pi_expl = self._compute_ace_exploration_policy(
            n_ep_per_iter, use_policyci, use_eps_const, verbose=verbose
        )

        # Collect samples from exploration policyx
        samples = self._collect_samples_from_policy(pi_expl, n_episodes=n_ep_per_iter)
        self._update_mdp(samples)

        epsilon = self._compute_ace_new_eps(
            n_ep_per_iter,
            use_policyci=use_policyci,
            use_eps_const=use_eps_const,
            verbose=verbose,
        )

        if verbose:
            print(f"epsilon: {epsilon}", end=" - ")

            if self.epsilon is not None and epsilon > self.epsilon:
                print()
                print("Old epsilon", self.epsilon)
                print("New epsilon", epsilon)
                print("ERROR")

        self.epsilon = epsilon

        if self.epsilon * 10 <= self.epsilon_stop:
            return samples, True
        return samples, False

    def _iteration_random_expl(
        self,
        iter: int,
        n_ep_per_iter: int = 1,
        use_policyci: bool = False,
        use_eps_const: bool = True,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """Iteration of random exploration."""

        # Uniform exploration policy
        pi_expl = np.ones((self.horizon, self.n_states, self.n_actions)) * (
            1 / self.n_actions
        )

        # Collect samples from exploration policy
        samples = self._collect_samples_from_policy(pi_expl, n_episodes=n_ep_per_iter)
        self._update_mdp(samples)

        self.epsilon = self._compute_ace_new_eps(
            n_ep_per_iter,
            use_policyci=use_policyci,
            use_eps_const=use_eps_const,
            verbose=verbose,
        )

        if self.epsilon * 10 <= self.epsilon_stop:
            return samples, True
        return samples, False

    def _actuate_sampling_strategy(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Execute a sampling strategy for the generative model."""

        sample_p = np.zeros(
            shape=(self.horizon, self.n_states, self.n_actions, self.n_states)
        )
        sample_pi = np.zeros((self.horizon, self.n_states, self.n_actions))

        for timestep in range(self.horizon):
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    # Number of sample to query in s, a
                    n_sample = n[timestep, state, action]

                    for _ in range(n_sample):
                        next_state, optimal_action = self.mdp.query_generative_model(
                            timestep, state, action
                        )
                        sample_p[timestep, state, action, next_state] += 1
                        sample_pi[timestep, state, optimal_action] += 1

        return sample_p, sample_pi

    def _collect_samples_from_policy(
        self, policy: np.ndarray, n_episodes: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Roll out `policy` and query expert at each state."""
        sample_p = np.zeros(
            shape=(self.horizon, self.n_states, self.n_actions, self.n_states)
        )
        sample_pi = np.zeros((self.horizon, self.n_states, self.n_actions))

        for ep in range(n_episodes):
            state = self.mdp.get_initial_state()
            for timestep in range(self.horizon):
                if policy.shape == (self.horizon, self.n_states):
                    action = policy[timestep, state]
                else:
                    assert policy.shape == (self.horizon, self.n_states, self.n_actions)
                    action = np.random.choice(self.n_actions, p=policy[timestep, state])
                next_state, optimal_action = self.mdp.query_generative_model(
                    timestep, state, action
                )
                sample_p[timestep, state, action, next_state] += 1
                sample_pi[timestep, state, optimal_action] += 1
                state = next_state

        return sample_p, sample_pi

    def _update_mdp(self, samples: np.ndarray) -> None:
        """Update the estimate of the mdp."""

        sample_p, sample_pi = samples
        self._update_p(sample_p)
        self._update_pi(sample_pi)
        self._update_r()

    def _update_p(self, sample: np.ndarray) -> None:
        """Update the maximum likelihood estimate of the transition model."""
        self.P_count += sample
        self.P_hat = self.P_count / self.P_count.sum(axis=3, keepdims=True)
        self.sample_count += sample.sum(axis=3)

    def _update_pi(self, sample_pi: np.ndarray) -> None:
        """Update estimated policy."""
        assert sample_pi.shape == (self.horizon, self.n_states, self.n_actions)
        self.pi_count += sample_pi
        if self.deterministic_policy:
            for timestep in range(self.horizon):
                for state in range(self.n_states):
                    i = np.argmax(self.pi_count[timestep, state])
                    if self.pi_count[timestep, state, i] >= 2:
                        self.pi_hat[timestep, state, :] = 0
                        self.pi_hat[timestep, state, i] = 1
        else:
            self.pi_hat = self.pi_count / self.pi_count.sum(axis=2, keepdims=True)

    def _update_r(self) -> None:
        """Compute the new enstimate of the reward function using an irl algorithm."""
        if self.run_irl:
            if self.irl_model == "maxent":
                irl = MaximumEntropyIRL(
                    self.pi_hat,
                    self.P_hat,
                    self.mdp.init_state_dist,
                    self.horizon,
                    time_dependent_reward=self.time_dependent_reward,
                )
            else:
                raise NotImplementedError("Unknown IRL method:", self.irl_model)

            self.R_hat = irl.run()
