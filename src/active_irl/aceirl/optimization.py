from typing import Optional

import numpy as np
import cvxpy as cp

from active_irl.rl import value_iteration
from active_irl.util.helpers import get_hoeffding_ci, check_transitions_rewards


class Optimizer:
    """Implements the convex optimization problem at the core of AceIRL."""

    def __init__(
        self,
        delta: float = 0.1,
    ):
        self.delta = delta

    def _get_problem_params(
        self,
        P_hat: np.ndarray,
        R_hat: np.ndarray,
        init_state_dist: np.ndarray,
        epsilon: float,
        sample_count: np.ndarray,
        n_ep_per_iter: int,
        init_state_dist_target: Optional[np.ndarray] = None,
        p_target: Optional[np.ndarray] = None,
        verbose: bool = False,
        next_step_n: bool = True,
    ):
        if p_target is None:
            p_target = P_hat
        if init_state_dist_target is None:
            init_state_dist_target = init_state_dist

        horizon, n_states, n_actions = check_transitions_rewards(P_hat, R_hat)
        assert p_target.shape == (horizon, n_states, n_actions, n_states)
        assert init_state_dist_target.shape == (n_states,)

        # Get optimal policy for estimated reward (in target)
        _, V_hat, _ = value_iteration(p_target, R_hat)
        V_hat = V_hat[0] @ init_state_dist_target

        # Variables
        mu_size = horizon * n_states * n_actions
        y_size = horizon * n_states + horizon + 1
        mu = cp.Variable(mu_size, pos=True)
        y = cp.Variable(y_size)

        # b vector
        b1 = init_state_dist_target
        b2 = np.zeros(((horizon - 1) * n_states,))
        b3 = np.ones((horizon,))
        b4 = -10 * epsilon * np.ones((1,))
        b = np.concatenate([b1, b2, b3, b4]).reshape(-1)
        bb_ = np.concatenate([b1, b2, b3]).reshape(-1)

        sample_count = sample_count.reshape((horizon * n_states * n_actions,))
        sample_count = cp.maximum(sample_count, 1)

        if next_step_n:
            n_tot = sample_count + mu * n_ep_per_iter
        else:
            n_tot = sample_count

        c = 2 * cp.multiply(
            (
                2
                * cp.log(
                    24
                    * n_states
                    * n_actions
                    * horizon
                    * cp.square(sample_count)
                    / self.delta
                )
            )
            ** 0.5,
            cp.power(n_tot, -0.5),
        )

        hh = np.arange(horizon).reshape((horizon, 1, 1))
        hh = np.repeat(hh, n_states, axis=1)
        hh = np.repeat(hh, n_actions, axis=2)
        hh = np.reshape(hh, (horizon * n_states * n_actions,))
        c = cp.multiply(c, horizon - hh)
        c = cp.reshape(c, (horizon * n_states * n_actions, 1))

        # Matrix A
        Csi = np.kron(np.eye(n_states), np.ones((1, n_actions)))
        zero1 = np.zeros((n_states, n_states * n_actions))
        zero2 = np.zeros((n_states, 1))

        A10 = np.concatenate([Csi] + [zero1] * (horizon - 1) + [zero2], axis=1)
        A_list1 = [A10]
        A_list2 = [A10[:, :-1]]
        for h in range(horizon - 1):
            P_target_ = p_target[h].reshape((n_states * n_actions, n_states))
            A_target = np.concatenate(
                [zero1] * h
                + [P_target_.T, -Csi]
                + [zero1] * (horizon - 2 - h)
                + [zero2],
                axis=1,
            )
            A_list1.append(A_target)

            P_hat_ = P_hat[h].reshape((n_states * n_actions, n_states))
            A_hat = np.concatenate(
                [zero1] * h + [P_hat_.T, -Csi] + [zero1] * (horizon - 2 - h) + [zero2],
                axis=1,
            )
            A_list2.append(A_hat[:, :-1])
        A11 = np.concatenate(
            [
                np.kron(np.eye(horizon), np.ones((1, n_states * n_actions))),
                np.zeros((horizon, 1)),
            ],
            axis=1,
        )
        A_list1.append(A11)
        A_list2.append(A11[:, :-1])
        A12 = np.concatenate(
            [
                np.concatenate(
                    R_hat.reshape((horizon, n_states * n_actions)), axis=0
                ).reshape((1, -1)),
                -np.ones((1, 1)),
            ],
            axis=1,
        )
        A_list1.append(A12)

        A1 = np.concatenate(A_list1, axis=0)
        A2 = np.concatenate(A_list2, axis=0)
        A3 = np.concatenate(A_list1[:-1], axis=0)
        A4 = A_list1[-1]

        return mu, y, b, c, bb_, A1, A2, A3, A4

    def _get_policy_from_mu(self, mu: np.ndarray) -> np.ndarray:
        """Get a policy that induces visitation frequencies `mu`."""
        horizon, n_states, n_actions = mu.shape
        pi_expl = np.copy(mu)
        for timestep in range(horizon):
            for state in range(n_states):
                pi_sum = pi_expl[timestep, state].sum()
                if np.allclose(pi_sum, 0):
                    pi_expl[timestep, state, :] = 1 / n_actions
                else:
                    pi_expl[timestep, state] /= pi_sum
        return pi_expl

    def _solve_opt_problem(self, prob: cp.Problem, solver=None, dqcp: bool = False):
        if dqcp:
            assert prob.is_dqcp()

        solver_args = {"max_iters": 100000}
        if solver is not None:
            solver_args["solver"] = solver

        try:
            prob.solve(verbose=False, **solver_args)
        except cp.error.SolverError:
            try:
                prob.solve(verbose=True, **solver_args)
            except cp.error.SolverError as e:
                data, chain, inverse_data = prob.get_problem_data(solver)
                breakpoint()

        return prob.value

    def compute_pi_expl_aceirl(
        self,
        P_hat: np.ndarray,
        R_hat: np.ndarray,
        init_state_dist: np.ndarray,
        epsilon: float,
        sample_count: np.ndarray,
        n_ep_per_iter: int,
        init_state_dist_target: Optional[np.ndarray] = None,
        p_target: Optional[np.ndarray] = None,
        use_eps_const: bool = True,
        verbose: bool = False,
    ):
        """Compute exploration policy."""
        horizon, n_states, n_actions = check_transitions_rewards(P_hat, R_hat)

        mu, y, b, c, bb_, A1, A2, A3, A4 = self._get_problem_params(
            P_hat,
            R_hat,
            init_state_dist,
            epsilon,
            sample_count,
            1,
            init_state_dist_target=init_state_dist_target,
            p_target=p_target,
            verbose=verbose,
            next_step_n=True,
        )

        # Objective
        f = b.T @ y
        objective = cp.Minimize(f)

        # Constraints
        constraints = [A2 @ mu == bb_, mu >= 1e-24]

        y_ = cp.reshape(y, (y.shape[0], 1))
        if use_eps_const:
            c_ = cp.vstack([c, np.ones((1, 1))])
            constraints.append(A1.T @ y_ >= c_)
        else:
            constraints.append(A3.T @ y_ >= c)

        # Define the problem
        prob = cp.Problem(objective, constraints)

        epsilon = self._solve_opt_problem(prob, solver=cp.SCS, dqcp=True)

        mu_res = mu.value
        if mu_res is None:
            raise Exception("Optimization error: mu.value is None")

        mu_res = mu_res.reshape((horizon, n_states, n_actions))
        pi_expl = self._get_policy_from_mu(mu_res)

        if verbose:
            print("epsilon1", epsilon)

        return pi_expl

    def compute_new_eps_aceirl(
        self,
        P_hat: np.ndarray,
        R_hat: np.ndarray,
        init_state_dist: np.ndarray,
        epsilon: float,
        sample_count: np.ndarray,
        n_ep_per_iter: int,
        init_state_dist_target: Optional[np.ndarray] = None,
        p_target: Optional[np.ndarray] = None,
        use_eps_const: bool = True,
        verbose: bool = False,
    ):
        """Update epsilon."""
        horizon, n_states, n_actions = check_transitions_rewards(P_hat, R_hat)

        mu, y, b, c, bb_, A1, A2, A3, A4 = self._get_problem_params(
            P_hat,
            R_hat,
            init_state_dist,
            epsilon,
            sample_count,
            1,
            init_state_dist_target=init_state_dist_target,
            p_target=p_target,
            verbose=verbose,
            next_step_n=False,
        )

        mu2 = cp.hstack([mu, np.ones((1,))])

        # Objective
        c = cp.vstack([c, np.ones((1, 1))])
        f = c.T @ mu2
        objective = cp.Maximize(f)

        # Constraints
        y_ = cp.reshape(y, (y.shape[0], 1))

        constraints = [A3 @ mu2 == bb_, mu >= 1e-24]

        if use_eps_const:
            constraints.append(A4 @ mu2 >= -10 * epsilon)

        # Define the problem
        prob = cp.Problem(objective, constraints)

        epsilon = self._solve_opt_problem(prob)

        if verbose:
            print("epsilon2", epsilon)

        return epsilon

    def compute_allocation_travel(
        self,
        P_hat: np.ndarray,
        R_hat: np.ndarray,
        init_state_dist: np.ndarray,
        epsilon: float,
        sample_count: np.ndarray,
        n_max: int,
        init_state_dist_target: Optional[np.ndarray] = None,
        p_target: Optional[np.ndarray] = None,
        use_eps_const: bool = True,
        verbose: bool = False,
    ):
        """Compute TRAVEL allocation."""
        horizon, n_states, n_actions = check_transitions_rewards(P_hat, R_hat)

        mu, y, b, c, bb_, A1, A2, A3, A4 = self._get_problem_params(
            P_hat,
            R_hat,
            init_state_dist,
            epsilon,
            sample_count,
            1,
            init_state_dist_target=init_state_dist_target,
            p_target=p_target,
            verbose=verbose,
        )

        ones = np.ones(mu.shape)

        # Objective
        f = b.T @ y
        objective = cp.Minimize(f)

        # Constraints
        constraints = [mu >= 1e-24, ones @ mu <= n_max]

        y_ = cp.reshape(y, (y.shape[0], 1))
        if use_eps_const:
            c_ = cp.vstack([c, np.ones((1, 1))])
            constraints.append(A1.T @ y_ >= c_)
        else:
            constraints.append(A3.T @ y_ >= c)

        # Define the problem
        prob = cp.Problem(objective, constraints)

        epsilon = self._solve_opt_problem(prob, solver=cp.SCS, dqcp=True)

        mu_res = mu.value
        if mu_res is None:
            raise Exception("Optimization error: mu_res is None")

        mu_res = mu_res.reshape((horizon, n_states, n_actions))

        if verbose:
            print("epsilon1", epsilon)

        return mu_res


if __name__ == "__main__":
    from active_irl.rl import policy_evaluation

    n_states = 20
    horizon = 20
    n_actions = 10

    # np.random.seed(3)

    transitions = np.random.random((horizon, n_states, n_actions, n_states))
    transitions = transitions / transitions.sum(axis=3, keepdims=True)
    init_state_dist = np.random.random((n_states,))
    init_state_dist = init_state_dist / init_state_dist.sum()

    reward = np.random.random((horizon, n_states, n_actions))
    sample_count = np.zeros((horizon, n_states, n_actions))

    ci = get_hoeffding_ci(n_states, n_actions, horizon, sample_count, 0.1)

    q_vi, v_vi, pi_vi = value_iteration(transitions, ci)
    opt = Optimizer(0.1)

    epsilon = 0.1
    while True:
        print("epsilon", epsilon)
        pi_expl, epsilon = opt.compute_pi_expl_aceirl(
            transitions, reward, init_state_dist, epsilon, sample_count, 1
        )
        print(
            "pi_expl", policy_evaluation(transitions, ci, pi_expl)[0] @ init_state_dist
        )
        print("pi_vi", policy_evaluation(transitions, ci, pi_vi)[0] @ init_state_dist)
