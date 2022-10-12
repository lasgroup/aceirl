import os
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from active_irl.envs import (
    random_mdp,
    chain_mdp,
    four_paths_mdp,
    double_chain_mdp,
    Gridworld,
)
from active_irl.rl import value_iteration, policy_evaluation
from active_irl.util.helpers import check_transitions_rewards
from active_irl.aceirl import ActiveIRL, MDP

from sacred import Experiment

ex = Experiment("aceirl-experiment")


@ex.named_config
def test():
    env = {"name": "four_paths"}
    n_max = 50000
    n_ep_per_iter = 1000
    uniform_sampling = False  # True
    random_exploration = False
    irl_models_travel = None  # ["maxent"]
    irl_models_ace = None  # ["maxent"]
    irl_models_ace_ci = ["maxent"]
    irl_models_ace_ci_no_eps = None
    n_runs = 50
    results_file = f"results/result_random_mdp_{n_runs}runs.csv"
    verbose = True


@ex.named_config
def double_chain():
    env = {"name": "double_chain"}
    n_max = 100000
    n_ep_per_iter = 100
    uniform_sampling = True
    random_exploration = True
    irl_models_travel = ["maxent"]
    irl_models_ace = ["maxent"]
    irl_models_ace_ci = ["maxent"]
    irl_models_ace_ci_no_eps = None
    transfer_reward = False
    reward_free_expl = False
    n_runs = 50
    results_file = f"results/result_double_chain_{n_runs}runs.csv"


@ex.named_config
def double_chain_rfe():
    env = {"name": "double_chain"}
    n_max = 1000000
    n_ep_per_iter = 1000
    uniform_sampling = True
    random_exploration = True
    irl_models_travel = None
    irl_models_ace = ["maxent"]
    irl_models_ace_ci = None
    irl_models_ace_ci_no_eps = ["maxent"]
    transfer_reward = False
    reward_free_expl = True
    n_runs = 50
    results_file = f"results/result_double_chain_rfe_{n_runs}runs.csv"


@ex.named_config
def four_paths():
    env = {"name": "four_paths"}
    n_max = 100000
    n_ep_per_iter = 100
    uniform_sampling = True
    random_exploration = True
    irl_models_travel = ["maxent"]
    irl_models_ace = ["maxent"]
    irl_models_ace_ci = ["maxent"]
    irl_models_ace_ci_no_eps = None
    transfer_reward = False
    reward_free_expl = False
    n_runs = 50
    results_file = f"results/result_four_paths_mdp_{n_runs}runs.csv"


@ex.named_config
def four_paths_rfe():
    env = {"name": "four_paths"}
    n_max = 100000
    n_ep_per_iter = 100
    uniform_sampling = True
    random_exploration = True
    irl_models_travel = None
    irl_models_ace = ["maxent"]
    irl_models_ace_ci = None
    irl_models_ace_ci_no_eps = ["maxent"]
    transfer_reward = False
    reward_free_expl = True
    n_runs = 50
    results_file = f"results/result_four_paths_mdp_rfe_{n_runs}runs.csv"


@ex.named_config
def random_env():
    env = {"name": "random", "n_states": 9, "n_actions": 4, "horizon": 10}
    n_max = 3000
    n_ep_per_iter = 1
    uniform_sampling = True
    random_exploration = True
    irl_models_travel = ["maxent"]
    irl_models_ace = ["maxent"]
    irl_models_ace_ci = ["maxent"]
    irl_models_ace_ci_no_eps = None
    transfer_reward = False
    reward_free_expl = False
    n_runs = 50
    results_file = f"results/result_random_mdp_{n_runs}runs.csv"


@ex.named_config
def random_env_transfer():
    env = {"name": "random", "n_states": 9, "n_actions": 4, "horizon": 10}
    n_max = 3000
    n_ep_per_iter = 1
    uniform_sampling = True
    random_exploration = True
    irl_models_travel = ["maxent"]
    irl_models_ace = ["maxent"]
    irl_models_ace_ci = ["maxent"]
    irl_models_ace_ci_no_eps = None
    transfer_reward = True
    reward_free_expl = False
    n_runs = 50
    results_file = f"results/result_random_mdp_transfer_{n_runs}runs.csv"


@ex.named_config
def chain():
    env = {"name": "chain", "n_states": 6, "n_actions": 10, "horizon": 10}
    n_max = 10000
    n_ep_per_iter = 1
    uniform_sampling = True
    random_exploration = True
    irl_models_travel = ["maxent"]
    irl_models_ace = ["maxent"]
    irl_models_ace_ci = ["maxent"]
    irl_models_ace_ci_no_eps = None
    transfer_reward = False
    reward_free_expl = False
    n_runs = 50
    results_file = f"results/result_chain_mdp_{n_runs}runs.csv"


@ex.named_config
def gridworld():
    env = {
        "name": "gridworld",
        "p_obst": 0.8,
        "p_fail": 0.3,
        "p_obst_t": 0,
        "horizon": 10,
    }
    n_max = 3000
    n_ep_per_iter = 1
    uniform_sampling = True
    random_exploration = True
    irl_models_travel = ["maxent"]
    irl_models_ace = ["maxent"]
    irl_models_ace_ci = ["maxent"]
    irl_models_ace_ci_no_eps = None
    transfer_reward = False
    reward_free_expl = False
    n_runs = 50
    results_file = f"results/result_gridworld_{n_runs}runs.csv"


@ex.named_config
def gridworld_transfer():
    env = {
        "name": "gridworld",
        "p_obst": 0.8,
        "p_fail": 0.3,
        "p_obst_t": 0,
        "horizon": 10,
    }
    n_max = 3000
    n_ep_per_iter = 1
    uniform_sampling = True
    random_exploration = True
    irl_models_travel = ["maxent"]
    irl_models_ace = ["maxent"]
    irl_models_ace_ci = ["maxent"]
    irl_models_ace_ci_no_eps = None
    transfer_reward = True
    reward_free_expl = False
    n_runs = 50
    results_file = f"results/result_gridworld_transfer_{n_runs}runs.csv"


@ex.config
def config():
    env = None
    n_max = None
    n_ep_per_iter = None
    uniform_sampling = False
    random_exploration = False
    irl_models_travel = None
    irl_models_ace = None
    irl_models_ace_ci = None
    irl_models_ace_ci_no_eps = None
    transfer_reward = False
    reward_free_expl = False
    n_runs = None
    results_file = None
    verbose = False
    n_jobs = 1


def evaluate_loss_vs_sample(
    method: str,
    mdp: MDP,
    horizon: int,
    n_max: int = 20,
    n_ep_per_iter: int = 1,
    irl_model: str = "maxent",
    use_policyci: bool = False,
    use_eps_const: bool = True,
    delta: float = 0.1,
    epsilon_stop: float = 0.01,
    target_mdp: Optional[MDP] = None,
    reward_free_expl=False,
    verbose: bool = False,
) -> Dict[str, List[float]]:
    """Auxiliary function to run experiment."""

    active_irl = ActiveIRL(
        delta=delta,
        epsilon_stop=epsilon_stop,
        mdp=mdp,
        irl_model=irl_model,
        target_mdp=target_mdp,
        run_irl=not reward_free_expl,
    )
    loss_data = active_irl.run(
        verbose=verbose,
        record_loss=True,
        n_max=n_max,
        n_ep_per_iter=n_ep_per_iter,
        method=method,
        use_policyci=use_policyci,
        use_eps_const=use_eps_const,
    )
    return loss_data


def run_experiment(
    i: int,
    env: MDP,
    n_max: int,
    n_ep_per_iter: int,
    uniform_sampling: bool,
    irl_models_travel: List[str],
    irl_models_ace: List[str],
    irl_models_ace_ci: List[str],
    irl_models_ace_ci_no_eps: List[str],
    random_exploration: bool,
    transfer_reward: bool,
    reward_free_expl: bool,
    verbose: bool,
):
    print("Experiment", i, end=" ")
    np.random.seed(i)
    target_mdp = None

    if env["name"] == "random":
        n_states, n_actions, horizon = (
            env["n_states"],
            env["n_actions"],
            env["horizon"],
        )
        transitions, reward, init_state_dist = random_mdp(n_states, n_actions, horizon)

        if transfer_reward:
            target_transitions, _, target_init_state_dist = random_mdp(
                n_states, n_actions, horizon
            )
            target_mdp = MDP(target_transitions, reward, target_init_state_dist)
    elif env["name"] == "chain":
        if transfer_reward:
            raise NotImplementedError("Chain cannot be used for transfer experiment.")
        n_states, n_actions, horizon = (
            env["n_states"],
            env["n_actions"],
            env["horizon"],
        )
        transitions, reward, init_state_dist = chain_mdp(n_states, n_actions, horizon)
    elif env["name"] == "gridworld":
        p_obst, p_fail, p_obst_t, horizon = (
            env["p_obst"],
            env["p_fail"],
            env["p_obst_t"],
            env["horizon"],
        )
        grid_src = Gridworld(p_obst=p_obst, p_fail=p_fail)
        transitions, reward, init_state_dist = grid_src.get_mdp()

        if transfer_reward:
            grid_src = Gridworld(p_obst=0, p_fail=p_fail)
            target_transitions, _, _ = grid_src.get_mdp()
            target_mdp = MDP(target_transitions, reward, init_state_dist)
    elif env["name"] == "four_paths":
        if transfer_reward:
            raise NotImplementedError(
                "Four Paths cannot be used for transfer experiment."
            )

        transitions, reward, init_state_dist = four_paths_mdp()
        horizon, n_states, n_actions = check_transitions_rewards(transitions, reward)
    elif env["name"] == "double_chain":
        if transfer_reward:
            raise NotImplementedError(
                "DoubleChain cannot be used for transfer experiment."
            )

        transitions, reward, init_state_dist = double_chain_mdp()
        horizon, n_states, n_actions = check_transitions_rewards(transitions, reward)
    else:
        raise Exception("Unknown environment:", env)

    if transfer_reward:
        assert target_mdp is not None
    mdp = MDP(transitions, reward, init_state_dist)

    results = dict()
    if uniform_sampling:
        print("Uniform sampling...", end=" ", flush=True)
        results["losses_us"] = evaluate_loss_vs_sample(
            "uniform_sampling",
            mdp,
            horizon,
            n_max=n_max,
            n_ep_per_iter=n_ep_per_iter,
            target_mdp=target_mdp,
            reward_free_expl=reward_free_expl,
            verbose=verbose,
        )
    if irl_models_travel is not None and len(irl_models_travel) > 0:
        if reward_free_expl:
            raise NotImplementedError("TRAVEL cannot be used for RFE.")

        print("TRAVEL: ", end=" ", flush=True)
        for irl_model in irl_models_travel:
            print(f"{irl_model}...", end=" ", flush=True)
            results[f"losses_{irl_model}"] = evaluate_loss_vs_sample(
                "travel",
                mdp,
                horizon,
                n_max=n_max,
                n_ep_per_iter=n_ep_per_iter,
                irl_model=irl_model,
                use_policyci=True,
                use_eps_const=True,
                target_mdp=target_mdp,
                reward_free_expl=reward_free_expl,
                verbose=verbose,
            )
    if irl_models_ace is not None and len(irl_models_ace) > 0:
        print("AceIRL: ", end=" ", flush=True)
        for irl_model in irl_models_ace:
            print(f"{irl_model}...", end=" ", flush=True)
            results[f"losses_ace_{irl_model}"] = evaluate_loss_vs_sample(
                "aceirl",
                mdp,
                horizon,
                n_max=n_max,
                n_ep_per_iter=n_ep_per_iter,
                irl_model=irl_model,
                use_policyci=False,
                target_mdp=target_mdp,
                reward_free_expl=reward_free_expl,
                verbose=verbose,
            )
    if irl_models_ace_ci is not None and len(irl_models_ace_ci) > 0:
        if reward_free_expl:
            raise NotImplementedError("AceIRL w/ policyci cannot be used for RFE.")

        print("AceIRL (w/ policyci): ", end=" ", flush=True)
        for irl_model in irl_models_ace_ci:
            print(f"{irl_model}...", end=" ", flush=True)
            results[f"losses_ace_ci_{irl_model}"] = evaluate_loss_vs_sample(
                "aceirl",
                mdp,
                horizon,
                n_max=n_max,
                n_ep_per_iter=n_ep_per_iter,
                irl_model=irl_model,
                use_policyci=True,
                use_eps_const=True,
                target_mdp=target_mdp,
                reward_free_expl=reward_free_expl,
                verbose=verbose,
            )
    if irl_models_ace_ci_no_eps is not None and len(irl_models_ace_ci_no_eps) > 0:
        print("AceIRL (w/ opt, w/o eps): ", end=" ", flush=True)
        for irl_model in irl_models_ace_ci_no_eps:
            print(f"{irl_model}...", end=" ", flush=True)
            results[f"losses_ace_ci_no_eps_{irl_model}"] = evaluate_loss_vs_sample(
                "aceirl",
                mdp,
                horizon,
                n_max=n_max,
                n_ep_per_iter=n_ep_per_iter,
                irl_model=irl_model,
                use_policyci=True,
                use_eps_const=False,
                target_mdp=target_mdp,
                reward_free_expl=reward_free_expl,
                verbose=verbose,
            )
    if random_exploration:
        print("Random exploration...", end=" ", flush=True)
        results["losses_rand_expl"] = evaluate_loss_vs_sample(
            "random_exploration",
            mdp,
            horizon,
            n_max=n_max,
            n_ep_per_iter=n_ep_per_iter,
            use_policyci=False,
            target_mdp=target_mdp,
            reward_free_expl=reward_free_expl,
            verbose=verbose,
        )
    print()

    return results


@ex.automain
def main(
    _run,
    env: MDP,
    n_max: int,
    n_ep_per_iter: int,
    uniform_sampling: bool,
    random_exploration: bool,
    irl_models_travel: List[str],
    irl_models_ace: List[str],
    irl_models_ace_ci: List[str],
    irl_models_ace_ci_no_eps: List[str],
    transfer_reward: bool,
    reward_free_expl: bool,
    n_runs: int,
    results_file: str,
    verbose: bool,
    n_jobs: int,
    seed,
):
    os.makedirs("results", exist_ok=True)

    if n_jobs <= 1:
        result_list = []
        for i in range(n_runs):
            result_list.append(
                run_experiment(
                    i,
                    env,
                    n_max,
                    n_ep_per_iter,
                    uniform_sampling,
                    irl_models_travel,
                    irl_models_ace,
                    irl_models_ace_ci,
                    irl_models_ace_ci_no_eps,
                    random_exploration,
                    transfer_reward,
                    reward_free_expl,
                    verbose,
                )
            )
    else:
        from functools import partial
        from multiprocessing import Pool

        with Pool(n_jobs + 1) as p:
            idx = list(range(n_runs))
            fct = partial(
                run_experiment,
                env=env,
                n_max=n_max,
                n_ep_per_iter=n_ep_per_iter,
                uniform_sampling=uniform_sampling,
                irl_models_travel=irl_models_travel,
                irl_models_ace=irl_models_ace,
                irl_models_ace_ci=irl_models_ace_ci,
                irl_models_ace_ci_no_eps=irl_models_ace_ci_no_eps,
                random_exploration=random_exploration,
                transfer_reward=transfer_reward,
                reward_free_expl=reward_free_expl,
                verbose=verbose,
            )
            result_list = p.map(fct, idx)

    result_dict = dict()
    for i, results in enumerate(result_list):
        if uniform_sampling:
            result_dict[f"loss_us_{i}"] = results["losses_us"]["loss"]
            result_dict[f"sample_us_{i}"] = results["losses_us"]["sample"]
        if irl_models_travel is not None and len(irl_models_travel) > 0:
            for irl_model in irl_models_travel:
                result_dict[f"loss_travel_{irl_model}_{i}"] = results[
                    f"losses_{irl_model}"
                ]["loss"]
                result_dict[f"sample_travel_{irl_model}_{i}"] = results[
                    f"losses_{irl_model}"
                ]["sample"]
        if irl_models_ace is not None and len(irl_models_ace) > 0:
            for irl_model in irl_models_ace:
                result_dict[f"loss_ace_{irl_model}_{i}"] = results[
                    f"losses_ace_{irl_model}"
                ]["loss"]
                result_dict[f"sample_ace_{irl_model}_{i}"] = results[
                    f"losses_ace_{irl_model}"
                ]["sample"]
        if irl_models_ace_ci is not None and len(irl_models_ace_ci) > 0:
            for irl_model in irl_models_ace_ci:
                result_dict[f"loss_ace_ci_{irl_model}_{i}"] = results[
                    f"losses_ace_ci_{irl_model}"
                ]["loss"]
                result_dict[f"sample_ace_ci_{irl_model}_{i}"] = results[
                    f"losses_ace_ci_{irl_model}"
                ]["sample"]
        if irl_models_ace_ci_no_eps is not None and len(irl_models_ace_ci_no_eps) > 0:
            for irl_model in irl_models_ace_ci_no_eps:
                result_dict[f"loss_ace_ci_no_eps_{irl_model}_{i}"] = results[
                    f"losses_ace_ci_no_eps_{irl_model}"
                ]["loss"]
                result_dict[f"sample_ace_ci_no_eps_{irl_model}_{i}"] = results[
                    f"losses_ace_ci_no_eps_{irl_model}"
                ]["sample"]
        if random_exploration:
            result_dict[f"loss_rand_expl_{i}"] = results["losses_rand_expl"]["loss"]
            result_dict[f"sample_rand_expl_{i}"] = results["losses_rand_expl"]["sample"]

        # make sure all columns are same length (necessary to create data frame)
        maxlen = max([len(v) for v in result_dict.values()])
        for k, v in result_dict.items():
            result_dict[k] += [None] * (maxlen - len(result_dict[k]))

        result = pd.DataFrame(data=result_dict)
        result.to_csv(results_file)
