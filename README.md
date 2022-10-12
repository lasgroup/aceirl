# Active Exploration for Inverse Reinforcement Learning (AceIRL)

This repository contains code for the paper ["Active Exploration for Inverse Reinforcement Learning"](https://arxiv.org/abs/2207.08645). Here, we describe how to reproduce the experiments presented in the paper.


### Citation

David Lindner, Andreas Krause, and Giorgia Ramponi. **Active Exploration for Inverse Reinforcement Learning**. In _Conference on Neural Information Processing Systems (NeurIPS)_, 2022.

```
@inproceedings{lindner2022active,
    title={Active Exploration for Inverse Reinforcement Learning},
    author={Lindner, David and Krause, Andreas and Ramponi, Giorgia},
    booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
    year={2022},
}
```

## Setup

We recommend to use [Anaconda](https://www.anaconda.com/) for setting up an environment with the required dependencies. After installing Anaconda, you can create an environment using:
```bash
conda create -n "aceirl" python=3.9
conda activate aceirl
```

Then, install dependencies by running
```bash
pip install -e . 
```
in the root folder of this repository.


## Reproducing experiments in the main paper

All experiments can be run using the `scripts/experiments/experiments.py` script. We use [`sacred`](https://github.com/IDSIA/sacred) for keeping track of experiment parameters.

### Running active IRL experiments

Run the following commands to reproduce the experiments in the main paper:
```bash
python scripts/experiments/experiment.py with four_paths n_ep_per_iter=50 results_file="result_aceirl_four_paths_50ep_50runs.csv"
python scripts/experiments/experiment.py with four_paths n_ep_per_iter=100 results_file="result_aceirl_four_paths_100ep_50runs.csv"
python scripts/experiments/experiment.py with four_paths n_ep_per_iter=200 results_file="result_aceirl_four_paths_200ep_50runs.csv"

python scripts/experiments/experiment.py with double_chain n_ep_per_iter=50 results_file="result_aceirl_double_chain_50ep_50runs.csv"
python scripts/experiments/experiment.py with double_chain n_ep_per_iter=100 results_file="result_aceirl_double_chain_100ep_50runs.csv"
python scripts/experiments/experiment.py with double_chain n_ep_per_iter=200 results_file="result_aceirl_double_chain_200ep_50runs.csv"

python scripts/experiments/experiment.py with random_env results_file="result_aceirl_random_mdp_50runs.csv"
python scripts/experiments/experiment.py with chain results_file="result_aceirl_chain_mdp_50runs.csv"
python scripts/experiments/experiment.py with gridworld results_file="result_aceirl_gridworld_50runs.csv"
```

To parallelize experiments, you can additionally pass the `n_jobs` parameter, for example:
```bash
python scripts/experiments/experiment.py with four_paths n_ep_per_iter=200 n_jobs=50
```

### Running reward-free exploration experiments in the appendix

Run the following commands to reproduce the reward-free exploration experiments:
```bash
python scripts/experiments/experiment.py with double_chain_rfe n_ep_per_iter=1000
python scripts/experiments/experiment.py with double_chain_rfe n_ep_per_iter=3000
python scripts/experiments/experiment.py with double_chain_rfe n_ep_per_iter=5000
```

### Creating plots

The results will be saved in `results/`. To plot the results and produce the results from Table 1 in the paper, use the `scripts/experiments/plot_results.ipynb` notebook.
