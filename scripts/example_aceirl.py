import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from active_irl.aceirl import ActiveIRL, MDP
from active_irl.envs import (
    random_mdp,
    chain_mdp,
    four_paths_mdp,
    double_chain_mdp,
    Gridworld,
)

sns.set_theme()

np.random.seed(2)

# States and Actions
horizon = 10
n_states, n_actions = 5, 15

transitions, reward, init_state_dist = random_mdp(n_states, n_actions, horizon)
# transitions, reward, init_state_dist = chain_mdp(
#     n_states, n_actions, horizon, p_fail=0.01
# )
# transitions, reward, init_state_dist = four_paths_mdp(n_each_dir=3, horizon=5)
# transitions, reward, init_state_dist = double_chain_mdp()


# Create a generative model
mdp = MDP(transitions, reward, init_state_dist)

# Pass the generative model to ActiveIRL algorithm and run it
model = ActiveIRL(
    delta=0.1,
    epsilon_stop=0.01,
    mdp=mdp,
    run_irl=True,
)
model.run(
    n_max=10000,
    n_ep_per_iter=1,
    verbose=True,
    method="aceirl",
    # method="random_exploration",
    # method="uniform_sampling",
    # method="travel",
    use_policyci=False,
    use_eps_const=False,
    record_loss=True,
)

# Visualize the final sampling strategy
ax = sns.heatmap(model.sample_count.sum(axis=0))
ax.set_title("Sampling strategy")
ax.set_xlabel("Actions")
ax.set_ylabel("States")
plt.show()
