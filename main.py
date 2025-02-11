import numpy as np
from tqdm import tqdm
from modules.drfgp import distributed_rf_gp, E_drfgp
from modules.utils import *
import argparse
import scipy.io

parser = argparse.ArgumentParser(
    prog="drfgp",
)
parser.add_argument("-D", "--dataset", type=str)
parser.add_argument("-N", "--num_agents", type=int)
parser.add_argument("-r", "--rand_seed", default=0, type=int)
parser.add_argument("-c", "--consensus", type=int)
parser.add_argument("-p", "--graph-alpha", default=0.2, type=float)


args = parser.parse_args()
num_agents = args.num_agents
graph_alpha = args.graph_alpha


print(
    f"Running dataset {args.dataset} with seed {args.rand_seed} and (num_agents,consensus) = ({num_agents},{args.consensus})"
)


np.random.seed(args.rand_seed)

# params
M = 50
s2_theta = 1.0
s2_obs = 0.01


if args.dataset == "tom":
    TomData = scipy.io.loadmat("TomData.mat")
    X = TomData["X"].T
    y = TomData["y"]
elif args.dataset == "har":
    HarData = scipy.io.loadmat("HarData.mat")
    X = HarData["X"].T
    y = HarData["y"]
elif args.dataset == "ene":
    EnergyData = scipy.io.loadmat("EnergyData.mat")
    X = EnergyData["X"].T
    y = EnergyData["y"]
elif args.dataset == "twi":
    TwitterData = scipy.io.loadmat("TwitterData.mat")
    X = TwitterData["X"].T
    y = TwitterData["y"]

# %% creating the network

dim = X.shape[1]
if num_agents > 1:
    alpha = args.graph_alpha
    graph = generate_connected_graph(num_agents, alpha)
    adj_mat = nx.adjacency_matrix(graph).todense()

    neighbors = adj_mat + np.eye(num_agents)

elif num_agents == 1:
    neighbors = np.eye(1)


# %% creating the models
ells = np.array(
    [
        1 * np.ones((dim,)),
        0.1 * np.ones((dim,)),
        10 * np.ones((dim,)),
    ]
)

# create list of instances of rfgp
models = [
    distributed_rf_gp(
        num_agents,
        neighbors,
        M,
        dim,
        s2_theta,
        s2_obs,
        ell_now,
        initial_data=None,
        consensus_iters=args.consensus,
    )
    for ell_now in ells
]

edrfgp = E_drfgp(models)


# %% Running the simulation

er_hat = []

# Simulate for T time steps
z = 0
for t in tqdm(range(0, X.shape[0], num_agents)):
    agent_predictions = []

    x_agents = [x.reshape(1, dim) for x in X[t : t + num_agents]]
    y_agents = [ynow.reshape(-1, 1) for ynow in y[t : t + num_agents]]

    if len(x_agents) < num_agents:
        break

    # Compute the mean squared error between agents and true function
    #     er_hat.append(
    #             [(agent_pred.squeeze() - np.array(y_agents).squeeze()) ** 2 for agent_pred in agent_predictions]  # list with num_agents items where each item is of length len(y_agents)
    #             )

    # each agent only predicts at its arriving observation
    if z % 5 == 0:
        for k in range(num_agents):
            # y_mean = edrfgp.ensemble_prediction_at_agent(k,np.array(x_agents).squeeze().reshape(-1,dim))
            y_mean = edrfgp.ensemble_prediction_at_agent(
                k, x_agents[0]
            )  # each agent ONLY predicts at the first observation in the next batch

            agent_predictions.append(y_mean)

        er_hat.append(
            [
                (agent_pred.squeeze() - y_agents[0].squeeze()) ** 2
                for i, agent_pred in enumerate(agent_predictions)
            ]  # list with num_agents items where each item is of length len(y_agents)
        )

    z += num_agents

    # update agents' posterior
    edrfgp.step_models(x_agents, y_agents)


# %%
# er_hat = np.array(er_hat).reshape(-1, num_agents**2)
# er_hat[0] = np.ones(num_agents**2)

er_hat = np.array(er_hat).reshape(
    -1, num_agents
)  # when agents predict only at their arriving observations
er_hat[0] = np.ones(er_hat.shape[1])

running_errs = np.cumsum(np.mean(er_hat, axis=1)) / np.arange(1, len(er_hat) + 1)


# %%
np.savez(
    f"results/{args.dataset}_num_agents_{num_agents}_seed_{args.rand_seed}_L_{args.consensus}_graph_alpha_{args.graph_alpha}_every_five.npz",
    running_errs,
)
