import numpy as np
from tqdm import tqdm
from modules.utils import *
from modules.domkl_funs import *
import argparse
import scipy.io

parser = argparse.ArgumentParser(
    prog="drfgp",
)
parser.add_argument("-D", "--dataset", type=str)
parser.add_argument("-N", "--num_agents", type=int)
parser.add_argument("-r", "--rand_seed", default=0, type=int)
parser.add_argument("-c", "--rho", default=0, type=float)
parser.add_argument("-l", "--eta_l", default=0, type=float)
parser.add_argument("-g", "--eta_g", default=0, type=float)


args = parser.parse_args()
num_agents = args.num_agents
rho = args.rho
eta_l = args.eta_l
eta_g = args.eta_g


print(
    f"Running dataset {args.dataset} with seed {args.rand_seed}, num_agents = {num_agents} and \
      (rho, eta_l, eta_g) = ({rho},{eta_l},{eta_g})"
)


np.random.seed(args.rand_seed)


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
    alpha = 0.25
    graph = generate_connected_graph(num_agents, alpha)
    adj_mat = nx.adjacency_matrix(graph).todense()

    network = adj_mat + np.eye(num_agents)

elif num_agents == 1:
    network = np.eye(1)


# %% params
# number of spectral frequencies
M = 50

# Number of models/kernels/lengthscales
P = 3  # number of models/kernels
ells = [10**c for c in np.linspace(-1, 1, P)]
v = np.random.randn(M, dim)  # Generate the same random vectors v_i for all models
v = [v / ell for ell in ells]


# Initialization
theta_models = np.zeros(
    (num_agents, P, 2 * M, 1)
)  # Each agent has its own theta vector of length 2M
lambda_models = np.zeros((num_agents, P, 2 * M, 1))
log_ws = np.zeros((num_agents, P))  # Initialize num_agents-by-P matrix of weights
qs = [[1 / P for _ in range(P)] for _ in range(num_agents)]  # initial ensemble weights


# mse_history = []  # Track mean squared error over time
mse_t = []
qs_history = [qs.copy()]


# %% Run DOMKL
N_to_predict = 1000

for t in tqdm(range(0, X.shape[0] - N_to_predict, num_agents)):

    x_agents = [
        x.reshape(1, dim) for x in X[t : t + num_agents]
    ]  # each agent receives x.shape[0] = 1 observation at each time step
    y_agents = [ynow.reshape(-1, 1) for ynow in y[t : t + num_agents]]

    if len(x_agents) < num_agents:
        break

    mse_t_agents = []
    theta_models_new = np.zeros((num_agents, P, 2 * M, 1))
    for k in range(num_agents):
        x_kt = x_agents[k]
        y_kt = y_agents[k]

        # we first update the weights ws (because we want the losses before updating the thetas)
        log_ws[k], yhat_models = update_w(
            k, x_kt, y_kt, log_ws, theta_models, eta_g, v, M
        )

        # compute the ensemble prediction of the arriving observation (this is for the rolling MSE)
        qs_k = qs[k]
        yhat_k = np.dot(np.array(qs_k), yhat_models)
        mse_t_agents.append((yhat_k.squeeze() - y_kt.squeeze()) ** 2)

        # Update theta_models for agent k (but don't overwrite them yet!)
        for p in range(P):
            theta_models_new[k][p] = update_theta(
                k,
                x_kt,
                y_kt,
                theta_models[:, p, :],
                lambda_models[:, p, :],
                v[p],
                M,
                rho,
                eta_l,
                network,
            )

    # and we update the theta_models (communicating between neighbors)
    theta_models = theta_models_new.copy()

    for k in range(num_agents):
        # update ensemble weights for agent k
        qs[k] = update_q(k, log_ws, network)
        # Update lambda for agent k
        for p in range(P):
            lambda_models[k][p] = update_lambda(
                k, theta_models[:, p, :], lambda_models[:, p, :], rho, network
            )

    # store history of model weights
    qs_history.append(qs)

for t in tqdm(range(X.shape[0] - N_to_predict, X.shape[0], num_agents)):

    x_agents = [
        x.reshape(1, dim) for x in X[t : t + num_agents]
    ]  # each agent receives x.shape[0] = 1 observation at each time step
    y_agents = [ynow.reshape(-1, 1) for ynow in y[t : t + num_agents]]

    if len(x_agents) < num_agents:
        break

    mse_t_agents = []
    theta_models_new = np.zeros((num_agents, P, 2 * M, 1))
    for k in range(num_agents):
        x_kt = x_agents[k]
        y_kt = y_agents[k]

        yhat_models = predict(k, x_kt, log_ws, theta_models, v, M)

        # compute the ensemble prediction of the arriving observation (this is for the rolling MSE)
        qs_k = qs[k]
        yhat_k = np.dot(np.array(qs_k), yhat_models)
        mse_t_agents.append((yhat_k.squeeze() - y_kt.squeeze()) ** 2)

    # store rolling mse of agents
    mse_t.append(mse_t_agents)


mse_t = np.array(mse_t)

avg_errs = np.mean(mse_t, axis=1)

np.savez(
    f"results/DOMKL_{args.dataset}_num_agents_{num_agents}_seed_{args.rand_seed}_last_1000.npz",
    avg_errs,
)
