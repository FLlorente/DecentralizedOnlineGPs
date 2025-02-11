import numpy as np
from tqdm import tqdm
import gpytorch
import argparse
import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser(
    prog="full_gp",
)
parser.add_argument("-D", "--dataset", type=str)
parser.add_argument("-r", "--rand_seed", default=0, type=int)


args = parser.parse_args()

print(f"Running dataset {args.dataset} with seed {args.rand_seed}")


np.random.seed(args.rand_seed)
torch.random.manual_seed(args.rand_seed)


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

X = torch.from_numpy(X)
y = torch.from_numpy(y).squeeze()

print(X.shape, y.shape)

mse_t = []

# %% Run full GP
N_to_predict = 1000

X_train = X[:-N_to_predict]
y_train = y[:-N_to_predict]

X_train = X_train.cuda()
y_train = y_train.cuda()


# Define the GP model
class SVIGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        # Variational strategy for inducing points
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=X.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model

n_inducing = 100

likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
X_mean = X_train.mean(dim=0)  # Mean for each dimension (1 x d)
X_std = X_train.std(dim=0)  # Standard deviation for each dimension (1 x d)

# Generate N_inducing points with the same mean and variance
inducing_points = torch.randn(n_inducing, X.size(1)).cuda() * X_std + X_mean
model = model = SVIGPModel(inducing_points=inducing_points).double()


model = model.cuda()
likelihood = likelihood.cuda()


# Find optimal model hyperparameters
model.train()
likelihood.train()

variational_elbo = gpytorch.mlls.VariationalELBO(
    likelihood, model, num_data=y_train.size(0)
)
optimizer = torch.optim.Adam(
    [{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=1e-3
)

# Prepare data for mini-batch training
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 100
model.train()
likelihood.train()

a = tqdm(range(num_epochs))
for epoch in a:
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -variational_elbo(output, y_batch)
        loss.backward()
        optimizer.step()
    a.set_description(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
likelihood.eval()


X_test = X[-N_to_predict:].cuda()
y_test = y[-N_to_predict:].cuda()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad():
    observed_pred = likelihood(model(X_test))
    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()

mean = mean.cpu()
lower = lower.cpu()
upper = upper.cpu()

avg_errs = (mean - y_test.cpu()) ** 2

np.savez(
    f"results/FullGP_{args.dataset}_seed_{args.rand_seed}_predict_last_1000.npz",
    avg_errs,
)
