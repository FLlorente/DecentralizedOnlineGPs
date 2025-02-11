"""
Distributed Random Feature Gaussian Process (DRFGP) Implementation

This module implements a distributed version of Random Feature Gaussian Process regression,
allowing multiple agents to collaboratively learn and make predictions in a network setting.
The implementation uses random Fourier features to approximate the GP kernel, enabling
scalable distributed learning.

Key Classes:
    - distributed_rf_gp: Main class implementing the distributed GP algorithm
    - E_drfgp: Ensemble class that manages multiple distributed_rf_gp models

The implementation supports:
    - Distributed learning across a network of agents
    - Thompson sampling for exploration
    - Bayesian Model Averaging (BMA) for ensemble predictions
    - Efficient posterior updates using random feature approximations

Author: Fernando Llorente & Daniel Waxman
Date: 2025-02-11
"""

from modules.utils import *
import numpy as np
import scipy as sp
from scipy.special import logsumexp


class distributed_rf_gp:
    """
    Distributed Random Feature Gaussian Process regression model.

    This class implements a distributed GP regression model where multiple agents
    collaborate to learn from data while maintaining their local approximations
    of the centralized posterior. The implementation uses random Fourier features
    for kernel approximation and supports consensus-based learning.

    Attributes:
        num_agents (int): Number of agents in the network
        M (int): Number of random features used for kernel approximation
        s2_theta (float): Prior variance for the GP parameters
        s2_obs (float): Observation noise variance
        ell (float): Length scale parameter for the RBF kernel
        network (list): Adjacency matrix representing the network connectivity
        dim (int): Input dimension
        consensus_iters (int): Number of consensus iterations
        logw_bma (list): Log weights for Bayesian Model Averaging
        v (np.ndarray): Random frequencies for feature mapping
        best_f_agents (list): Best function values found by each agent
        best_x_agents (list): Best inputs found by each agent
        P_agents (list): Precision matrices for each agent
        s_agents (list): Sufficient statistics for each agent
        D_agents (list): Updated precision matrices after consensus
        eta_agents (list): Updated statistics after consensus
    """

    def __init__(
        self,
        num_agents,
        network,
        num_freqs,
        dim,
        prior_theta_var,
        prior_obs_var,
        ell,
        initial_data=None,
        consensus_iters=1,
    ):
        self.num_agents = num_agents
        self.M = num_freqs
        self.s2_theta = prior_theta_var
        self.s2_obs = prior_obs_var
        self.ell = ell
        self.network = network
        self.dim = dim
        self.consensus_iters = consensus_iters
        self.logw_bma = [0 for _ in range(self.num_agents)]
        self.v = np.random.randn(self.M, self.dim) / self.ell  # random freqs

        # Initialize best-so-far x and y of each agent (assuming noiseless observations)
        self.best_f_agents = [-np.inf] * self.num_agents
        self.best_x_agents = [None] * self.num_agents

        # Initialize P_agents and s_agents (they are used to update the approximation of the centralized posterior)
        self.P_agents = []
        self.s_agents = []
        for _ in range(self.num_agents):
            self.s_agents.append(
                np.zeros((2 * self.M, 1))
            )  # agents' prior precision x prior mean
            self.P_agents.append(
                1 / (self.num_agents * self.s2_theta) * np.eye(2 * self.M)
            )  # agents' prior precision
        self.P_consensus = self._consensus(self.P_agents)
        self.s_consensus = self._consensus(self.s_agents)

        # Initialize D_agents and eta_agents
        self.D_agents = self.P_consensus.copy()
        self.eta_agents = self.s_consensus.copy()

        if initial_data is not None:
            self.step_agents(
                [sublist[0] for sublist in initial_data],
                [sublist[1] for sublist in initial_data],
            )

    def _update_agents_best(
        self, x_agents, y_agents
    ):  # when evaluations of the objective are noiseless
        for k in range(self.num_agents):
            for x, y in zip(x_agents[k], y_agents[k]):
                if y > self.best_f_agents[k]:
                    self.best_f_agents[k] = y
                    self.best_x_agents[k] = x

    def _consensus_agent(self, k, local_quantities):
        updated_k = sum(
            [local_quantities[i] * edge for i, edge in enumerate(self.network[k])]
        ) / np.sum(self.network[k])

        return updated_k

    def _consensus(self, local_quantities):

        assert self.consensus_iters >= 1

        for _ in range(self.consensus_iters):
            updated_quantities = [None] * self.num_agents
            for k in range(self.num_agents):
                updated_quantities[k] = self._consensus_agent(k, local_quantities)
            local_quantities = updated_quantities.copy()

        return [lq * self.num_agents for lq in local_quantities]

    def _compute_P(self, x):
        phi = self._random_feature_map(x)
        assert phi.shape == (2 * self.M, x.shape[0])

        return phi @ phi.T / self.s2_obs

    def _compute_s(self, x, y):
        phi = self._random_feature_map(x)
        assert phi.shape[1] == y.shape[0]

        return (np.dot(phi, y)).reshape(2 * self.M, 1) / self.s2_obs

    def _random_feature_map(self, x):

        assert x.ndim == 2
        assert x.shape[1] == self.dim

        sin_features = np.sin(self.v @ x.T)  # (M,dim) x (dim,N) = (M,N)
        cos_features = np.cos(self.v @ x.T)

        assert sin_features.shape == cos_features.shape == (self.M, x.shape[0])

        phi = np.vstack([sin_features, cos_features]) / np.sqrt(self.M)
        assert phi.shape == (2 * self.M, x.shape[0])

        return phi  # (2M,N)

    def step_agents(self, x_agents, y_agents):
        """
        Update all agents with new observations.

        This method:
        1. Updates BMA weights
        2. Computes local quantities
        3. Performs consensus
        4. Updates posterior approximations

        Args:
            x_agents (list): New input points for each agent
            y_agents (list): New observations for each agent
        """
        logw_bma_now = [None] * self.num_agents
        # each agent receives a new observation and computes the local quantities P and s
        for k in range(self.num_agents):
            x_kt = x_agents[k]
            y_kt = y_agents[k]

            # updated agent's bma weights
            mu, cov = self.predict_with_agent(k, x_kt, diag=False)
            logw_bma_now[k] = sp.stats.multivariate_normal.logpdf(
                y_kt.squeeze(), mu.squeeze(), cov
            )

            # Compute P and s for each agent
            self.P_agents[k] = self._compute_P(x_kt)
            self.s_agents[k] = self._compute_s(x_kt, y_kt)

        # optional consensus on agents' bma weights before updating
        logw_bma_consensus = self._consensus(logw_bma_now)
        self.logw_bma = [
            self.logw_bma[k] + logw_bma_consensus[k] for k in range(self.num_agents)
        ]

        # the agents reach consensus on P_bar and s_bar
        self.P_consensus = self._consensus(self.P_agents)
        self.P_consensus = [(P.T + P) / 2 for P in self.P_consensus]
        self.s_consensus = self._consensus(self.s_agents)

        # the agents update their current approx to the centralized posterior
        self.D_agents = [
            self.D_agents[k] + self.P_consensus[k] for k in range(self.num_agents)
        ]
        self.eta_agents = [
            self.eta_agents[k] + self.s_consensus[k] for k in range(self.num_agents)
        ]

        # and update agents' approximations of the maximum value and point
        self._update_agents_best(x_agents, y_agents)

    def predict_with_agent(self, k, x_new, diag=True, predict_y=True):
        """
        Make predictions using agent k's model.

        Args:
            k (int): Agent index
            x_new (np.ndarray): Test points
            diag (bool): If True, return diagonal of covariance matrix
            predict_y (bool): If True, include observation noise

        Returns:
            tuple: (mean predictions, variances)
        """
        D = self.D_agents[k]
        eta = self.eta_agents[k]

        phi_new = self._random_feature_map(x_new)
        assert phi_new.shape == (2 * self.M, x_new.shape[0])

        L = stable_cholesky(D)
        Linveta = np.linalg.solve(L, eta)
        LinvPhi = np.linalg.solve(L, phi_new)

        fpred = LinvPhi.T @ Linveta
        fcovpred = LinvPhi.T @ LinvPhi

        if predict_y:
            fcovpred += self.s2_obs * np.eye(fcovpred.shape[0])

        if diag:
            return fpred.squeeze(), np.diag(fcovpred)
        else:
            return fpred.squeeze(), fcovpred

    def _sample_from_agent(self, k, x_grid, num_samples=1, fast=False):
        """
        Draw samples from agent k's posterior distribution.

        Args:
            k (int): Agent index
            x_grid (np.ndarray): Points to sample at
            num_samples (int): Number of samples to draw
            fast (bool): If True, use faster sampling method

        Returns:
            np.ndarray: Samples from the posterior
        """
        if fast:
            # sampling from the posterior of theta
            D = self.D_agents[k]
            eta = self.eta_agents[k]

            L = stable_cholesky(D)

            # Step 1: Solve for the mean mu = D^-1 @ eta by solving L^T x = eta
            mu = sp.linalg.solve_triangular(
                L.T, sp.linalg.solve_triangular(L, eta, lower=True), lower=False
            )

            # Step 2: Generate standard normal samples z ~ N(0, I)
            z = np.random.normal(size=(L.shape[0], num_samples))

            # Step 3: Solve L^T x = z to get samples with covariance D^-1
            theta_sample = sp.linalg.solve_triangular(L.T, z, lower=False)

            # Step 4: Add the mean to the samples
            theta_sample += mu.reshape(-1, 1)

            phi_new = self._random_feature_map(x_grid)
            assert phi_new.shape == (2 * self.M, x_grid.shape[0])

            samples = phi_new.T @ theta_sample
            assert samples.shape == (x_grid.shape[0], 1)

        else:
            # sample from the predictive posterior
            ymu, ycov = self.predict_with_agent(k, x_grid, False)
            samples = draw_gaussian_samples(num_samples, ymu, ycov)

        return samples

    def _decide_next_x_agent(self, k, x_grid, fast):
        """
        Select next point for agent k using Thompson sampling.

        Args:
            k (int): Agent index
            x_grid (np.ndarray): Candidate points
            fast (bool): If True, use faster sampling method

        Returns:
            np.ndarray: Selected point
        """
        ysamples = self._sample_from_agent(k, x_grid, fast=fast)
        ysamples = ysamples.squeeze()

        imax = np.argmax(ysamples)

        return x_grid[imax].reshape(1, self.dim)

    def thompson_sampling(self, x_grid, fast):
        """
        Perform Thompson sampling for all agents.

        Args:
            x_grid (np.ndarray): Candidate points
            fast (bool): If True, use faster sampling method

        Returns:
            list: Selected points for each agent
        """
        x_batch = []
        for k in range(self.num_agents):
            x_batch.append(self._decide_next_x_agent(k, x_grid, fast=fast))

        return x_batch


class E_drfgp:
    """
    Ensemble of Distributed Random Feature Gaussian Process models.

    This class manages multiple distributed_rf_gp models and combines their
    predictions using Bayesian Model Averaging (BMA).

    Attributes:
        models (list): List of distributed_rf_gp instances
        num_models (int): Number of models in the ensemble
    """

    def __init__(self, models):
        """
        Pass the list of instances of class distributed_rf_gp
        """
        self.models = models  # models contains a list of drfgp models
        self.num_models = len(models)

    def step_models(self, x_agents, y_agents):
        """
        Updates agents' approx to the centralized posterior for each model; it also updates the BMA weights.
        To check the BMA weight at each of the agents for a particular model do self.models[m].logw_bma
        """
        for m in range(self.num_models):
            self.models[m].step_agents(x_agents, y_agents)

    def get_bma_weights_at_agent(self, k):  # unnormalized weights in log-scale
        logw_bma_models = [None] * self.num_models
        for m in range(self.num_models):
            logw_bma_models[m] = self.models[m].logw_bma[
                k
            ]  # collect m-th model agent's bma weight

        logw_bma_models = np.array(logw_bma_models)

        return logw_bma_models

    def ensemble_prediction_at_agent(self, k, x_test):
        """
        Predicts at x_test using the ensemble of models at agent k
        """
        ymu_models = [None] * self.num_models
        for m in range(self.num_models):
            ymu, _ = self.models[m].predict_with_agent(k, x_test)
            # collect m-th model agent's mean prediction
            ymu_models[m] = ymu

        # collect bma weights at agent k
        logw_bma_models = self.get_bma_weights_at_agent(k)

        # normalize the weights before BMA
        logw_bma_models_normalized = logw_bma_models - logsumexp(logw_bma_models)

        # fuse predictions
        ymu_models = np.array(ymu_models)

        # print(ymu_models.shape, logw_bma_models.shape)
        ymu_ensemble = ymu_models.T @ np.exp(logw_bma_models_normalized)

        return ymu_ensemble
