import numpy as np
from math import prod
from scipy.special import logsumexp


# Random Feature Map (Eq. 5) for 1D x
def random_feature_map(x, v, M):
    """
    Compute the random feature map of input data using sine and cosine transformations.

    Parameters:
        x (np.ndarray): A 2D numpy array with shape (N, dim) representing N data points.
        v (np.ndarray): A numpy array of shape (M, dim) containing random projection vectors.
        M (int): The number of random features (half the total features before stacking).

    Returns:
        np.ndarray: The transformed feature matrix of shape (2*M, N), computed as:
                    [sin(v @ x.T); cos(v @ x.T)] / sqrt(M)
    """
    assert x.ndim == 2

    sin_features = np.sin(v @ x.T)  # (M,dim) x (dim,N) = (M,N)
    cos_features = np.cos(v @ x.T)

    assert sin_features.shape == cos_features.shape == (M, x.shape[0])

    phi = np.vstack([sin_features, cos_features]) / np.sqrt(M)
    assert phi.shape == (2 * M, x.shape[0])

    return phi  # (2M,N)


# Update theta (Closed-form Eq. 11)
def update_theta(k, x_kt, y_kt, theta, lambda_, v, M, rho, eta_l, network):
    """
    Update the local model parameter theta for agent k using a closed-form solution.

    This function calculates a new theta for agent k based on the random feature mapping of the current
    observation, a consensus term from neighboring agents, and regularization parameters.

    Parameters:
        k (int): Index of the current agent.
        x_kt (np.ndarray): Input data for agent k at time t (expected shape compatible with feature mapping).
        y_kt (float or np.ndarray): Target label corresponding to the input data.
        theta (list or dict): Collection of current theta values for all agents. theta[k] must be of shape (2*M, 1).
        lambda_ (list or dict): Current Lagrange multipliers for the agents, with lambda_[k] of shape (2*M, 1).
        v (np.ndarray): Random projection vector/matrix used for the feature mapping.
        M (int): Number of random features (before doubling via sine and cosine).
        rho (float): ADMM parameter that controls the consensus regularization strength.
        eta_l (float): Local regularization parameter.
        network (list): Connectivity information, where network[k] is an iterable indicating the presence (or weight)
                        of an edge between agent k and its neighbors.

    Returns:
        np.ndarray: Updated theta for agent k with shape (2*M, 1).
    """
    z_p = random_feature_map(x_kt, v, M)
    assert z_p.shape == (2 * M, 1)

    # Compute consensus term gamma_k from neighboring agents
    gamma_k = 0.5 * (
        sum([(theta[k] + theta[i]) * edge for i, edge in enumerate(network[k])])
        - 2 * theta[k]
    )

    assert gamma_k.shape == theta[k].shape == (2 * M, 1)

    term1 = 2 * z_p @ z_p.T + (eta_l + rho * (np.sum(network[k]) - 1)) * np.eye(2 * M)
    term2 = 2 * y_kt * z_p + eta_l * theta[k] + rho * gamma_k - lambda_[k]

    theta_k_new = np.linalg.solve(term1, term2)

    assert theta_k_new.shape == (2 * M, 1)

    return theta_k_new


# Update lambda (Eq. 12)
def update_lambda(k, theta, lambda_, rho, network):
    """
    Update the Lagrange multiplier (lambda) for agent k in the ADMM framework.

    This update uses the weighted differences between the local theta and its neighbors, scaled by the parameter rho.

    Parameters:
        k (int): Index of the current agent.
        theta (list or dict): Current model parameters for all agents.
        lambda_ (list or dict): Current lambda values, where lambda_[k] corresponds to agent k.
        rho (float): ADMM update parameter.
        network (list): Connectivity information for agents; network[k] is iterable over edge weights for neighbors.

    Returns:
        np.ndarray: The updated lambda for agent k.
    """
    sum_diff = sum([(theta[k] - theta[i]) * edge for i, edge in enumerate(network[k])])
    lambda_k_new = lambda_[k] + (rho / 2) * sum_diff
    return lambda_k_new


def predict(k, x_kt, log_ws, theta_models, v, M):
    """
    Make predictions for agent k using the current model parameters.

    For each model (kernel) p, compute the predicted output as the inner product between the random feature
    transformation of the input and the corresponding model parameters.

    Parameters:
        k (int): Index of the current agent.
        x_kt (np.ndarray): Input data for agent k at time t.
        log_ws (np.ndarray): A 2D array representing log weights with shape (num_agents, P),
                             where P is the number of models/kernels.
        theta_models (list or dict): Model parameters for each agent; theta_models[k][p] has shape (2*M, 1)
                                     for each model p.
        v (list or np.ndarray): Collection of random projection matrices/vectors for models; v[p] is used for model p.
        M (int): Number of random features (before doubling).

    Returns:
        np.ndarray: Array of predictions for each model with shape (P, 1).
    """
    P = log_ws.shape[1]
    yhat = np.zeros((P, 1))
    for p in range(P):
        yhat[p] = random_feature_map(x_kt, v[p], M).T @ theta_models[k][p]

    return yhat


def update_w(k, x_kt, y_kt, log_ws, theta_models, eta_g, v, M):
    """
    Update the log weights of the models for agent k based on the prediction errors.

    For each model, the prediction error (squared difference) is used to adjust the corresponding log weight.
    This update occurs before the model parameters (theta) are updated.

    Parameters:
        k (int): Index of the current agent.
        x_kt (np.ndarray): Input data for agent k at time t.
        y_kt (float or np.ndarray): True target value for the current observation.
        log_ws (np.ndarray): A 2D array of log weights, where log_ws[k][p] is the log weight for model p.
        theta_models (list or dict): Model parameters for each agent; theta_models[k][p] is the parameter for model p.
        eta_g (float): Learning rate for the weight update.
        v (list or np.ndarray): Collection of random projection matrices/vectors used for models; v[p] is used for model p.
        M (int): Number of random features.

    Returns:
        tuple:
            np.ndarray: Updated log weights for agent k with shape (P,).
            np.ndarray: Predictions (yhat) for each model with shape (P, 1).
    """
    P = log_ws.shape[1]
    yhat = np.zeros((P, 1))
    for p in range(P):
        yhat[p] = random_feature_map(x_kt, v[p], M).T @ theta_models[k][p]
        # Use the loss of kernel p for agent k BEFORE updating theta[k][p]
        log_ws[k][p] = log_ws[k][p] + (
            -1.0 / eta_g * (yhat[p].squeeze() - y_kt.squeeze()) ** 2
        )

    return log_ws[k], yhat


def update_q(k, log_ws, network):
    """
    Update the consensus weights (q) for agent k based on aggregated log weights from its neighbors.

    The function sums the log weights for each model over the neighboring agents (where an edge exists)
    and then normalizes these log weights using the softmax function (via logsumexp) to obtain probabilities.

    Parameters:
        k (int): Index of the current agent.
        log_ws (np.ndarray): A 2D array of log weights with shape (num_agents, P), where P is the number of models.
        network (list): Connectivity information for agents; network[k] is an iterable with edge weights (typically 0 or 1).

    Returns:
        np.ndarray: Normalized consensus weights for each model for agent k, with shape (P,).
    """
    P = log_ws.shape[1]
    log_qs_k_new = np.zeros(P)
    for p in range(P):
        # Filter only neighbors (including agent k itself) with a valid connection
        neighbors_weights = [
            log_ws[i][p] for i, edge in enumerate(network[k]) if edge == 1
        ]
        log_qs_k_new[p] = sum(neighbors_weights)

    # Normalize using logsumexp for numerical stability
    total = logsumexp(log_qs_k_new)
    qs_k_new = np.exp(log_qs_k_new - total)
    return qs_k_new
