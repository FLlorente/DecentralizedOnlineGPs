import numpy as np
import networkx as nx


def generate_connected_graph(N, alpha):
    # Generate a random graph with edge probability alpha
    G = nx.erdos_renyi_graph(N, alpha)

    # Ensure the graph is connected
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(N, alpha)

    return G


# https://github.com/kirthevasank/gp-parallel-ts/blob/master/utils/general_utils.py
# MIT License (c) Kirthevasan Kandasamy
def stable_cholesky(M):
    """Returns L, a 'stable' cholesky decomposition of M. L is lower triangular and
    satisfies L*L' = M.
    Sometimes nominally psd matrices are not psd due to numerical issues. By adding a
    small value to the diagonal we can make it psd. This is what this function does.
    Use this iff you know that K should be psd. We do not check for errors
    """
    # pylint: disable=superfluous-parens
    if M.size == 0:
        return M  # if you pass an empty array then just return it.
    try:
        # First try taking the Cholesky decomposition.
        L = np.linalg.cholesky(M)
    except np.linalg.linalg.LinAlgError:
        # If it doesn't work, then try adding diagonal noise.
        diag_noise_power = -11
        max_M = np.diag(M).max()
        diag_noise = np.diag(M).max() * 1e-11
        chol_decomp_succ = False
        while not chol_decomp_succ:
            try:
                L = np.linalg.cholesky(
                    M + (10**diag_noise_power * max_M) * np.eye(M.shape[0])
                )
                chol_decomp_succ = True
            except np.linalg.linalg.LinAlgError:
                diag_noise_power += 1
        if diag_noise_power >= 5:
            print(
                "**************** Cholesky failed: Added diag noise = %e" % (diag_noise)
            )
    return L


def draw_gaussian_samples(num_samples, mu, K):
    """Draws num_samples samples from a Gaussian distribution with mean mu and
    covariance K.
    """
    num_pts = len(mu)
    L = stable_cholesky(K)
    U = np.random.normal(size=(num_pts, num_samples))
    V = L.dot(U).T + mu
    # print((L.dot(U).T).shape)
    return V
