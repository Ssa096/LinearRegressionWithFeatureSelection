import numpy as np

def gradient(X, y, beta):
    return -X.T @ (y - X @ beta)

def hard_thresholding(v, k):
    idx = np.argsort(np.abs(v))[-k:]
    w = np.zeros_like(v)
    w[idx] = v[idx]
    return w


def iterative_hard_thresholding(X, y, k, L=None, epsilon=1e-4, max_iter=1000):
    """_summary_
    first algortihms
    Args:
        X (_type_): _description_
        y (_type_): _description_
        k (_type_): _description_
        L (_type_, optional): _description_. Defaults to None.
        epsilon (_type_, optional): _description_. Defaults to 1e-4.
        max_iter (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    
    n, p = X.shape
    beta = np.zeros(p)
    
    if L is None:
        L = np.linalg.norm(X.T @ X, 2) 

    for _ in range(max_iter):
        grad = gradient(X, y, beta)
        beta_new = hard_thresholding(beta - (1/L) * grad, k)
        if np.abs(np.linalg.norm(X @ beta - y)**2 - np.linalg.norm(X @ beta_new - y)**2) < epsilon:
            break
        beta = beta_new

    return beta

def interpolated_hard_thresholding(X, y, k, L=None, epsilon=1e-4, max_iter=1000):
    """_summary_
    second algortihms
    Args:
        X (_type_): _description_
        y (_type_): _description_
        k (_type_): _description_
        L (_type_, optional): _description_. Defaults to None.
        epsilon (_type_, optional): _description_. Defaults to 1e-4.
        max_iter (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    
    n, p = X.shape
    beta = np.zeros(p)
    
    if L is None:
        L = np.linalg.norm(X.T @ X, 2)

    best_beta = beta.copy()
    best_loss = np.inf

    for _ in range(max_iter):
        grad = gradient(X, y, beta)
        eta = hard_thresholding(beta - (1/L) * grad, k)

        lambdas = np.linspace(0, 1, 20)
        losses = [np.linalg.norm(y - X @ (l * eta + (1 - l) * beta))**2 / 2 for l in lambdas]
        lambda_star = lambdas[np.argmin(losses)]

        beta_new = lambda_star * eta + (1 - lambda_star) * beta

        if np.abs(np.linalg.norm(X @ beta_new - y)**2 - np.linalg.norm(X @ beta - y)**2) < epsilon:
            break

        if losses[np.argmin(losses)] < best_loss:
            best_beta = beta_new.copy()
            best_loss = losses[np.argmin(losses)]

        beta = beta_new

    return best_beta
