import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


def gradient(X, y, beta):
    return -X.T @ (y - X @ beta)


def hard_thresholding(v, k):
    idx = np.argsort(np.abs(v))[-k:]
    w = np.zeros_like(v)
    w[idx] = v[idx]
    return w


def iterative_hard_thresholding(X, y, k, L=None, epsilon=1e-4, max_iter=100):
    n, p = X.shape
    beta = np.zeros(p)
    if L is None:
        L = np.linalg.norm(X.T @ X, 2)

    for _ in range(max_iter):
        grad = gradient(X, y, beta)
        beta_new = hard_thresholding(beta - (1 / L) * grad, k)
        if np.abs(np.linalg.norm(X @ beta - y)**2 - np.linalg.norm(X @ beta_new - y)**2) < epsilon:
            break
        beta = beta_new

    return beta


def interpolated_hard_thresholding(X, y, k, L=None, epsilon=1e-4, max_iter=100):
    n, p = X.shape
    beta = np.zeros(p)
    if L is None:
        L = np.linalg.norm(X.T @ X, 2)

    best_beta = beta.copy()
    best_loss = np.inf

    for _ in range(max_iter):
        grad = gradient(X, y, beta)
        eta = hard_thresholding(beta - (1 / L) * grad, k)

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

    return beta


def run_lasso(X, y, alpha=0.1):
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=100)
    model.fit(X, y)
    return model.coef_


def refine_with_least_squares(X, y, beta_sparse):
    support = np.flatnonzero(beta_sparse)
    if len(support) == 0:
        return beta_sparse
    X_sub = X[:, support]
    beta_refined = np.linalg.pinv(X_sub.T @ X_sub) @ X_sub.T @ y
    beta_final = np.zeros_like(beta_sparse)
    beta_final[support] = beta_refined
    return beta_final


def evaluate_model(X, y, beta):
    pred = X @ beta
    mse = np.mean((y - pred)**2)
    sparsity = np.sum(beta != 0)
    return {"MSE": mse, "Non-zero coefficients": sparsity}


def plot_coefficients(betas, labels):
    for beta, label in zip(betas, labels):
        plt.plot(beta, label=label, marker='o')
    plt.xlabel('Feature index')
    plt.ylabel('Coefficient value')
    plt.title('Comparison of coefficients')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_all_methods(X, y, k):
    beta_iht = iterative_hard_thresholding(X, y, k, max_iter=100)
    beta_interp = interpolated_hard_thresholding(X, y, k, max_iter=100)

    beta_iht_refined = refine_with_least_squares(X, y, beta_iht)
    beta_interp_refined = refine_with_least_squares(X, y, beta_interp)

    beta_lasso = run_lasso(X, y, alpha=0.1)

    results = {
        "IHT (1st-order)": evaluate_model(X, y, beta_iht),
        "Interpolated (1st-order)": evaluate_model(X, y, beta_interp),
        "IHT + Newton": evaluate_model(X, y, beta_iht_refined),
        "Interpolated + Newton": evaluate_model(X, y, beta_interp_refined),
        "LASSO": evaluate_model(X, y, beta_lasso)
    }

    betas = {
        "IHT (1st-order)": beta_iht,
        "Interpolated (1st-order)": beta_interp,
        "IHT + Newton": beta_iht_refined,
        "Interpolated + Newton": beta_interp_refined,
        "LASSO": beta_lasso
    }

    return results, betas


################
# === Main === #
################

X, y = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X)
k = 5

results_dict, betas_dict = evaluate_all_methods(X, y, k)

sorted_results = sorted(results_dict.items(), key=lambda x: x[1]['MSE'])
print(sorted_results)

labels_ordered = [
    "LASSO",
    "IHT + Newton",
    "Interpolated + Newton",
    "IHT (1st-order)",
    "Interpolated (1st-order)"
]
betas_to_plot = [betas_dict[label] for label in labels_ordered]

plot_coefficients(betas_to_plot, labels_ordered)