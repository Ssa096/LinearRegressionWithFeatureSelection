import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from src.models import FOLeastSquares, FOLAD
import openml
np.random.seed(42)


def generate_data(n, p, k, method='1', rho=0.5):
    error = np.random.normal(size=n)
    if method == '1':
        beta = np.zeros(p)
        beta[np.round(np.linspace(0, p - 1, k)).astype(int)] = 1
        cov = np.repeat(rho, p ** 2).reshape(p, p)
        for row_idx in range(p):
            for col_idx in range(p):
                cov[row_idx, col_idx] = cov[row_idx, col_idx] ** np.abs(row_idx - col_idx)
    elif method == '2':
        cov = np.eye(p)
        beta = np.array([1] * min(p, 5) + [0] * max(0, (p - 5)))
    elif method == '3':
        cov = np.eye(p)
        beta = np.array([0.5 + 0.95 * i for i in range(min(p, 10))] + [0] * max(0, (p - 10)))
    elif method == '4':
        cov = np.eye(p)
        beta = np.array([-10 + i * 4 for i in range(min(p, 6))] + [0] * max(0, (p - 6)))
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)
    y = X @ beta + error
    return X, y

def preprocess_diabetes_data():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    for i in range(10):
        if i != 1:
            X[f"{X.columns[i]}_pow"] = X[f"{X.columns[i]}"] ** 2
        for j in range(i + 1, 10):
            X[f"{X.columns[i]}_{X.columns[j]}"] = X[f"{X.columns[i]}"] * X[f"{X.columns[j]}"]
    selected = np.arange(len(X))
    np.random.shuffle(selected)
    X = X.iloc[selected[:350], :]
    y = y.iloc[selected[:350]]
    X, y = X.to_numpy(), y.to_numpy()
    X -= np.mean(X, axis=1).reshape(-1, 1)
    X /= np.linalg.norm(X, axis=1).reshape(-1, 1)
    return X, y

def preprocess_leukemia_data():
    dataset = openml.datasets.get_dataset(1104)
    X, _, _, _ = dataset.get_data()
    y = X['CLASS']
    X = X.drop('CLASS', axis=1)
    y = y.map({'ALL':1, 'AML':0})
    X, y = X.to_numpy(), y.to_numpy()
    X -= np.mean(X, axis=1).reshape(-1, 1)
    X /= np.linalg.norm(X, axis=1).reshape(-1, 1)
    corr = np.array([np.abs(np.corrcoef(X[:, i], y)[1][0]) for i in range(X.shape[1])])
    cols = np.argsort(corr)[-1000:]
    X = X[:, cols]
    beta = np.array(995 * [0] + 5 * [1])
    variance = np.var(X @ beta) / 7
    error = np.random.normal(scale=np.sqrt(variance), size=len(y))
    y = X @ beta + error
    return X, y

def run_lasso(X, y, alpha=0.1):
    model = Lasso(alpha=alpha, fit_intercept=False)
    model.fit(X, y)
    return model.coef_


def evaluate_model(X, y, beta):
    pred = X @ beta
    mse = np.mean((y - pred) ** 2)
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
    model = FOLeastSquares(k=k, fit_intercept=False).fit(X, y)
    beta_iht = model.beta_
    model = FOLeastSquares(k=k, fit_intercept=False, method='interpolated').fit(X, y)
    beta_interp = model.beta_

    model = FOLeastSquares(k=k, fit_intercept=False, refine_least_squares=True).fit(X, y)
    beta_iht_refined = model.beta_
    model = FOLeastSquares(k=k, fit_intercept=False, method='interpolated', refine_least_squares=True).fit(X, y)
    beta_interp_refined = model.beta_

    beta_lasso = run_lasso(X, y)

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


def main():
    X, y = load_diabetes(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    k = 5

    # results_dict, betas_dict = evaluate_all_methods(X, y, k)
    #
    # sorted_results = sorted(results_dict.items(), key=lambda x: x[1]['MSE'])
    # print(sorted_results)
    #
    # labels_ordered = [
    #     "LASSO",
    #     "IHT + Newton",
    #     "Interpolated + Newton",
    #     "IHT (1st-order)",
    #     "Interpolated (1st-order)"
    # ]
    # betas_to_plot = [betas_dict[label] for label in labels_ordered]
    # plot_coefficients(betas_to_plot, labels_ordered)

    # model = FOLAD(k=k, fit_intercept=False).fit(X, y)

    # generate_data(1000, 10, 5, method='4')

    # preprocess_diabetes_data()

    preprocess_leukemia_data()


if __name__ == "__main__":
    main()
