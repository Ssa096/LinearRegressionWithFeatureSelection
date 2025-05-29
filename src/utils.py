"""
Data generation and error calculation.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import openml
import seaborn as sns
from sklearn.datasets import load_diabetes

from src.models import FOLeastSquares

np.random.seed(42)


def calculate_error(y_true, y_pred):
    """
    Calculation of prediction error, based on a paper.
    :param y_true: Vector of true values.
    :param y_pred: Vector of predicted values.
    :return: Prediction error - squared norm of difference between true and predicted values divided by squared norm of
    true values.
    """
    return (np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)) ** 2


def generate_data(n, p, k=5, method='1', rho=0.5, num=10, snr=None):
    """
    Generate data for regression.
    :param n: Number of samples.
    :param p: Number of features.
    :param k: Number of relevant features.
    :param method: Method of generating data. Possible values: '1', '2', '3', '4'. Methods differ in beta generation
    and covariance matrix. Descriptions of these methods are in the paper and README.
    :param rho: Covariance between features. Only used for method '1'.
    :param num: Number of different sets of target variable, used for some experiments.
    :param snr: Signal-to-noise ratio. If None, then variance of generated data is 1.
    :return: Explanatory variables, target variable, beta, indices of training samples, indices of test samples.
    """
    y = []
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
    else:
        print("Wrong method!")
        return None
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)
    sigma2 = 1 if snr is None else np.var(X @ beta) / snr
    for _ in range(num):
        error = np.random.normal(scale=sigma2, size=n)
        y.append(X @ beta + error)
    if num == 1:
        y = y[0]
    a = np.arange(n)
    np.random.shuffle(a)
    train, test = a[:int(n * 0.8)], a[int(n * 0.8):]
    return X, y, beta, train, test


def preprocess_diabetes_data():
    """
    Preprocess diabetes dataset.
    :return: Preprocessed data.
    """
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
    """
    Preprocess leukemia dataset.
    :return: Preprocessed data.
    """
    dataset = openml.datasets.get_dataset(1104)
    X, _, _, _ = dataset.get_data()
    y = X['CLASS']
    X = X.drop('CLASS', axis=1)
    y = y.map({'ALL': 1, 'AML': 0})
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
    return X, y, beta


def single_experiment(k, runs_num, num_feat, X_train, y_train, X_test, y_test, snr=None):
    """
    Single iterations of experiment on First Order only.
    :param k: Number of relevant features.
    :param runs_num: Number of runs.
    :param num_feat: Number of features.
    :param X_train: Explanatory training variables.
    :param y_train: Target training variable.
    :param X_test: Explanatory test variables.
    :param y_test: Target test variable.
    :param snr: Signal-to-noise ratio. If None, then variance of generated data is 1.
    """
    print(f"k={k}")
    scores, times, iterations = np.zeros(runs_num), np.zeros(runs_num), np.zeros(runs_num)
    for i in range(runs_num):
        print(f"Run {i + 1}/{runs_num}")
        e = np.random.multivariate_normal(np.zeros(num_feat), np.eye(num_feat)) * min(1, i)
        model = FOLeastSquares(k=k, method='interpolated', beta=e)
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        y_pred = model.predict(X_test)
        scores[i] = calculate_error(y_test, y_pred)
        times[i] = end_time - start_time
        iterations[i] = model.iterations_
    print(f"Lowest error for k={k}: {np.min(scores)}, achieved for i={np.argmin(scores)}")
    x = np.arange(runs_num)
    sns.lineplot(x=x, y=scores).set_title(f"Prediction error for k={k} and SNR={snr}")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()
    sns.lineplot(x=x, y=times).set_title(f"Training time for k={k} and SNR={snr}")
    plt.xlabel("Iterations")
    plt.ylabel("Training time")
    plt.show()
    sns.lineplot(x=x, y=iterations).set_title(f"Iterations for k={k} and SNR={snr}")
    plt.xlabel("Iterations")
    plt.ylabel("Number of iterations")
    plt.show()
