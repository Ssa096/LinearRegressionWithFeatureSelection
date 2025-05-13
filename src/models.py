"""
Implementation of the first order method for least squares regression and least absolute deviation.
"""
import numpy as np
from scipy.optimize import minimize


def _generate_beta(p, k):
    beta = np.random.normal(size=p)
    k_choice = np.random.permutation(p)[k:]
    beta[k_choice] = 0
    return beta


class FOLeastSquares:
    """
    First Order Least Squares Regression.
    """

    def __init__(self, k, L=None, method='iterative', fit_intercept=True, max_iter=1000, tol=0.0001,
                 solver='auto', refine_least_squares=False):
        self.k = k
        self.L = L
        self.method = method
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.refine_least_squares = refine_least_squares
        self.beta_ = None
        self.intercept_ = None

    @staticmethod
    def _gradient(X, y, beta):
        return -X.T @ (y - X @ beta)

    @staticmethod
    def _hard_thresholding(v, k):
        idx = np.argsort(np.abs(v))[-k:]
        w = np.zeros_like(v)
        w[idx] = v[idx]
        return w

    def _iterative_hard_thresholding(self, X, y):
        n, p = X.shape
        beta = _generate_beta(p, self.k)
        if self.L is None:
            self.L = np.linalg.norm(X.T @ X, 2)

        for _ in range(self.max_iter):
            grad = self._gradient(X, y, beta)
            beta_new = self._hard_thresholding(beta - (1 / self.L) * grad, self.k)
            if np.abs(np.linalg.norm(X @ beta - y) ** 2 - np.linalg.norm(X @ beta_new - y) ** 2) < self.tol:
                break
            beta = beta_new

        intercept = np.mean(y) - np.mean(X, axis=0) @ beta if self.fit_intercept else 0.0

        return beta, intercept

    def _interpolated_hard_thresholding(self, X, y):
        n, p = X.shape
        beta = _generate_beta(p, self.k)
        if self.L is None:
            self.L = np.linalg.norm(X.T @ X, 2)

        best_loss = np.inf

        for _ in range(self.max_iter):
            grad = self._gradient(X, y, beta)
            eta = self._hard_thresholding(beta - (1 / self.L) * grad, self.k)

            lambda_star = minimize(lambda l: np.linalg.norm(y - X @ (l * eta + (1 - l) * beta)) ** 2 / 2,
                                   np.array([0.5]),
                                   method='Nelder-Mead' if self.solver == 'auto' else self.solver,
                                   options={'maxiter': 1000} if self.solver == 'auto' else {})
            lambda_star = lambda_star.x[0]

            beta_new = lambda_star * eta + (1 - lambda_star) * beta

            if np.abs(np.linalg.norm(X @ beta_new - y) ** 2 - np.linalg.norm(X @ beta - y) ** 2) < self.tol:
                break

            curr_loss = np.linalg.norm(y - X @ (lambda_star * eta + (1 - lambda_star) * beta)) ** 2 / 2
            best_loss = min(best_loss, curr_loss)

            beta = beta_new

        intercept = np.mean(y) - np.mean(X, axis=0) @ beta if self.fit_intercept else 0.0

        return beta, intercept

    def _refine_with_least_squares(self, X, y):
        support = np.flatnonzero(self.beta_)
        if len(support) == 0:
            return self.beta_, self.intercept_
        X_sub = X[:, support]
        beta_refined = np.linalg.pinv(X_sub.T @ X_sub) @ X_sub.T @ y
        beta_final = np.zeros_like(self.beta_)
        beta_final[support] = beta_refined
        intercept = np.mean(y) - np.mean(X, axis=0) @ beta_final if self.fit_intercept else 0.0
        return beta_final, intercept

    def fit(self, X, y):
        if self.method == 'iterative':
            self.beta_, self.intercept_ = self._iterative_hard_thresholding(X, y)
        elif self.method == 'interpolated':
            self.beta_, self.intercept_ = self._interpolated_hard_thresholding(X, y)
        if self.refine_least_squares:
            self.beta_, self.intercept_ = self._refine_with_least_squares(X, y)
        return self

    def predict(self, X):
        return X @ self.beta_ + self.intercept_

    def get_params(self):
        return {'L': self.L, 'method': self.method, 'fit_intercept': self.fit_intercept,
                'max_iter': self.max_iter, 'tol': self.tol, 'solver': self.solver}

    def score(self, X, y):
        y = y.astype(float)
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u / v


class FOLAD:
    """
    First Order Least Absolute Deviation.
    """

    def __init__(self, k, method='iterative', fit_intercept=True, max_iter=1000, tol=0.0001,
                 solver='auto', refine_least_squares=False, annealing=1, gamma=0.8, threshold=0.0001):
        self.k = k
        self.method = method
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.refine_least_squares = refine_least_squares
        self.annealing = annealing
        self.gamma = gamma
        self.threshold = threshold
        self.beta_ = None
        self.intercept_ = None

    @staticmethod
    def min_arg(X, y, beta, annealing):
        return np.clip((y - X @ beta) / annealing, -1, 1)

    @staticmethod
    def _gradient(X, y, beta, annealing):
        return -X.T @ np.clip((y - X @ beta) / annealing, -1, 1)

    @staticmethod
    def _hard_thresholding(v, k):
        idx = np.argsort(np.abs(v))[-k:]
        w = np.zeros_like(v)
        w[idx] = v[idx]
        return w

    def _iterative_hard_thresholding(self, X, y):
        n, p = X.shape
        beta = _generate_beta(p, self.k) if self.beta_ is None else self.beta_
        L = np.linalg.norm(X.T @ X, 2) / self.annealing

        for _ in range(self.max_iter):
            grad = self._gradient(X, y, beta, self.annealing)
            beta_new = self._hard_thresholding(beta - (1 / L) * grad, self.k)
            arg_min_new = self.min_arg(X, y, beta_new, self.annealing)
            arg_min = self.min_arg(X, y, beta, self.annealing)
            if np.abs((np.dot(X @ beta_new - y, arg_min_new) - self.annealing * np.linalg.norm(arg_min_new) ** 2 / 2)
                      - (np.dot(X @ beta - y, arg_min) - self.annealing * np.linalg.norm(arg_min) ** 2 / 2)) < self.tol:
                break
            beta = beta_new

        intercept = np.mean(y) - np.mean(X, axis=0) @ beta if self.fit_intercept else 0.0

        return beta, intercept

    def _interpolated_hard_thresholding(self, X, y):
        n, p = X.shape
        beta = _generate_beta(p, self.k) if self.beta_ is None else self.beta_
        L = np.linalg.norm(X.T @ X, 2) / self.annealing

        best_loss = np.inf

        for _ in range(self.max_iter):
            grad = self._gradient(X, y, beta, self.annealing)
            eta = self._hard_thresholding(beta - (1 / L) * grad, self.k)

            lambda_star = minimize(lambda l: np.linalg.norm(y - X @ (l * eta + (1 - l) * beta)) ** 2 / 2,
                                   np.array([0.5]),
                                   method='Nelder-Mead' if self.solver == 'auto' else self.solver,
                                   options={'maxiter': 1000} if self.solver == 'auto' else {})
            lambda_star = lambda_star.x[0]

            beta_new = lambda_star * eta + (1 - lambda_star) * beta
            arg_min_new = self.min_arg(X, y, beta_new, self.annealing)
            arg_min = self.min_arg(X, y, beta, self.annealing)
            if np.abs((np.dot(X @ beta_new - y, arg_min_new) - self.annealing * np.linalg.norm(arg_min_new) ** 2 / 2)
                      - (np.dot(X @ beta - y, arg_min) - self.annealing * np.linalg.norm(arg_min) ** 2 / 2)) < self.tol:
                break

            curr_loss = np.linalg.norm(y - X @ (lambda_star * eta + (1 - lambda_star) * beta)) ** 2 / 2
            best_loss = min(best_loss, curr_loss)

            beta = beta_new

        intercept = np.mean(y) - np.mean(X, axis=0) @ beta if self.fit_intercept else 0.0

        return beta, intercept

    def _refine_with_least_squares(self, X, y):
        support = np.flatnonzero(self.beta_)
        if len(support) == 0:
            return self.beta_, self.intercept_
        X_sub = X[:, support]
        beta_refined = np.linalg.pinv(X_sub.T @ X_sub) @ X_sub.T @ y
        beta_final = np.zeros_like(self.beta_)
        beta_final[support] = beta_refined
        intercept = np.mean(y) - np.mean(X, axis=0) @ beta_final if self.fit_intercept else 0.0
        return beta_final, intercept

    def fit(self, X, y):
        while self.annealing > self.threshold:
            if self.method == 'iterative':
                self.beta_, self.intercept_ = self._iterative_hard_thresholding(X, y)
            elif self.method == 'interpolated':
                self.beta_, self.intercept_ = self._interpolated_hard_thresholding(X, y)
            self.annealing *= self.gamma
        if self.refine_least_squares:
            self.beta_, self.intercept_ = self._refine_with_least_squares(X, y)
        return self

    def predict(self, X):
        return X @ self.beta_ + self.intercept_

    def get_params(self):
        return {'method': self.method, 'fit_intercept': self.fit_intercept,
                'max_iter': self.max_iter, 'tol': self.tol, 'solver': self.solver}

    def score(self, X, y):
        y = y.astype(float)
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - u / v