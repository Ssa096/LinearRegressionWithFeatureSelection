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


def _calculate_beta(y, X, eta, beta, solver):
    lambda_star = minimize(lambda l: np.linalg.norm(y - X @ (l * eta + (1 - l) * beta)) ** 2 / 2,
                           np.array([0.5]),
                           method='Nelder-Mead' if solver == 'auto' else solver,
                           options={'maxiter': 1000} if solver == 'auto' else {})
    lambda_star = lambda_star.x[0]

    return lambda_star * eta + (1 - lambda_star) * beta, lambda_star


def _refine_with_least_squares(X, y, beta, intercept, fit_intercept=True):
    support = np.flatnonzero(beta)
    if len(support) == 0:
        return beta, intercept
    X_sub = X[:, support]
    beta_refined = np.linalg.pinv(X_sub.T @ X_sub) @ X_sub.T @ y
    beta_final = np.zeros_like(beta)
    beta_final[support] = beta_refined
    intercept = np.mean(y) - np.mean(X, axis=0) @ beta_final if fit_intercept else 0.0
    return beta_final, intercept


class FOLeastSquares:
    """
    First Order Least Squares Regression.
    """

    def __init__(self, k, L=None, method='iterative', fit_intercept=True, max_iter=1000, tol=0.0001,
                 solver='auto', refine_least_squares=False, beta=None):
        self.k = k
        self.L = L
        self.method = method
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.refine_least_squares = refine_least_squares
        self.beta_ = beta
        self.intercept_ = None
        self.iterations_ = None

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

        for i in range(self.max_iter):
            grad = self._gradient(X, y, beta)
            beta_new = self._hard_thresholding(beta - (1 / self.L) * grad, self.k)
            if np.abs(np.linalg.norm(X @ beta - y) ** 2 - np.linalg.norm(X @ beta_new - y) ** 2) < self.tol:
                self.iterations_ = i + 1
                break
            beta = beta_new

        self.iterations_ = self.max_iter if self.iterations_ is None else self.iterations_
        intercept = np.mean(y) - np.mean(X, axis=0) @ beta if self.fit_intercept else 0.0

        return beta, intercept

    def _interpolated_hard_thresholding(self, X, y):
        n, p = X.shape
        if self.beta_ is None:
            beta = _generate_beta(p, self.k)
        else:
            beta = self.beta_
        if self.L is None:
            self.L = np.linalg.norm(X.T @ X, 2)

        best_loss = np.inf

        for i in range(self.max_iter):
            grad = self._gradient(X, y, beta)
            eta = self._hard_thresholding(beta - (1 / self.L) * grad, self.k)

            beta_new, lambda_star = _calculate_beta(y, X, eta, beta, self.solver)

            if np.abs(np.linalg.norm(X @ beta_new - y) ** 2 - np.linalg.norm(X @ beta - y) ** 2) < self.tol:
                self.iterations_ = i + 1
                break

            curr_loss = np.linalg.norm(y - X @ (lambda_star * eta + (1 - lambda_star) * beta)) ** 2 / 2
            best_loss = min(best_loss, curr_loss)

            beta = beta_new

        self.iterations_ = self.max_iter if self.iterations_ is None else self.iterations_
        intercept = np.mean(y) - np.mean(X, axis=0) @ beta if self.fit_intercept else 0.0

        return beta, intercept

    def fit(self, X, y):
        """
        Fit the model.
        :param X: Training data.
        :param y: Target values.
        :return: Fitted model.
        """
        if self.method == 'iterative':
            self.beta_, self.intercept_ = self._iterative_hard_thresholding(X, y)
        elif self.method == 'interpolated':
            self.beta_, self.intercept_ = self._interpolated_hard_thresholding(X, y)
        if self.refine_least_squares:
            self.beta_, self.intercept_ = _refine_with_least_squares(X, y, self.beta_, self.intercept_,
                                                                     self.fit_intercept)
        return self

    def predict(self, X):
        """
        Predict using the model.
        :param X: Test data.
        :return: Predicted values.
        """
        return X @ self.beta_ + self.intercept_

    def get_params(self):
        """
        Get model parameters.
        :return: Dictionary of parameters.
        """
        return {'L': self.L, 'method': self.method, 'fit_intercept': self.fit_intercept,
                'max_iter': self.max_iter, 'tol': self.tol, 'solver': self.solver}


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
        """
        Function returning minimal argument of LAD function.
        :param X: Training data.
        :param y: Target values.
        :param beta: Coefficients.
        :param annealing: Annealing parameter.
        :return: Minimal argument.
        """
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

    def _arg_mins(self, X, y, beta, beta_new):
        return self.min_arg(X, y, beta_new, self.annealing), self.min_arg(X, y, beta, self.annealing)

    def _iterative_hard_thresholding(self, X, y):
        n, p = X.shape
        beta = _generate_beta(p, self.k) if self.beta_ is None else self.beta_
        L = np.linalg.norm(X.T @ X, 2) / self.annealing

        for _ in range(self.max_iter):
            grad = self._gradient(X, y, beta, self.annealing)
            beta_new = self._hard_thresholding(beta - (1 / L) * grad, self.k)
            arg_min_new, arg_min = self._arg_mins(X, y, beta_new, beta)
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

            beta_new, lambda_star = _calculate_beta(y, X, eta, beta, self.solver)
            arg_min_new, arg_min = self._arg_mins(X, y, beta_new, beta)
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
        """
        Fit the model.
        :param X: Training data.
        :param y: Target values.
        :return: Fitted model.
        """
        while self.annealing > self.threshold:
            if self.method == 'iterative':
                self.beta_, self.intercept_ = self._iterative_hard_thresholding(X, y)
            elif self.method == 'interpolated':
                self.beta_, self.intercept_ = self._interpolated_hard_thresholding(X, y)
            self.annealing *= self.gamma
        if self.refine_least_squares:
            self.beta_, self.intercept_ = _refine_with_least_squares(X, y, self.beta_, self.intercept_,
                                                                     self.fit_intercept)
        return self

    def predict(self, X):
        """
        Predict using the model.
        :param X: Test data.
        :return: Predicted values.
        """
        return X @ self.beta_ + self.intercept_

    def get_params(self):
        """
        Get model parameters.
        :return: Dictionary of parameters.
        """
        return {'method': self.method, 'fit_intercept': self.fit_intercept,
                'max_iter': self.max_iter, 'tol': self.tol, 'solver': self.solver}
