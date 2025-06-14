import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd


class SparseNetRegression(BaseEstimator, RegressorMixin):
    """
    SparseNet Regression implementation.

    SparseNet combines multiple penalty functions to achieve effective feature selection
    while maintaining good predictive performance.

    Parameters:
    -----------
    lambda_values : array-like, default=None
        Penalty parameters for each feature. If None, a single lambda value is used for all features.

    lambda_value : float, default=0.1
        Global penalty parameter when lambda_values is None.

    alpha_values : array-like, default=None
        Shape parameters for penalty functions. If None, a single alpha value is used.

    alpha_value : float, default=0.5
        Global shape parameter when alpha_values is None. Controls the mix between
        L1 (alpha=1) and L2 (alpha=0) penalties.

    penalty : str, default='elastic_net'
        Type of penalty to use. Options: 'elastic_net', 'scad', 'mcp'.

    max_iter : int, default=1000
        Maximum number of iterations for the optimization algorithm.

    tol : float, default=1e-4
        Tolerance for the optimization algorithm.

    standardize : bool, default=True
        Whether to standardize the input features.
    """

    def __init__(self, lambda_values=None, lambda_value=0.1, alpha_values=None,
                 alpha_value=0.5, penalty='elastic_net', max_iter=1000, tol=1e-4,
                 standardize=True):
        self.lambda_values = lambda_values
        self.lambda_value = lambda_value
        self.alpha_values = alpha_values
        self.alpha_value = alpha_value
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.coef_ = None
        self.intercept_ = 0.0
        self.scaler = StandardScaler() if standardize else None

    def _elastic_net_penalty(self, beta, alpha, lambda_val):
        """Elastic Net penalty: alpha * L1 + (1-alpha) * L2"""
        l1_penalty = np.sum(np.abs(beta))
        l2_penalty = np.sum(beta ** 2)
        return lambda_val * (alpha * l1_penalty + (1 - alpha) * l2_penalty / 2)

    def _scad_penalty(self, beta, alpha, lambda_val):
        """Smoothly Clipped Absolute Deviation (SCAD) penalty"""
        abs_beta = np.abs(beta)
        penalty = np.zeros_like(abs_beta)

        # Region 1: |beta| <= lambda
        idx1 = abs_beta <= lambda_val
        penalty[idx1] = lambda_val * abs_beta[idx1]

        # Region 2: lambda < |beta| <= alpha*lambda
        idx2 = (abs_beta > lambda_val) & (abs_beta <= alpha * lambda_val)
        penalty[idx2] = (2 * alpha * lambda_val * abs_beta[idx2] - abs_beta[idx2] ** 2 - lambda_val ** 2) / (
                    2 * (alpha - 1))

        # Region 3: |beta| > alpha*lambda
        idx3 = abs_beta > alpha * lambda_val
        penalty[idx3] = lambda_val ** 2 * (alpha + 1) / 2

        return np.sum(penalty)

    def _mcp_penalty(self, beta, alpha, lambda_val):
        """Minimax Concave Penalty (MCP)"""
        abs_beta = np.abs(beta)
        penalty = np.zeros_like(abs_beta)

        # Region 1: |beta| <= alpha*lambda
        idx1 = abs_beta <= alpha * lambda_val
        penalty[idx1] = lambda_val * abs_beta[idx1] - abs_beta[idx1] ** 2 / (2 * alpha)

        # Region 2: |beta| > alpha*lambda
        idx2 = abs_beta > alpha * lambda_val
        penalty[idx2] = alpha * lambda_val ** 2 / 2

        return np.sum(penalty)

    def _get_penalty(self, beta):
        """Calculate the total penalty based on the selected penalty type"""
        if self.lambda_values is None:
            lambda_vals = np.ones(len(beta)) * self.lambda_value
        else:
            lambda_vals = self.lambda_values

        if self.alpha_values is None:
            alpha_vals = np.ones(len(beta)) * self.alpha_value
        else:
            alpha_vals = self.alpha_values

        penalty = 0
        for i, (b, alpha, lambda_val) in enumerate(zip(beta, alpha_vals, lambda_vals)):
            if self.penalty == 'elastic_net':
                penalty += self._elastic_net_penalty(b, alpha, lambda_val)
            elif self.penalty == 'scad':
                penalty += self._scad_penalty(b, alpha, lambda_val)
            elif self.penalty == 'mcp':
                penalty += self._mcp_penalty(b, alpha, lambda_val)

        return penalty

    def _objective(self, beta, X, y):
        """Objective function: MSE + penalty"""
        mse = 0.5 * np.mean((y - X @ beta) ** 2)
        penalty = self._get_penalty(beta)
        return mse + penalty

    def _gradient(self, beta, X, y):
        """Gradient of the objective function"""
        n_samples = X.shape[0]
        grad_mse = -X.T @ (y - X @ beta) / n_samples

        # Add gradient of the penalty
        if self.penalty == 'elastic_net':
            if self.lambda_values is None:
                lambda_vals = np.ones(len(beta)) * self.lambda_value
            else:
                lambda_vals = self.lambda_values

            if self.alpha_values is None:
                alpha_vals = np.ones(len(beta)) * self.alpha_value
            else:
                alpha_vals = self.alpha_values

            grad_penalty = np.zeros_like(beta)
            for i, (b, alpha, lambda_val) in enumerate(zip(beta, alpha_vals, lambda_vals)):
                # Gradient of L1: sign(beta)
                grad_l1 = np.sign(b)
                # Gradient of L2: beta
                grad_l2 = b
                grad_penalty[i] = lambda_val * (alpha * grad_l1 + (1 - alpha) * grad_l2)
        else:
            # For SCAD and MCP, use numerical gradient
            eps = 1e-8
            grad_penalty = np.zeros_like(beta)
            for i in range(len(beta)):
                beta_plus = beta.copy()
                beta_plus[i] += eps
                beta_minus = beta.copy()
                beta_minus[i] -= eps
                grad_penalty[i] = (self._get_penalty(beta_plus) - self._get_penalty(beta_minus)) / (2 * eps)

        return grad_mse + grad_penalty

    def fit(self, X, y):
        """
        Fit the SparseNet model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Standardize features if required
        if self.standardize:
            X = self.scaler.fit_transform(X)

        # Center y (we'll store the mean as the intercept)
        self.intercept_ = np.mean(y)
        y_centered = y - self.intercept_

        # Initialize coefficients
        n_features = X.shape[1]
        beta_init = np.zeros(n_features)

        # Set up the optimization
        result = minimize(
            fun=self._objective,
            x0=beta_init,
            args=(X, y_centered),
            method='L-BFGS-B',
            jac=self._gradient,
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )

        self.coef_ = result.x
        self.n_iter_ = result.nit
        self.convergence_ = result.success

        # Apply thresholding to enforce sparsity
        self.coef_[np.abs(self.coef_) < 1e-6] = 0

        return self

    def predict(self, X):
        """
        Predict using the SparseNet model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Returns predicted values.
        """
        X = np.asarray(X)

        if self.standardize:
            X = self.scaler.transform(X)

        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True values for X.

        Returns:
        --------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        return r2_score(y, self.predict(X))


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression

    # Generate synthetic data
    X, y, coef = make_regression(
        n_samples=200,
        n_features=50,
        n_informative=10,
        noise=5.0,
        coef=True,
        random_state=42
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit SparseNet with Elastic Net penalty
    sparsenet_en = SparseNetRegression(
        lambda_value=0.1,
        alpha_value=0.5,
        penalty='elastic_net'
    )
    sparsenet_en.fit(X_train, y_train)

    # Fit SparseNet with SCAD penalty
    sparsenet_scad = SparseNetRegression(
        lambda_value=0.1,
        alpha_value=3.7,  # Common value for SCAD
        penalty='scad'
    )
    sparsenet_scad.fit(X_train, y_train)

    # Fit SparseNet with MCP penalty
    sparsenet_mcp = SparseNetRegression(
        lambda_value=0.1,
        alpha_value=3.0,  # Common value for MCP
        penalty='mcp'
    )
    sparsenet_mcp.fit(X_train, y_train)

    # Evaluate models
    print("Elastic Net SparseNet:")
    print(f"R² score: {sparsenet_en.score(X_test, y_test):.4f}")
    print(f"Non-zero coefficients: {np.count_nonzero(sparsenet_en.coef_)}")

    print("\nSCAD SparseNet:")
    print(f"R² score: {sparsenet_scad.score(X_test, y_test):.4f}")
    print(f"Non-zero coefficients: {np.count_nonzero(sparsenet_scad.coef_)}")

    print("\nMCP SparseNet:")
    print(f"R² score: {sparsenet_mcp.score(X_test, y_test):.4f}")
    print(f"Non-zero coefficients: {np.count_nonzero(sparsenet_mcp.coef_)}")

    # Plot true vs estimated coefficients
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.stem(coef, markerfmt='ro', label='True')
    plt.stem(sparsenet_en.coef_, markerfmt='bx', label='Elastic Net')
    plt.title('Elastic Net SparseNet')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.stem(coef, markerfmt='ro', label='True')
    plt.stem(sparsenet_scad.coef_, markerfmt='bx', label='SCAD')
    plt.title('SCAD SparseNet')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.stem(coef, markerfmt='ro', label='True')
    plt.stem(sparsenet_mcp.coef_, markerfmt='bx', label='MCP')
    plt.title('MCP SparseNet')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Advanced usage with feature-specific penalties
def advanced_example():
    from sklearn.datasets import make_regression

    # Generate synthetic data with varying coefficient magnitudes
    X, y, coef = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=10,
        noise=5.0,
        coef=True,
        random_state=42
    )

    # Make some coefficients larger than others
    coef[:5] *= 3
    y = X @ coef + np.random.normal(0, 5, size=200)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create feature-specific lambda values
    # Higher lambda for features we suspect are less important
    lambda_values = np.ones(X.shape[1]) * 0.1
    lambda_values[10:] = 0.5  # Higher penalty for potentially irrelevant features

    # Create feature-specific alpha values
    # Mix of L1 and L2 penalties for different features
    alpha_values = np.ones(X.shape[1]) * 0.5
    alpha_values[:5] = 0.2  # More L2 for important features (less shrinkage)
    alpha_values[15:] = 0.8  # More L1 for less important features (more sparsity)

    # Fit SparseNet with feature-specific penalties
    sparsenet = SparseNetRegression(
        lambda_values=lambda_values,
        alpha_values=alpha_values,
        penalty='elastic_net',
        max_iter=2000
    )
    sparsenet.fit(X_train, y_train)

    print("Feature-specific SparseNet:")
    print(f"R² score: {sparsenet.score(X_test, y_test):.4f}")
    print(f"Non-zero coefficients: {np.count_nonzero(sparsenet.coef_)}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.stem(coef, markerfmt='ro', label='True')
    plt.stem(sparsenet.coef_, markerfmt='bx', label='SparseNet')
    plt.title('Feature-specific SparseNet')
    plt.legend()
    plt.show()

    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': [f'X{i}' for i in range(X.shape[1])],
        'True Coefficient': coef,
        'SparseNet Coefficient': sparsenet.coef_,
        'Lambda': lambda_values,
        'Alpha': alpha_values
    })
    print(feature_importance.sort_values(by='SparseNet Coefficient', key=abs, ascending=False))