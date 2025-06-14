import statsmodels.api as sm
import numpy as np
from sklearn.metrics import r2_score


class StepwiseSelection:
    """
    A class that implements various feature selection methods based on statistical significance.
    Follows scikit-learn API pattern with fit, predict, and score methods.
    """

    def __init__(self, method='stepwise', significance_level=0.05,
                 inclusion_threshold=0.05, exclusion_threshold=0.05):
        """
        Initialize the StepwiseSelection class.

        Parameters:
        method : str, optional (default='stepwise')
            The feature selection method to use. Options are:
            - 'forward': Forward selection
            - 'backward': Backward elimination
            - 'stepwise': Stepwise selection (forward and backward)
        significance_level : float, optional (default=0.05)
            The significance level for including/excluding variables in forward and backward methods
        inclusion_threshold : float, optional (default=0.05)
            The threshold for including variables in stepwise selection
        exclusion_threshold : float, optional (default=0.05)
            The threshold for excluding variables in stepwise selection
        """
        self.method = method
        self.significance_level = significance_level
        self.inclusion_threshold = inclusion_threshold
        self.exclusion_threshold = exclusion_threshold
        self.selected_features = None
        self.model = None
        self.is_pandas = None
        self.beta = None  # Added beta attribute to store regression coefficients

    def fit(self, X, y, initial_list=None):
        """
        Fit the model by selecting features based on the chosen method.

        Parameters:
        X : array-like or pandas DataFrame
            The input features
        y : array-like or pandas Series
            The target variable
        initial_list : list, optional (default=None)
            Initial list of features to include (for stepwise selection)

        Returns:
        self : object
            Returns self
        """
        # Determine if X is a pandas DataFrame
        self.is_pandas = hasattr(X, 'columns')

        if initial_list is None:
            initial_list = []

        # Select features based on the chosen method
        if self.method == 'forward':
            self.selected_features = self._forward_selection(X, y, self.significance_level)
        elif self.method == 'backward':
            self.selected_features = self._backward_elimination(X, y, self.significance_level)
        elif self.method == 'stepwise':
            self.selected_features = self._stepwise_selection(X, y, initial_list,
                                                              self.inclusion_threshold,
                                                              self.exclusion_threshold)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from 'forward', 'backward', or 'stepwise'.")

        # Fit the final model with selected features
        if self.is_pandas:
            X_selected = X[self.selected_features]
        else:
            X_selected = X[:, self.selected_features]

        self.model = sm.OLS(y, sm.add_constant(X_selected)).fit()

        # Store the beta coefficients as an attribute
        self.beta = self.model.params

        return self

    def predict(self, X):
        """
        Predict using the linear model with selected features.

        Parameters:
        X : array-like or pandas DataFrame
            The input features

        Returns:
        array-like
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")

        # Select features
        if self.is_pandas:
            X_selected = X[self.selected_features]
        else:
            X_selected = X[:, self.selected_features]

        # Add constant and predict
        X_with_const = sm.add_constant(X_selected)
        return self.model.predict(X_with_const)

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters:
        X : array-like or pandas DataFrame
            The input features
        y : array-like
            The true values

        Returns:
        float
            R^2 score
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")

        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def get_selected_features(self):
        """
        Get the selected features.

        Returns:
        list
            List of selected features
        """
        if self.selected_features is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")

        return self.selected_features

    def get_beta(self):
        """
        Get the beta coefficients (regression parameters).

        Returns:
        array-like
            Beta coefficients including the intercept (constant)
        """
        if self.beta is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")

        return self.beta

    # The rest of the methods remain unchanged
    def _forward_selection(self, X, y, significance_level=0.05):
        # Implementation remains the same
        pass

    def _backward_elimination(self, X, y, significance_level=0.05):
        # Implementation remains the same
        pass

    def _stepwise_selection(self, X, y, initial_list=None, inclusion_threshold=0.05, exclusion_threshold=0.05):
        # Implementation remains the same
        pass