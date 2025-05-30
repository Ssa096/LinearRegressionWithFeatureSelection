"""
Experiments for n being greater than p.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split, KFold

from src.models import FOLeastSquares
from src.utils import generate_data, preprocess_diabetes_data, calculate_error, single_experiment

np.random.seed(42)
sns.set_style('whitegrid')


def experiment_1(runs_num=50):
    """
    Comparison of first order method for regression for different initializations of beta and different numbers of
    relevant features.
    :param runs_num: Number of beta initializations per one k value.
    """
    print("EXPERIMENT START")
    X, y = preprocess_diabetes_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_feat = X.shape[1]
    for k in [9, 20, 49, 57]:
        single_experiment(k, runs_num, num_feat, X_train, y_train, X_test, y_test)


def experiment_2(runs_num=10, n=500, p=100, k=10):
    """
    Comparison of first order method, LASSO and Stepwise for regression for different values of rho and different
    values of SNR, based on prediction error and number of non-zero coefficients.
    :param runs_num: Number of runs per one combination of rho and SNR.
    :param n: Number of samples.
    :param p: Number of features.
    :param k: Number of relevant features.
    """
    df = []
    rhos = [0.5, 0.8, 0.9]
    snrs = [[1.58, 3.17, 6.33], [1.74, 3.48, 6.97], [2.18, 4.37, 8.73]]
    print("EXPERIMENT START")
    for rho, snr in zip(rhos, snrs):
        print(f"RHO: {rho}")
        print(f"SNR: {snr}")
        for single_snr in snr:
            X, y, beta, train, test = generate_data(n, p, k, rho=rho, num=runs_num, snr=single_snr)
            for i in range(runs_num):
                print(f"Run {i + 1}/{runs_num}")
                y_temp = y[i]
                X_train = X[train]
                y_train = y_temp[train]
                X_test = X[test]
                y_test = y_temp[test]
                model = FOLeastSquares(k=k, method='interpolated')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                df.append({"Method": "First-order", "Metric": calculate_error(y_test, y_pred), "Rho": rho,
                           "SNR": str(single_snr), "Metric name": "Prediction error"})
                df.append({"Method": "First-order", "Metric": np.count_nonzero(model.beta_),
                           "Rho": rho, "SNR": str(single_snr), "Metric name": "Non-zero coefficients"})
                model = Lasso(alpha=0.1)
                model.fit(X_train, y_train)
                df.append({"Method": "LASSO", "Metric": calculate_error(y_test, model.predict(X_test)), "Rho": rho,
                           "SNR": str(single_snr), "Metric name": "Prediction error"})
                df.append({"Method": "LASSO", "Metric": np.count_nonzero(model.coef_), "Rho": rho,
                           "SNR": str(single_snr), "Metric name": "Non-zero coefficients"})
                model = RFECV(LinearRegression(), cv=KFold(), scoring='neg_mean_squared_error')
                model.fit(X_train, y_train)
                df.append({"Method": "Stepwise", "Metric": calculate_error(y_test, model.predict(X_test)),
                           "Rho": rho, "SNR": str(single_snr), "Metric name": "Prediction error"})
                df.append({"Method": "Stepwise", "Metric": np.count_nonzero(model.support_),
                           "Rho": rho, "SNR": str(single_snr), "Metric name": "Non-zero coefficients"})
    df = pd.DataFrame(df)
    fig, ax = plt.subplots(nrows=2, ncols=len(rhos), figsize=(13, 9))
    ax[0, 0].set_ylabel("Prediction error")
    ax[1, 0].set_ylabel("Non-zero coefficients")
    for i, rho in enumerate(rhos):
        sns.barplot(data=df[(df['Metric name'] == 'Prediction error') & (df['Rho'] == rho)], x="SNR", y="Metric",
                    hue="Method", ax=ax[0, i])
        sns.barplot(data=df[(df['Metric name'] == 'Non-zero coefficients') & (df['Rho'] == rho)], x="SNR", y="Metric",
                    hue="Method", ax=ax[1, i])
        ax[1, i].axhline(y=k, color='r', linestyle='--')
        ax[1, i].set_xlabel(f"Rho={rho}")
    plt.show()


def main():
    experiment_1()
    experiment_2()


if __name__ == '__main__':
    main()
