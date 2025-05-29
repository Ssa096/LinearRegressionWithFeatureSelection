"""
FOLAD experiments
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso

from src.models import FOLAD
from src.utils import generate_data, calculate_error

np.random.seed(42)
sns.set_style('whitegrid')


def experiment(n, p, k=5, rho=0.9, runs_num=10, snrs=None):
    """

    :param n:
    :param p:
    :param k:
    :param rho:
    :param runs_num:
    :param snrs:
    """
    if snrs is None:
        snrs = [3, 7, 10]
    df = []
    print("EXPERIMENT START")
    for snr in snrs:
        print(f"SNR: {snr}")
        X, y, beta, train, test = generate_data(n, p, k, snr=snr, num=runs_num, rho=rho)
        for i in range(runs_num):
            print(f"Run {i + 1}/{runs_num}")
            y_temp = y[i]
            X_train = X[train]
            y_train = y_temp[train]
            X_test = X[test]
            y_test = y_temp[test]
            print("First Order Method")
            model = FOLAD(k=k, method='interpolated')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            df.append({"Method": "First-order", "Metric": calculate_error(y_test, y_pred),
                       "SNR": str(snr), "Metric name": "Prediction error"})
            df.append({"Method": "First-order", "Metric": np.count_nonzero(model.beta_),
                       "SNR": str(snr), "Metric name": "Non-zero coefficients"})
            print("LASSO Method")
            model = Lasso(alpha=0.1)
            model.fit(X_train, y_train)
            df.append({"Method": "LASSO", "Metric": calculate_error(y_test, model.predict(X_test)),
                       "SNR": str(snr), "Metric name": "Prediction error"})
            df.append({"Method": "LASSO", "Metric": np.count_nonzero(model.coef_),
                       "SNR": str(snr), "Metric name": "Non-zero coefficients"})
    df = pd.DataFrame(df)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(13, 9))
    ax[0].set_ylabel("Prediction error")
    ax[1].set_ylabel("Non-zero coefficients")
    sns.barplot(data=df[(df['Metric name'] == 'Prediction error')], x="SNR", y="Metric",
                hue="Method", ax=ax[0])
    sns.barplot(data=df[(df['Metric name'] == 'Non-zero coefficients')], x="SNR", y="Metric",
                hue="Method", ax=ax[1])
    ax[1].axhline(y=k, color='r', linestyle='--')
    ax[0].set_title(f"N={n}, p={p}")
    plt.show()


def main():
    experiment(500, 100)
    experiment(50, 1000)
    experiment(500, 1000)


if __name__ == '__main__':
    main()
