"""
Experiments for p being greater than n.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split, KFold

from src.mio_solver import MIOSolver
from src.models import FOLeastSquares
from src.stepwise import StepwiseSelection
from src.sparsenet import SparseNetRegression
from src.utils import generate_data, preprocess_leukemia_data, calculate_error, single_experiment

np.random.seed(42)
sns.set_style('whitegrid')


def experiment_1(runs_num=50):
    """
    Comparison of first order method for regression for different initializations of beta and different numbers of
    relevant features.
    :param runs_num: Number of beta initializations per one k value.
    """
    print("EXPERIMENT START")
    for snr in [3, 7]:
        X, y, _, _, _ = generate_data(2000, 30, method='2', snr=snr, num=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        num_feat = X.shape[1]
        for k in [5, 6, 7, 8, 9]:
            single_experiment(k, runs_num, num_feat, X_train, y_train, X_test, y_test, snr)


def experiment_2(runs_num=50):
    """
    Comparison of first order method for regression for different initializations of beta and different numbers of
    relevant features.
    :param runs_num: Number of beta initializations per one k value.
    """
    print("EXPERIMENT START")
    for snr in [3, 7]:
        X, y, _ = preprocess_leukemia_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        num_feat = X.shape[1]
        for k in [6, 8, 10, 12, 16, 18]:
            single_experiment(k, runs_num, num_feat, X_train, y_train, X_test, y_test, snr)


def experiment_3(runs_num=10):
    """
    Comparison of first order method, LASSO and Stepwise for regression for different values of rho and different
    values of SNR, based on prediction error and number of non-zero coefficients.
    :param runs_num: Number of runs per one combination of rho and SNR.
    """
    df = []
    snrs = [3, 7, 10]
    methods = ['1', '2', '3', '4']
    ks = {'1': 5, '2': 5, '3': 10, '4': 10}
    print("EXPERIMENT START")
    for snr in snrs:
        print(f"SNR: {snr}")
        for method in methods:
            if method == '1':
                X, y, beta, train, test = generate_data(50, 1000, method=method, snr=snr, num=runs_num, rho=0.8)
            elif method == '2':
                X, y, beta, train, test = generate_data(30, 1000, method=method, snr=snr, num=runs_num)
            elif method == '3':
                X, y, beta, train, test = generate_data(30, 1000, method=method, snr=snr, num=runs_num)
            else:
                X, y, beta, train, test = generate_data(50, 2000, method=method, snr=snr, num=runs_num)
            for i in range(runs_num):
                print(f"Run {i + 1}/{runs_num}")
                y_temp = y[i]
                X_train = X[train]
                y_train = y_temp[train]
                X_test = X[test]

                # True signal for prediction error calculation
                true_signal_test = X_test @ beta

                # 1. First-order method
                model_fo = FOLeastSquares(k=ks[method], method='interpolated')
                model_fo.fit(X_train, y_train)
                y_pred_fo = model_fo.predict(X_test)
                pred_error_fo = calculate_error(true_signal_test, y_pred_fo)
                nonzeros_fo = np.count_nonzero(model_fo.beta_)

                # 2. MIO
                model_mio = MIOSolver(k=ks[method], time_limit=500)
                model_mio.fit(X_train, y_train)
                y_pred_mio = model_mio.predict(X_test)
                pred_error_mio = calculate_error(true_signal_test, y_pred_mio)
                nonzeros_mio = np.count_nonzero(model_mio.beta_)

                # 3. Lasso
                model_lasso = Lasso(alpha=0.1)
                model_lasso.fit(X_train, y_train)
                y_pred_lasso = model_lasso.predict(X_test)
                pred_error_lasso = calculate_error(true_signal_test, y_pred_lasso)
                nonzeros_lasso = np.count_nonzero(model_lasso.coef_)

                # 4. Stepwise regression
                model_step = StepwiseSelection()
                model_step.fit(X_train, y_train)
                y_pred_step = model_step.predict(X_test)
                pred_error_step = calculate_error(true_signal_test, y_pred_step)
                nonzeros_step = np.count_nonzero(model_step.beta_)

                # 5. SparseNet regression
                model_sparse = SparseNetRegression()
                model_sparse.fit(X_train, y_train)
                y_pred_sparse = model_sparse.predict(X_test)
                pred_error_sparse = calculate_error(true_signal_test, y_pred_sparse)
                nonzeros_sparse = np.count_nonzero(model_sparse.coef_)

                # Append results
                for method, pred_error, nonzeros in zip(
                        ['First-order', 'MIO', 'Lasso', 'Stepwise', "SparseNet"],
                        [pred_error_fo, pred_error_mio, pred_error_lasso, pred_error_step, pred_error_sparse],
                        [nonzeros_fo, nonzeros_mio, nonzeros_lasso, nonzeros_step, nonzeros_sparse]
                ):
                    df.append({
                        'Method': method,
                        'Metric': pred_error,
                        'SNR': str(snr),
                        'Metric name': 'Prediction error'
                    })
                    df.append({
                        'Method': method,
                        'Metric': nonzeros,
                        'SNR': str(snr),
                        'Metric name': 'Non-zero coefficients'
                    })
    df = pd.DataFrame(df)
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(13, 9))
    ax[0, 0].set_ylabel("Prediction error")
    ax[1, 0].set_ylabel("Non-zero coefficients")
    for i, method in enumerate(methods):
        sns.barplot(data=df[(df['Metric name'] == 'Prediction error') & (df['Data'] == method)], x="SNR", y="Metric",
                    hue="Method", ax=ax[0, i])
        sns.barplot(data=df[(df['Metric name'] == 'Non-zero coefficients') & (df['Data'] == method)], x="SNR",
                    y="Metric",
                    hue="Method", ax=ax[1, i])
        ax[1, i].axhline(y=ks[method], color='r', linestyle='--')
        ax[1, i].set_xlabel(f"Method={method}")
    plt.show()


def main():
    experiment_1()
    experiment_2()
    experiment_3()


if __name__ == '__main__':
    main()
