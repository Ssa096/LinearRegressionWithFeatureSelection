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

from models import FOLeastSquares
from mio_solver import MIOSolver
from utils import generate_data, preprocess_diabetes_data, calculate_error, single_experiment
from stepwise import StepwiseSelection
from sparsenet import SparseNetRegression

np.random.seed(42)
sns.set_style('whitegrid')


def experiment_1(runs_num=20):
    """
    Comparison of first order method for regression for different initializations of beta and different numbers of
    relevant features.
    :param runs_num: Number of beta initializations per one k value.
    """
    print("EXPERIMENT START")
    X, y = preprocess_diabetes_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_feat = X.shape[1]
    results=[]
    for k in [9, 20, 49, 57]:
        results = single_experiment(k, runs_num, num_feat, X_train, y_train, X_test, y_test, results=results)

    results_df = pd.DataFrame(results)
    print("\nResults Table:")
    print(results_df)
    results_df.to_csv('experiment1_results.csv', index=False)
    
    pivot_time = results_df.pivot(index='k', columns='Method', values='Time')
    pivot_acc = results_df.pivot(index='k', columns='Method', values='Accuracy')
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    pivot_time.plot(kind='bar', ax=ax[0])
    ax[0].set_title('Time to Best Solution (s)')
    ax[0].set_ylabel('Seconds')
    
    pivot_acc.plot(kind='bar', ax=ax[1])
    ax[1].set_title('Relative Suboptimality')
    ax[1].set_ylabel('(f_alg - f*)/f*')
    
    plt.tight_layout()
    plt.savefig('experiment1_results.png')
    plt.show()    


def experiment_optimality_gap():
    """
    Replicates Figure 5 from section 5.2.2 of the paper.
    Evaluates how MIO performs under cold vs warm start for various time limits.
    """
    X, y = preprocess_diabetes_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    k_values = [9, 20, 49, 57]
    #k_values = [49, 57]
    time_limit = 200  
    runs_num = 50
    
    results = {k: {'with_ws': None, 'without_ws': None} for k in k_values}

    for k in k_values:
        print(f"Running k={k} without warm start...")
        solver_no_ws = MIOSolver(k=k, time_limit=time_limit)
        solver_no_ws.fit(X, y)
        results[k]['without_ws'] = solver_no_ws.gap_progress
        fo_results = []

        for i in range(runs_num):
            print(f"Run {i + 1}/{runs_num}")
            e = np.random.multivariate_normal(np.zeros(64), np.eye(64)) * min(1, i)
            model = FOLeastSquares(k=k, method='interpolated', beta=e)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            obj_val = 0.5 * np.sum((y_train - y_pred)**2)

            fo_results.append({
                'beta': model.beta_,
                'obj_val': obj_val
            })

        best_fo = min(fo_results, key=lambda x: x['obj_val'])
        
        solver_ws = MIOSolver(k=k, time_limit=time_limit, 
                                        warm_start_beta=best_fo['beta'])
        solver_ws.fit(X, y)
        results[k]['with_ws'] = solver_ws.gap_progress


    for k in k_values:
        plt.figure(figsize=(8, 6))
        
        # With warm start
        gap_progress_ws = results[k]['with_ws']
        if gap_progress_ws:
            times_ws, gaps_ws = zip(*gap_progress_ws)
            log_gaps_ws = np.log10(np.maximum(gaps_ws, 1e-16))
            plt.plot(times_ws, log_gaps_ws, 'r-', linewidth=2, label='Warm Start')
        
        # Without warm start
        gap_progress_nws = results[k]['without_ws']
        if gap_progress_nws:
            times_nws, gaps_nws = zip(*gap_progress_nws)
            log_gaps_nws = np.log10(np.maximum(gaps_nws, 1e-16))
            plt.plot(times_nws, log_gaps_nws, 'b-', linewidth=2, label='Cold Start')
        
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('$\log_{10}$(Optimality Gap)', fontsize=12)
        plt.title(f'MIO Optimality Gap Evolution (k={k})\nDiabetes Dataset (n=350, p=64)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if k==39:
            plt.ylim(-4, 0)
        else:
            plt.ylim(-2.5, 0.5)
        
        plt.tight_layout()
        plt.savefig(f'mio_gap_k_{k}.png', dpi=300)
        plt.close()



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
            X, y, beta_true, train, test = generate_data(n, p, k, rho=rho, num=runs_num, snr=single_snr)
            for i in range(runs_num):
                print(f"Run {i + 1}/{runs_num}")
                y_temp = y[i]
                X_train = X[train]
                y_train = y_temp[train]
                X_test = X[test]
                
                # True signal for prediction error calculation
                true_signal_test = X_test @ beta_true
                
                # 1. First-order method
                model_fo = FOLeastSquares(k=k, method='interpolated')
                model_fo.fit(X_train, y_train)
                y_pred_fo = model_fo.predict(X_test)
                pred_error_fo = calculate_error(true_signal_test, y_pred_fo)
                nonzeros_fo = np.count_nonzero(model_fo.beta_)
                
                # 2. MIO 
                model_mio = MIOSolver(k=k, time_limit=500)
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
                        'Rho': rho,
                        'SNR': str(single_snr),
                        'Metric name': 'Prediction error'
                    })
                    df.append({
                        'Method': method,
                        'Metric': nonzeros,
                        'Rho': rho,
                        'SNR': str(single_snr),
                        'Metric name': 'Non-zero coefficients'
                    })
    df = pd.DataFrame(df)
    fig, ax = plt.subplots(nrows=2, ncols=len(rhos), figsize=(15, 10))
    
    for i, rho in enumerate(rhos):
        # Prediction error
        sns.barplot(
            data=df[(df['Metric name'] == 'Prediction error') & (df['Rho'] == rho)],
            x='SNR', y='Metric', hue='Method', ax=ax[0, i]
        )
        ax[0, i].set_title(f'ρ = {rho}')
        ax[0, i].set_ylabel('Prediction Error')
        
        # Non-zero coefficients
        sns.barplot(
            data=df[(df['Metric name'] == 'Non-zero coefficients') & (df['Rho'] == rho)],
            x='SNR', y='Metric', hue='Method', ax=ax[1, i]
        )
        ax[1, i].axhline(y=k, color='r', linestyle='--', label='True k')
        ax[1, i].set_ylabel('Non-zero Coefficients')
    
    plt.tight_layout()
    plt.savefig('experiment2_results.png')
    plt.show()


def main():
    print("Experiment run")
    experiment_1()
    #experiment_optimality_gap()
    #experiment_2()


if __name__ == '__main__':
    main()
