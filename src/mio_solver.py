import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from models import FOLeastSquares, FOLAD
from utils import generate_data, calculate_error
import time

np.random.seed(42)

def best_subset_miqp(X, y, k, beta0=None, time_limit=None, refine=True):
    """
    Mixed-integer quadratic solver for min 0.5||y - Xβ||²  s.t. ||β||₀ ≤ k
    Parameters
    ----------
    X, y : data
    k : sparsity level
    beta0: optional warm-start vector (from FOLeastSquares / FOLAD)
    refine: if True, ordinary LS is refit on the final support
    """
    n, p = X.shape
    Q = X.T @ X
    c = X.T @ y

    # big-M choice for safety
    if beta0 is None:
        M = np.linalg.norm(y) / (np.linalg.norm(X, axis=0) + 1e-12)
    else:
        # we inflate beta0 to stay feasible
        M = 1.2 * np.abs(beta0) + 1e-3

    m = gp.Model('best_subset')
    if time_limit: m.setParam('TimeLimit', time_limit)
    m.setParam('OutputFlag', 0)

    beta = m.addVars(p, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='beta')
    z = m.addVars(p, vtype=GRB.BINARY,    name='z')

    m.addConstrs( beta[j] <=  M[j]*z[j] for j in range(p) )
    m.addConstrs( beta[j] >= -M[j]*z[j] for j in range(p) )
    m.addConstr(  gp.quicksum(z[j] for j in range(p)) <= k )

    # Quadratic objective 0.5 βᵀQβ - cᵀβ  (drop constant yᵀy/2)
    obj = 0.5 * gp.QuadExpr()
    for i in range(p):
        obj += Q[i, i] * beta[i]*beta[i]
        for j in range(i+1, p):
            obj += Q[i, j] * beta[i]*beta[j]
    obj -= gp.quicksum(float(c[i])*beta[i] for i in range(p))
    m.setObjective(obj, GRB.MINIMIZE)

    # Revision 2: direct warm-start from FO object
    if beta0 is not None:
        for j in range(p):
            beta[j].start = float(beta0[j])
            z[j].start    = 1 if abs(beta0[j]) > 1e-12 else 0

    m.optimize()

    beta_star = np.array([beta[j].X for j in range(p)])
    z_star    = np.array([int(round(z[j].X)) for j in range(p)])

    # Optional refit to eliminate big-M bias
    if refine and z_star.sum() > 0:
        S = np.flatnonzero(z_star)
        beta_ls = np.zeros_like(beta_star)
        beta_ls[S] = np.linalg.pinv(X[:,S].T @ X[:,S]) @ X[:,S].T @ y
        beta_star = beta_ls

    return beta_star, z_star


def count_effective_nonzeros(beta, threshold=1e-6):
    return np.sum(np.abs(beta) > threshold)


n, p, k_true = 500, 100, 10
X, y, beta_true, train_idx, test_idx = generate_data(n, p, k=k_true, method='1', num=1)
y = np.asarray(y)
X_train, y_train = X[train_idx], y[train_idx]
X_test , y_test = X[test_idx ], y[test_idx]

k_model = k_true

#1) FO
tic = time.perf_counter()
fo = FOLeastSquares(k=k_model, method='interpolated', refine_least_squares=False).fit(X_train, y_train)
toc = time.perf_counter()
beta_fo = fo.beta_


# print("FO β:", beta_fo)
# print("Non-zeros (raw):", np.count_nonzero(beta_fo))
# print("Non-zeros (thresholded):", count_effective_nonzeros(beta_fo, threshold=1e-6))

# nonzero_vals = beta_fo[np.abs(beta_fo) > 1e-8]
# print("Nonzero values:", nonzero_vals)
# print("Sorted abs values:", np.sort(np.abs(nonzero_vals)))

err_fo = calculate_error(y_test, fo.predict(X_test))
t_fo = toc - tic

#2) MIQP cold 
tic = time.perf_counter()
beta_cold, _ = best_subset_miqp(X_train, y_train, k_model, beta0=None, time_limit=120)
toc = time.perf_counter()
err_cold = calculate_error(y_test, X_test @ beta_cold)
t_cold = toc - tic

#3) MIQP   warm (FO incumbent)
tic = time.perf_counter()
beta_warm, _ = best_subset_miqp(X_train, y_train, k_model, beta0=beta_fo, time_limit=120)
toc = time.perf_counter()
err_warm = calculate_error(y_test, X_test @ beta_warm)
t_warm = toc - tic



results = pd.DataFrame({
    "method"     : ["FO heuristic", "MIQP cold", "MIQP warm"],
    "test error" : [err_fo, err_cold, err_warm],
    "run-time s" : [t_fo , t_cold , t_warm ],
    "non-zeros"  : [np.count_nonzero(beta_fo),
                    np.count_nonzero(beta_cold),
                    np.count_nonzero(beta_warm)]
})

print(results.to_string(index=False))
# print(evaluate_model(X, y, beta_fo))
# print(evaluate_model(X, y, beta_star))
