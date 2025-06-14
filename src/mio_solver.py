import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import time

np.random.seed(42)


class MIOSolver:
    def __init__(self, k, time_limit=None, warm_start_beta=None):
        """
        MIO solver for best subset selection in n>p scenario
        :param k: Number of features to select
        :param time_limit: Time limit in seconds
        :param warm_start_beta: Warm start solution from first-order method
        """
        self.k = k
        self.time_limit = time_limit
        self.warm_start_beta = warm_start_beta
        self.beta_ = None
        self.support_ = None
        self.obj_value_ = None
        self.gap_ = None
        self.time_ = None
        self.gap_progress = []

    def fit(self, X, y):
        n, p = X.shape
        model = gp.Model("BestSubset")
        
        #solver parameters
        if self.time_limit:
            model.Params.TimeLimit = self.time_limit
        model.Params.OutputFlag = 0
        model.Params.MIPGapAbs = 1e-8
        model.Params.MIPGap = 1e-8
        # variables
        beta = model.addVars(p, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta")
        z = model.addVars(p, vtype=GRB.BINARY, name="z")
        
        # obj func: 1/2 * ||y - Xβ||^2
        residuals = gp.quicksum((y[i] - gp.quicksum(X[i, j] * beta[j] for j in range(p)))**2 
                                for i in range(n))
        model.setObjective(0.5 * residuals, GRB.MINIMIZE)
        
        model.addConstr(gp.quicksum(z[j] for j in range(p)) <= self.k, "cardinality")
        
        # M = 10 * np.linalg.norm(y)
        if self.warm_start_beta is not None:
            abs_beta = np.abs(self.warm_start_beta)
            M = 2 * np.max(abs_beta[abs_beta > 1e-5])
        else:
            try:
                ols_beta = np.linalg.lstsq(X, y, rcond=None)[0]
                M = 2 * np.max(np.abs(ols_beta))
            except:
                M = 2 * np.linalg.norm(y) / np.sqrt(n)
        
        for j in range(p):
            model.addConstr(beta[j] <= M * z[j], f"bigM_upper_{j}")
            model.addConstr(beta[j] >= -M * z[j], f"bigM_lower_{j}")
        

        if self.warm_start_beta is not None:
            #starting values
            for j in range(p):
                beta[j].Start = self.warm_start_beta[j]
                z[j].Start = 1 if np.abs(self.warm_start_beta[j]) > 1e-5 else 0

            model.NumStart = 1
            model.params.StartNumber = 0
            model.update()
        
        start_time = time.time()
        
        def callback(model, where):
            if where == gp.GRB.Callback.MIP:
                runtime = model.cbGet(gp.GRB.Callback.RUNTIME)
                obj_bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
                obj_best = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                if abs(obj_bound) > 1e-10:
                    gap = abs(obj_best - obj_bound) / abs(obj_bound)
                    self.gap_progress.append((runtime, gap))
        
        model.optimize(callback)
        self.time_ = time.time() - start_time
        
        # saving the results
        if model.SolCount > 0:
            self.beta_ = np.array([beta[j].X for j in range(p)])
            self.support_ = np.array([z[j].X > 0.5 for j in range(p)])
            self.obj_value_ = model.ObjVal
            self.gap_ = model.MIPGap
        else:
            # Fzallback to warm start if available
            if self.warm_start_beta is not None:
                self.beta_ = self.warm_start_beta
                self.support_ = np.abs(self.warm_start_beta) > 1e-5
                self.obj_value_ = 0.5 * np.sum((y - X @ self.warm_start_beta)**2)
                self.gap_ = np.nan
            else:
                #zero solution
                self.beta_ = np.zeros(p)
                self.support_ = np.zeros(p, dtype=bool)
                self.obj_value_ = 0.5 * np.sum(y**2)
                self.gap_ = np.nan
        
        return self

    def predict(self, X):
        return X @ self.beta_