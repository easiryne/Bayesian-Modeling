
#Obj_1: Generating the csv file
import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

s1_val = s1_val = list(range(-4, 5, 2))
s2_val = list(range(-10, 11, 1))
s1_var = 0.01
s2_var = s2_var = [0.25, 0.5, 1.0, 2.0, 4.0]
data={"Trial": [], "S1_val": [], "S1_var": [], "S2_val": [], "S2_var": []}
cpt_trial = 1

for s1 in s1_val:
    for s2 in s2_val:
        for var in s2_var:
            for i in range(20):
                trial = (s1, s2, var)
                data["Trial"].append(cpt_trial)
                data["S1_val"].append(s1)
                data["S1_var"].append(s1_var)
                data["S2_val"].append(s2)
                data["S2_var"].append(var)
                cpt_trial += 1

df = pd.DataFrame(data)
df.to_csv("Political_project.csv", index=False)

##########################################
#Obj_2: Analyzing the csv data by optimizing the parameters

#Function to compute the Gaussian CDF for the given parameters mu0 and sigma0, for a trial i
def psi_i(var_0, mu_0, x_ref, var_ref, x_test, var_test):
#Computation of weights
    to_ref = 1/var_ref
    to_test = 1/var_test
    to_0= 1/var_0
    w_ref = to_ref/(to_ref + to_0)
    w_test = to_test/(to_test + to_0)

    z_num = (w_test*x_test-w_ref*x_ref)+(mu_0*(w_ref-w_test))
    z_deno = np.sqrt((w_ref**2)*var_ref + (w_test**2)*var_test)
    z_i = z_num/z_deno
    psi_i = norm.cdf(z_i)

    return psi_i

#Function to compute the negative log-likelihood for the given parameters mu_0 and sigma_0, for a trial i
def objective_function(params, x_ref, x_test, var_ref, var_test, y):
    sigma_0, mu_0 = params
    p_i = psi_i(sigma_0, mu_0, x_ref, var_ref, x_test, var_test)
    p_i = np.clip(p_i, 1e-12, 1 - 1e-12)
    l = -np.mean(y*np.log(p_i) + (1 - y)*np.log(1 - p_i))
    return l

#Function to optimize the parameters mu_0 and sigma_0 using the negative log-likelihood function
def optimize_parameters(file_path):
    df_results = pd.read_csv(file_path)

    x_ref = df_results['S1_val'].values
    x_test = df_results['S2_val'].values
    var_ref = df_results['S1_var'].values
    var_test = df_results['S2_var'].values
    y= 1-df_results['Decision (S1>S2)'].values

    res_opt =minimize(
        objective_function,
        x0=[1.0, 0.0],
        args=(x_ref, x_test, var_ref, var_test, y), 
        bounds=[(0.01, 64.0),(None, None)], 
        method='L-BFGS-B', 
        )

    return res_opt.x, res_opt.fun, res_opt.success, res_opt.message 
if __name__ == "__main__":
    print(optimize_parameters("G10_level_01_experiment_results.csv"))

