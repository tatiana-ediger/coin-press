import torch
import algos
import numpy as np
import math
import pandas as pd

def coinpress_linreg(x, y, beta, c, r, total_budget):
    """
    input:
    output:
    beta_hat = mean_est * cov_est
    where mean_est comes from multivariate mean iterative
    and cov_est comes from
    """
    n = len(x)

    z = []
    for i in range(n):
        z.append(x[i] * y[i])
    z = np.array(z)

    # TODO: private beta_norm_sqr !!
    beta_norm_sqr = np.norm(beta) ** 2

    z = z / np.sqrt(2 * beta_norm_sqr + 1)
    rho = [(1.0 / 4.0) * total_budget, (3.0 / 4.0) * total_budget]
    mean_est = algos.multivariate_mean_iterative(z, c, r, 2, rho) * np.sqrt(2 * beta_norm_sqr + 1)