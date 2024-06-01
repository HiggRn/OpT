# -*- coding: utf-8 -*-
# Simulate asset price movement using Monte Carlo
import numpy as np


# TODO或许模拟可以放到GPU上？
# Suppose every parameter is scaled to Δt=1
def simulate(S_0, mu, sigma, rho, D, N, T):
    # S_0: initial price (D,1)
    # mu: expected return (D,1)
    # sigma: volatility (D,1)
    # rho: correlation matrix (D,D)
    # D: number of assets (1)
    # N: number of timesteps (1)
    # T: maturity(in years) (1)
    # Output: simulated price trajectories of size (D,N)

    # 首先对相关系数矩阵rho做Cholesky分解
    L = np.linalg.cholesky(rho)  # (D,D)

    # 生成每一步的增量
    delta_t = T / N
    Delta_W = np.sqrt(delta_t) * L @ np.random.normal(0.0, 1.0, size=(D, N))  # (D,N)

    # 价格模拟
    S_simlated = np.zeros((D, N))
    S_simlated[:, 0] = S_0
    for n in range(1, N):
        S_simlated[:, n] = (1 + mu * delta_t) * S_simlated[:, n - 1]
        S_simlated[:, n] += sigma * S_simlated[:, n - 1] * Delta_W[:, n - 1]

    return S_simlated


if __name__ == "__main__":
    S_0 = np.array([2.0, 4.0])
    mu = np.array([0.5, 1.0])
    sigma = np.array([1.0, 4.0])
    rho = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = 2
    N = 10
    T = 1.0
    test_data = simulate(S_0, mu, sigma, rho, D, N, T)
    print(test_data)
