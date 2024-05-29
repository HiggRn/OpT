import numpy as np
import torch
from scipy.stats import random_correlation
from torch import nn, optim

from data.dataset import simulate_option
from monte_carlo.monte_carlo import simulate
from transformer.model_percentiles import OptionTransformer


def set_seed(seed=2024):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_once(model, option, S_0, mu, sigma, rho, T, N, M, r):
    # first simulate a real stock trajectory
    real_S = torch.tensor(simulate(S_0, mu, sigma, rho, len(S_0), N, T))  # (D,N)

    # then generate the dataset
    dataset = simulate_option(option, S_0, mu, sigma, rho, T, N, M, r)

    # give an initial guess
    V_0, Delta_0 = model(dataset[0])  # V_0 & Delta_0 should be (1) & (D)
    Delta_prev = Delta_0

    # delta hedge
    X_0 = V_0 - torch.dot(Delta_0, S_0)  # (1)
    X = X_0

    for i, data in enumerate(dataset[1:]):
        X *= 1 + r
        S = real_S[:, i + 1]  # stock price, (D)
        with torch.no_grad():  # same model-same prediction, same idea, no longer need to calculate gradient again
            V, Delta = model(data)  # V & Delta should be (1) & (D)
        # option value if exercise
        V_exercise = torch.relu(option(S))
        if V < V_exercise:  # if exercise
            X -= V_exercise
            X += torch.dot(Delta_prev, S)
            break
        X += torch.dot(Delta - Delta_prev, S)
        Delta_prev = Delta

    if i + 1 == N - 1:
        S = real_S[:, -1]
        X -= torch.relu(option(S))
        X += torch.dot(Delta, S)

    # calculate premium
    alpha = torch.pow(X / X_0, 1 / (N - 1)) - 1 - r
    return alpha


def train(option, r, config, num_run):
    set_seed()

    rng = np.random.default_rng()

    # Monte Carlo config
    D = config["n_assets"]
    M = config["n_mcmc"]
    N = config["n_timesteps"]

    # Model config
    d_model = config["d_model"]
    n_layers = config["n_layers"]
    n_head = config["n_head"]
    dropout = config["dropout"]

    model = OptionTransformer(d_model, n_layers, n_head, D, M, dropout)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for n in range(num_run):
        T = 1.0  # the absolute value for `T` is really unnecessary for us

        S_0 = np.random.lognormal(0, 1, D)  # random generate S_0
        mu = np.random.normal(r, 1, D)  # random generate mu
        sigma = np.random.lognormal(0, 1, D)  # random generate sigma
        rho = random_correlation(np.random.lognormal(0, 1, D), rng)

        # run once till the maturity, give back the premium
        alpha = run_once(model, option, S_0, mu, sigma, rho, T, N, M, r)
        print(f"Trial {n}--Premium:{alpha:6f}")

        # compute loss
        criterion = nn.MSELoss()
        loss = criterion(alpha, 0)

        # optimize model
        optimizer.zero_grad()
        loss.backward()
