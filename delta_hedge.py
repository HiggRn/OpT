import argparse

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


def run_once(model, option, real_S, mu, sigma, rho, T, N, M, r):
    real_S_cpu = real_S.detach().clone().cpu().numpy()

    # then generate the dataset
    dataset = simulate_option(option, real_S_cpu, mu, sigma, rho, T, N, M, r)

    # give an initial guess
    V_0, Delta_0 = model(dataset[0][0])  # V_0 & Delta_0 should be (1) & (D)

    if V_0 < torch.relu(torch.tensor(option(real_S_cpu[:, 0]))):
        return -torch.inf

    Delta_prev = Delta_0

    # delta hedge
    X_0 = V_0 - torch.dot(Delta_0, real_S[:, 0])  # (1)
    X = X_0.detach().clone()

    for i, data in enumerate(dataset[1:]):
        X *= 1 + r
        S_cpu = real_S_cpu[:, i + 1]  # stock price, (D)
        S = real_S[:, i + 1]
        # with torch.no_grad():  # same model-same prediction, same idea, no longer need to calculate gradient again
        V, Delta = model(data[0])  # V & Delta should be (1) & (D)
        # option value if exercise
        V_exercise = torch.relu(torch.tensor(option(S_cpu)).float())
        if V < V_exercise:  # if exercise
            X -= V_exercise
            X += torch.dot(Delta_prev, S)
            break
        X += torch.dot(Delta - Delta_prev, S)
        Delta_prev = Delta

    if i + 1 == N - 1:
        S_cpu = real_S_cpu[:, -1]
        S = real_S[:, -1]
        X -= torch.relu(torch.tensor(option(S_cpu)).float())
        X += torch.dot(Delta, S)

    # calculate return
    alpha = X / X_0 - 1
    return alpha


def run_dual(model1, model2, option, real_S, mu, sigma, rho, T, N, M, r):
    # Suppose we use model1 as the seller, model2 as the buyer
    real_S_cpu = real_S.detach().clone().cpu().numpy()

    # then generate the dataset
    dataset = simulate_option(option, real_S_cpu, mu, sigma, rho, T, N, M, r)

    # give an initial guess
    V_0_1, Delta_0_1 = model1(dataset[0][0])  # V_0 & Delta_0 should be (1) & (D)
    V_0_2, Delta_0_2 = model2(dataset[0][0])

    Delta_prev_1 = Delta_0_1
    Delta_prev_2 = Delta_0_2

    # delta hedge
    # suppose in the end V_0 is the average
    V_0 = 0.5 * (V_0_1 + V_0_2)

    if V_0 < torch.relu(torch.tensor(option(real_S_cpu[:, 0]))):
        return -torch.inf, torch.inf

    X_0_1 = V_0 - torch.dot(Delta_0_1, real_S[:, 0])  # (1)
    X_0_2 = -V_0 + torch.dot(Delta_0_2, real_S[:, 0])  # (1)
    X_1 = X_0_1.detach().clone()
    X_2 = X_0_2.detach().clone()
    print(f"Start--Seller:{X_1:6f}, Buyer:{X_2:6f}")
    print(f"Price:{V_0}")

    for i, data in enumerate(dataset[1:]):
        X_1 *= 1 + r
        X_2 *= 1 + r
        S_cpu = real_S_cpu[:, i + 1]  # stock price, (D)
        S = real_S[:, i + 1]
        # with torch.no_grad():  # same model-same prediction, same idea, no longer need to calculate gradient again
        _, Delta_1 = model1(data[0])  # V & Delta should be (1) & (D)
        V, Delta_2 = model2(data[0])  # V & Delta should be (1) & (D)
        # option value if exercise
        V_exercise = torch.relu(torch.tensor(option(S_cpu)).float())
        if V < V_exercise:  # if exercise
            X_1 -= V_exercise
            X_1 += torch.dot(Delta_prev_1, S)
            X_2 += V_exercise
            X_2 -= torch.dot(Delta_prev_2, S)
            print(f"Exercised at t={i+1}")
            break
        # update delta hedge
        X_1 += torch.dot(Delta_1 - Delta_prev_1, S)
        X_2 -= torch.dot(Delta_2 - Delta_prev_2, S)

        Delta_prev_1 = Delta_1
        Delta_prev_2 = Delta_2

    if i + 1 == N - 1:
        S_cpu = real_S_cpu[:, -1]
        S = real_S[:, -1]
        X_1 -= torch.relu(torch.tensor(option(S_cpu)).float())
        X_2 += torch.relu(torch.tensor(option(S_cpu)).float())
        X_1 += torch.dot(Delta_1, S)
        X_2 -= torch.dot(Delta_2, S)

    # calculate return
    print(f"End--Seller:{X_1:6f}, Buyer:{X_2:6f}")
    alpha_1 = (X_1 / X_0_1 - 1) * torch.sign(X_0_1)
    alpha_2 = (X_2 / X_0_2 - 1) * torch.sign(X_0_2)
    return alpha_1, alpha_2


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

    model = OptionTransformer(d_model, n_layers, n_head, D, M, dropout).to(
        config["device"]
    )

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for n in range(num_run):
        T = 1.0  # the absolute value for `T` is really unnecessary for us

        S_0 = np.random.lognormal(0, 5, D)  # random generate S_0
        mu = np.random.normal(r, 1, D)  # random generate mu
        sigma = np.random.lognormal(0, 1, D)  # random generate sigma
        eigs = np.random.lognormal(0, 1, D)
        eigs /= np.sum(eigs)
        eigs *= D
        rho = random_correlation.rvs(eigs, rng)

        # first simulate a real stock trajectory
        real_S = simulate(S_0, mu, sigma, rho, len(S_0), N, T)  # (D,N)
        real_S = torch.tensor(real_S).to(config["device"])

        # run once till the maturity, give back the premium
        alpha = run_once(model, option, real_S, mu, sigma, rho, T, N, M, r)
        print(f"Trial {n}--Premium:{alpha:6f}")

        # compute loss
        criterion = nn.MSELoss().to(config["device"])
        loss = criterion(alpha, torch.tensor(0.0))

        # optimize model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad
def eval(model, option, r, config, num_run):
    set_seed()

    rng = np.random.default_rng()

    # Monte Carlo config
    D = config["n_assets"]
    M = config["n_mcmc"]
    N = config["n_timesteps"]

    for n in range(num_run):
        T = 1.0  # the absolute value for `T` is really unnecessary for us

        S_0 = np.random.lognormal(0, 1, D)  # random generate S_0
        mu = np.random.normal(r, 1, D)  # random generate mu
        sigma = np.random.lognormal(0, 1, D)  # random generate sigma
        eigs = np.random.lognormal(0, 1, D)
        eigs /= np.sum(eigs)
        eigs *= D
        rho = random_correlation.rvs(eigs, rng)

        # first simulate a real stock trajectory
        real_S = simulate(S_0, mu, sigma, rho, len(S_0), N, T)  # (D,N)
        real_S = torch.tensor(real_S).to(config["device"])

        # run once till the maturity, give back the premium
        alpha = run_once(model, option, real_S, mu, sigma, rho, T, N, M, r)
        print(f"Trial {n}--Premium:{alpha:6f}")


def get_args():
    """Main function."""

    parser = argparse.ArgumentParser()

    parser.add_argument("-n_assets", type=int, default=5)
    parser.add_argument("-n_mcmc", type=int, default=1000)
    parser.add_argument("-n_timesteps", type=int, default=1000)
    parser.add_argument("-d_model", type=int, default=20)
    parser.add_argument("-n_layers", type=int, default=5)
    parser.add_argument("-n_head", type=int, default=10)
    parser.add_argument("-dropout", type=float, default=0.1)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-num_run", type=int, default=10)
    parser.add_argument("-device", type=str, default="cuda")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    config = args.__dict__
    train(lambda S: np.max(S) - 2, 0.05, config, config["num_run"])
