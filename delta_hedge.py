import argparse
import pickle
import random

import numpy as np
import torch
from scipy.stats import random_correlation
from torch import nn, optim
from tqdm import tqdm

from data.dataset import simulate_option
from monte_carlo.monte_carlo import simulate
from transformer.model_percentiles import OptionTransformer


def set_seed(seed=2024):
    random.seed(seed)
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
        X -= torch.dot(Delta - Delta_prev, S)
        Delta_prev = Delta

    if i + 1 == N - 1:
        S_cpu = real_S_cpu[:, -1]
        S = real_S[:, -1]
        X -= torch.relu(torch.tensor(option(S_cpu)).float())
        X += torch.dot(Delta, S)

    return X


def run_dual(model1, model2, option, dataset, r, device):
    # data: list of length 'N', each element is a simulated data
    # Suppose we use model1 as the seller, model2 as the buyer

    N = len(dataset)

    # give an initial guess
    S_0 = dataset[0].transpose(0, 2).to(device).float()
    S_0_cpu = S_0[-1, :-1, 0].detach().cpu().numpy()
    V_0_1, Delta_0_1 = model1(S_0, device=device)  # V_0 & Delta_0 should be (1) & (D)
    V_0_2, Delta_0_2 = model2(S_0, device=device)

    Delta_prev_1 = Delta_0_1
    Delta_prev_2 = Delta_0_2

    # delta hedge
    # suppose in the end V_0 is the average
    V_0 = 0.5 * (V_0_1 + V_0_2)
    # V_0 += torch.relu(torch.tensor(option(S_0_cpu))).to(device)

    # if V_0 < torch.relu(torch.tensor(option(data[0][0, :-1, -1]))).to(device).float():
    #     return -torch.inf, torch.inf
    V_0_exercise = torch.relu(torch.tensor(option(S_0_cpu))).to(device).float()
    X_1 = torch.max(torch.stack((V_0, V_0_exercise))) - torch.dot(
        Delta_0_1, S_0[-1, :-1, 0]
    )  # (1)
    X_2 = -torch.max(torch.stack((V_0, V_0_exercise))) + torch.dot(
        Delta_0_2, S_0[-1, :-1, 0]
    )  # (1)
    with open(config["log"], "a") as log:
        print(f"Price:{V_0}", file=log)

    for i, data in enumerate(dataset[1:]):
        X_1 *= 1 + r
        X_2 *= 1 + r

        S = data.transpose(0, 2).to(device).float()
        # with torch.no_grad():  # same model-same prediction, same idea, no longer need to calculate gradient again
        _, Delta_1 = model1(S, device=device)  # V & Delta should be (1) & (D)
        V, Delta_2 = model2(S, device=device)  # V & Delta should be (1) & (D)
        # option value if exercise
        S_cpu = data[0, :-1, -1].detach().cpu().numpy()
        V_exercise = torch.relu(torch.tensor(option(S_cpu))).to(device).float()
        if V < V_exercise:  # if exercise
            X_1 -= V_exercise
            X_1 += torch.dot(Delta_prev_1, S[-1, :-1, 0])
            X_2 += V_exercise
            X_2 -= torch.dot(Delta_prev_2, S[-1, :-1, 0])
            with open(config["log"], "a") as log:
                print(f"Exercised at t={i+1}", file=log)
            break
        # update delta hedge
        X_1 -= torch.dot(Delta_1 - Delta_prev_1, S[-1, :-1, 0])
        X_2 += torch.dot(Delta_2 - Delta_prev_2, S[-1, :-1, 0])

        Delta_prev_1 = Delta_1
        Delta_prev_2 = Delta_2

    if i + 1 == N - 1:
        S = data[N - 1][:-1, :].squeeze().to(device)
        S_cpu = S.detach().cpu().numpy()
        X_1 -= torch.relu(torch.tensor(option(S_cpu)).float()).to(device)
        X_2 += torch.relu(torch.tensor(option(S_cpu)).float()).to(device)
        X_1 += torch.dot(Delta_1, S)
        X_2 -= torch.dot(Delta_2, S)

    return X_1, X_2


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

    model1 = OptionTransformer(d_model, n_layers, n_head, D, M, dropout).to(
        config["device"]
    )
    model2 = OptionTransformer(d_model, n_layers, n_head, D, M, dropout).to(
        config["device"]
    )

    optimizer1 = optim.Adam(model1.parameters(), lr=config["lr"])
    optimizer2 = optim.Adam(model1.parameters(), lr=config["lr"])

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
        alpha_1, alpha_2 = run_dual(
            model1, model2, option, real_S, mu, sigma, rho, T, N, M, r, config["device"]
        )
        print(f"Trial {n}--Seller Return:{alpha_1:6f}, Buyer Return:{alpha_2:6f}")

        # compute loss
        criterion = nn.MSELoss().to(config["device"])
        alpha = torch.min(torch.stack((alpha_1, alpha_2)))
        loss = criterion(torch.sigmoid(alpha), torch.tensor(1.0))

        # optimize model
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()


""" @torch.no_grad
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
        print(f"Trial {n}--Premium:{alpha:6f}") """


def train_dual(option, dataset, r, config):
    set_seed()
    random.shuffle(dataset)

    # split data
    data_len = len(dataset)
    train_len = int(0.8 * data_len) - 1
    dataset_train = dataset[:train_len]
    dataset_eval = dataset[train_len:]

    # Monte Carlo config
    D = config["n_assets"]
    M = config["n_mcmc"]
    N = config["n_timesteps"]

    # Model config
    d_model = config["d_model"]
    n_layers = config["n_layers"]
    n_head = config["n_head"]
    dropout = config["dropout"]

    model1 = OptionTransformer(d_model, n_layers, n_head, D, M, dropout).to(
        config["device"]
    )
    model2 = OptionTransformer(d_model, n_layers, n_head, D, M, dropout).to(
        config["device"]
    )

    optimizer1 = optim.Adam(model1.parameters(), lr=config["lr"])
    optimizer2 = optim.Adam(model2.parameters(), lr=config["lr"])

    for n in range(config["num_run"]):
        print(f"\nEpoch {n+1}:\n")
        with open(config["log"], "a") as log:
            print(f"\nEpoch {n+1}:\n", file=log)
        random.shuffle(dataset_train)

        for i, data in tqdm(enumerate(dataset_train)):
            # run once till the maturity, give back the premium
            with open(config["log"], "a") as log:
                print(f"Trial {i}:", file=log)
            X_1, X_2 = run_dual(model1, model2, option, data[0], r, config["device"])
            with open(config["log"], "a") as log:
                print(f"Seller income:{X_1:6f}, Buyer income:{X_2:6f}\n", file=log)

            # compute loss
            criterion = nn.MSELoss().to(config["device"])
            loss = criterion(
                torch.sigmoid(torch.stack((X_1, X_2))),
                torch.tensor([1.0] * 2),
            )

            # optimize model
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

        eval_dual(model1, model2, option, dataset_eval, r, config)
    return model1, model2


@torch.no_grad()
def eval_dual(model1, model2, option, dataset, r, config):
    print("Testing:")
    with open(config["log"], "a") as log:
        print("Testing:", file=log)
    random.shuffle(dataset)
    X_1_list = []
    X_2_list = []
    for data in tqdm(dataset):
        print("\n")
        X_1, X_2 = run_dual(model1, model2, option, data[0], r, config["device"])
        X_1_list.append(X_1.detach().cpu().numpy())
        X_2_list.append(X_2.detach().cpu().numpy())
    X_1_avg = np.average(X_1_list)
    X_2_avg = np.average(X_2_list)
    with open(config["log"], "a") as log:
        print(
            f"Seller average income:{X_1_avg:6f}, Buyer average income:{X_2_avg:6f}\n",
            file=log,
        )


def get_args():
    """Main function."""

    parser = argparse.ArgumentParser()

    parser.add_argument("-data", type=str, required=True)
    parser.add_argument("-n_assets", type=int, default=5)
    parser.add_argument("-n_mcmc", type=int, default=200)
    parser.add_argument("-n_timesteps", type=int, default=50)
    parser.add_argument("-d_model", type=int, default=20)
    parser.add_argument("-n_layers", type=int, default=5)
    parser.add_argument("-n_head", type=int, default=10)
    parser.add_argument("-dropout", type=float, default=0.1)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-num_run", type=int, default=5)
    parser.add_argument("-log", type=str, default="log.log")
    parser.add_argument(
        "-device",
        type=str,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    config = args.__dict__
    r = 0.05 / 252
    torch.set_default_device(config["device"])
    with open(config["data"], "rb") as f:
        data = pickle.load(f)
    model1, model2 = train_dual(
        lambda S: np.max(np.array([np.max(S) - 1100, 0])), data, r, config
    )

    torch.save(model1.state_dict(), "seller.pth")
    torch.save(model2.state_dict(), "buyer.pth")
