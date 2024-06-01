import numpy as np
import torch
from torch.utils.data import Dataset

from monte_carlo.monte_carlo import simulate


class OptionDataset(Dataset):
    def __init__(self, data, option, M) -> None:
        """
        :param data: list[dict{"S_0":list[float]"mu":list[float],"sigma":list[float],"rho":list[list[float]],"r":float,"T":float,"option_prices":list[float]}]
        :param option: Callable[ndarray[float]],float], function to calculate the option payoff, given the prices of assets
        :param M: int, number of Monte-Carlo simulations
        """
        super().__init__()

        self.data = []
        for i, sample in enumerate(data):
            S_0 = np.array(sample["S_0"])
            mu = np.array(sample["mu"])
            sigma = np.array(sample["sigma"])
            rho = np.array(sample["rho"])
            option_prices = np.array(sample["option_prices"])
            D = len(S_0)
            N = len(option_prices)
            T = sample["T"]

            # Simulation
            simulated_data = []
            for _ in range(M):
                simulated = simulate(S_0, mu, sigma, rho, D, N, T)
                option_payoff = np.apply_along_axis(option, 0, simulated)
                simulated = np.vstack((simulated, option_payoff))
                # We predict from the fture to the present
                # So the simulation has to be flipped
                simulated = np.flip(simulated, 1)
                simulated_data.append(simulated)
            simulated_data = torch.tensor(np.array(simulated_data)).float()
            self.data.append((simulated_data, sample["r"], option_prices))
            if (i + 1) % 100 == 0:
                print(f"Sample {i+1} is added.")
        print("Dataset built.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def simulate_option(option, S_real, mu, sigma, rho, T, N, M, r) -> OptionDataset:
    data = []
    for n in range(N):
        S = S_real[:, n]
        sample = {
            "S_0": S,
            "mu": mu,
            "sigma": sigma,
            "rho": rho,
            "option_prices": range(1, N + 1 - n),
            "r": r,
            "T": T,
        }
        data.append(sample)
    return OptionDataset(data, option, M)


if __name__ == "__main__":
    option = lambda S: max(0, S[0] - S[1])
    T = 1.0
    D = 2
    N = 10
    M = 15
    S_real = np.array(np.random.rand(D, N)) * 10
    mu = np.array([0.5, 1.0])
    sigma = np.array([1.0, 4.0])
    rho = np.array([[1.0, 0.0], [0.0, 1.0]])
    r = 0.05
    test_dataset = simulate_option(option, S_real, mu, sigma, rho, T, N, M, r)
