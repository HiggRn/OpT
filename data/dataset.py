# Construct the dataset for training
import numpy as np
from torch.utils.data import DataLoader, Dataset

from monte_carlo.monte_carlo import simulate


class OptionDataset(Dataset):
    def __init__(self, data, option, M) -> None:
        """
        :param data: list[dict{"S_0":list[float]"mu":list[float],"sigma":list[float],"rho":list[list[float]],"r":float,"option_prices":list[float]}]
        :param options: Callable[ndarray[float]],float], function to calculate the option payoff, given the prices of assets
        :param M: int, number of Monte-Carlo simulations
        """
        super().__init__()

        self.data = []
        for sample in data:
            S_0 = np.array(sample["S_0"])
            mu = np.array(sample["mu"])
            sigma = np.array(sample["sigma"])
            rho = np.array(sample["rho"])
            option_prices = np.array(sample["option_prices"])
            D = len(S_0)
            N = len(option_prices)

            # Simulation
            simulated_data = []
            for _ in range(M):
                simulated = simulate(S_0, mu, sigma, rho, D, N)
                option_payoff = np.apply_along_axis(option, 0, simulated)
                simulated = np.vstack((simulated, option_payoff))
                simulated_data.append(simulated)
            simulated_data = np.array(simulated_data)
            self.data.append((simulated_data, sample["r"], option_prices))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":
    data = {
        "S_0": [2.0, 4.0],
        "mu": [0.5, 1.0],
        "sigma": [1.0, 4.0],
        "rho": [[1.0, 0.0], [0.0, 1.0]],
        "option_prices": range(1, 11),
        "r": 0.05,
    }
    test_dataset = OptionDataset([data], lambda S: max(0, S[0] - S[1]), 15)
    print(test_dataset.data)
