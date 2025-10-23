import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import skfuzzy as fuzz
from common_fuzzy import plot_mfs, print_value_table, build_rules, evaluate_model

def functionCompare(value, means, sigma):
    best_func_value = -float("inf")
    best_index = -1
    for index, mean in enumerate(means):
        ff = np.exp(-((value - mean) ** 2) / (2 * sigma ** 2))
        if ff > best_func_value:
            best_func_value = ff
            best_index = index
    return best_index


def main():
    x_values = np.linspace(0, 20, 100)
    y_values = x_values * np.sin(x_values) * np.cos(x_values)
    z_values = np.cos(np.sin(y_values)) * np.sin(x_values)

    x_means = np.linspace(min(x_values), max(x_values), 6)
    y_means = np.linspace(min(y_values), max(y_values), 6)
    z_means = np.linspace(min(z_values), max(z_values), 9)

    x_sigma = (max(x_values) - min(x_values)) / 6 / 2
    y_sigma = (max(y_values) - min(y_values)) / 6 / 2
    z_sigma = (max(z_values) - min(z_values)) / 9 / 2

    x_mf_gaussian = [fuzz.gaussmf(x_values, x_means[i], x_sigma) for i in range(6)]
    y_range = np.linspace(min(y_values), max(y_values), 100)
    y_mf_gaussian = [fuzz.gaussmf(y_range, y_means[i], y_sigma) for i in range(6)]
    z_range = np.linspace(min(z_values), max(z_values), 100)
    z_mf_gaussian = [fuzz.gaussmf(z_range, z_means[i], z_sigma) for i in range(9)]

    plot_mfs(x_values, x_mf_gaussian, "X Gaussian MF")
    plot_mfs(y_range, y_mf_gaussian, "Y Gaussian MF")
    plot_mfs(z_range, z_mf_gaussian, "Z Gaussian MF")

    print_value_table(x_means, y_means)
    rules = build_rules(x_means, y_means, z_means, functionCompare, z_sigma)
    evaluate_model(x_values, x_means, y_means, z_means, rules, functionCompare)


if __name__ == "__main__":
    main()
