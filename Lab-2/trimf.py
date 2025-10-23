import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import skfuzzy as fuzz
from common_fuzzy import plot_mfs, print_value_table, build_rules, evaluate_model, plot_graph


def calculate_trimf(x, a, b, c):
    if a <= x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)
    else:
        return 0


def functionCompare(value, means, diff):
    best_func_value = -float("inf")
    best_index = -1
    for index, mean in enumerate(means):
        ff = calculate_trimf(value, mean - diff, mean, mean + diff)
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

    mx = [fuzz.trimf(x_values, [x_means[i] - 3, x_means[i], x_means[i] + 3]) for i in range(6)]
    my = [fuzz.trimf(np.linspace(min(y_values), max(y_values), 100),
                     [y_means[i] - 3, y_means[i], y_means[i] + 3]) for i in range(6)]
    mf = [fuzz.trimf(np.linspace(min(z_values), max(z_values), 100),
                     [z_means[i] - 4, z_means[i], z_means[i] + 4]) for i in range(9)]

    plot_graph(x_values, y_values, "Y-function")
    plot_graph(x_values, z_values, "Z-function")

    plot_mfs(x_values, mx, "X Trimf")
    plot_mfs(np.linspace(min(y_values), max(y_values), 100), my, "Y Trimf")
    plot_mfs(np.linspace(min(z_values), max(z_values), 100), mf, "Z Trimf")

    print_value_table(x_means, y_means)
    rules = build_rules(x_means, y_means, z_means, functionCompare, 4)
    evaluate_model(x_values, x_means, y_means, z_means, rules, functionCompare)


if __name__ == "__main__":
    main()
