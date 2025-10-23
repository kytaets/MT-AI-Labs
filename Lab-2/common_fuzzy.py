import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, mean_absolute_error

def plot_graph(a, b, title):
    plt.plot(a, b)
    plt.title(title)
    plt.show()
def plot_mfs(x, mfs, title):
    for mf in mfs:
        plt.plot(x, mf)
    plt.title(title)
    plt.show()


def print_value_table(x_means, y_means):
    print("Table of values")
    table = [["y\\x"] + [str(round(x, 2)) for x in x_means]]
    for y_value in y_means:
        row = [round(y_value, 2)]
        for x in x_means:
            z = np.cos(np.sin(y_value)) * np.sin(x)
            row.append(round(z, 2))
        table.append(row)
    print(tabulate(table, tablefmt="grid"))


def build_rules(x_means, y_means, z_means, compare_func, z_sigma_or_diff):
    print("\nTable with function names")
    table = [["y\\x"] + ["mx" + str(i + 1) for i in range(len(x_means))]]
    rules = {}
    for i, y_val in enumerate(y_means):
        row = ["my" + str(i + 1)]
        for j, x_val in enumerate(x_means):
            z = np.cos(np.sin(y_val)) * np.sin(x_val)
            best_func = compare_func(z, z_means, z_sigma_or_diff)
            row.append("mf" + str(best_func + 1))
            rules[(j, i)] = best_func
        table.append(row)
    print(tabulate(table, tablefmt="grid"))

    print("\nRules:")
    for rule in rules.keys():
        print(f"if (x is mx{rule[0] + 1} and y is my{rule[1] + 1}) then (z is mf{rules[rule] + 1})")
    return rules


def evaluate_model(x_values, x_means, y_means, z_means, rules, compare_func):
    z_values = np.cos(np.sin(x_values * np.sin(x_values) * np.cos(x_values))) * np.sin(x_values)
    z_output = []
    for x in x_values:
        best_x_func = compare_func(x, x_means, 3)
        best_y_func = compare_func(x * np.sin(x) * np.cos(x), y_means, 0.5)
        best_z_func = rules[(best_x_func, best_y_func)]
        z_output.append(z_means[best_z_func])

    plt.plot(x_values, z_output, label="Model")
    plt.plot(x_values, z_values, label="True")
    plt.title("True & modelled functions")
    plt.legend()
    plt.show()

    mse = mean_squared_error(z_values, z_output)
    mae = mean_absolute_error(z_values, z_output)
    print(f"\nMean Squared Error (MSE) = {mse}")
    print(f"Mean Absolute Error (MAE) = {mae}")
