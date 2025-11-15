import numpy as np


def target_function(x, y):
    return np.sin(np.abs(x)) * np.sin(x + y)

def generate_training_data(num_samples=1000, interval_start=0, interval_end=1):
    X = np.random.uniform(interval_start, interval_end, (num_samples, 2))
    Z = target_function(X[:, 0], X[:, 1]).reshape(-1, 1)

    return X, Z


if __name__ == '__main__':
    X_data, Z_data = generate_training_data()
    print(f"Форма вхідних даних (X): {X_data.shape}")
    print(f"Форма вихідних даних (Z): {Z_data.shape}")
    print(f"Приклад: f({X_data[0]}) = {Z_data[0]}")