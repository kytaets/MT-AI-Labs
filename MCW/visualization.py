import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from data_generator import target_function
from neural_network import NeuralNetwork

# Встановлення стандартного бекенду для уникнення проблем PyCharm
plt.switch_backend('TkAgg')

INTERVAL_START = 0
INTERVAL_END = 1
GRID_SIZE = 50


def generate_grid_data(nn, interval_start, interval_end):
    """Генерує дані для побудови 3D-поверхонь."""
    x = np.linspace(interval_start, interval_end, GRID_SIZE)
    y = np.linspace(interval_start, interval_end, GRID_SIZE)
    X, Y = np.meshgrid(x, y)

    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Input_data = np.vstack([X_flat, Y_flat]).T

    Z_true = target_function(X_flat, Y_flat).reshape(X.shape)
    Z_pred = nn.forward_pass(Input_data).ravel().reshape(X.shape)

    return X, Y, Z_true, Z_pred


def plot_learning_curve(mse_log):
    """Побудова графіка навчання (Крива MSE) в окремому вікні."""

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(mse_log, label='MSE найкращого індивіда', color='blue')
    ax.set_title('Вікно 1: Крива Навчання (GA) - Зменшення MSE')
    ax.set_xlabel('Покоління')
    ax.set_ylabel('MSE (лог. шкала)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    plt.show()  # Показує перше вікно і чекає на його закриття


def plot_3d_results(nn, architecture, X_grid, Y_grid, Z_true, Z_pred, Z_diff):
    """Побудова 3D-графіків (Справжня та Смодельована) у другому вікні."""

    # Розмітка 1 ряд, 2 колонки
    fig = plt.figure(figsize=(16, 8))

    # --- Графік 1: Справжня функція ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X_grid, Y_grid, Z_true, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.8)
    ax1.set_title(r'2. Справжня функція $z = \sin(|x|) \cdot \sin(x+y)$')
    ax1.set_xlabel('X');
    ax1.set_ylabel('Y');
    ax1.set_zlabel('Z')

    # --- Графік 2: Смодельована функція ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X_grid, Y_grid, Z_pred, cmap=cm.plasma, linewidth=0, antialiased=False, alpha=0.8)
    ax2.set_title(f'3. Смодельована НМ (Архітектура: {"-".join(map(str, architecture))})')
    ax2.set_xlabel('X');
    ax2.set_ylabel('Y');
    ax2.set_zlabel('Z')

    # --- Графік 3: Абсолютна Похибка (ВИДАЛЕНО) ---

    plt.suptitle('Вікно 2: Порівняння Моделювання Функції Двох Змінних', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()  # Показує друге вікно


def plot_results(nn, mse_log, architecture):
    """Головна функція, яка викликає послідовне відображення вікон."""

    # --- Крок 1: Вікно 1 (Крива Навчання) ---
    print("\nВікно 1: Показано криву навчання (MSE). Будь ласка, закрийте його для переходу до наступного вікна.")
    plot_learning_curve(mse_log)

    # --- Підготовка даних для Вікна 2 ---
    X_grid, Y_grid, Z_true, Z_pred = generate_grid_data(nn, INTERVAL_START, INTERVAL_END)
    Z_diff = np.abs(Z_true - Z_pred)  # Похибка все ще обчислюється, але не відображається

    # --- Крок 2: Вікно 2 (3D-графіки) ---
    print("Вікно 2: Показано 3D-графіки моделювання. Закрийте його для завершення програми.")
    plot_3d_results(nn, architecture, X_grid, Y_grid, Z_true, Z_pred, Z_diff)