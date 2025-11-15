import numpy as np
from neural_network import NeuralNetwork
from genetic_algorithm import GeneticAlgorithm
from data_generator import generate_training_data, target_function
from visualization import plot_results

if __name__ == '__main__':
    architecture = [2, 4, 8, 10, 8, 8, 1]
    print(f"--- Базове Завдання: Навчання НМ {architecture} Генетичним Алгоритмом ---")

    ga = GeneticAlgorithm(
        nn_architecture=architecture,
        population_size=150,
        max_generations=3000,
        mutation_rate=0.1,
        target_mse=0.005
    )

    best_weights, final_mse, mse_log = ga.run()

    # --- Тестування ---
    trained_nn = NeuralNetwork(architecture)
    trained_nn.set_weights(best_weights)

    X_test, Z_test = generate_training_data(num_samples=200, interval_start=0, interval_end=1)
    test_mse = np.mean((Z_test - trained_nn.forward_pass(X_test)) ** 2)

    print("\n--- Фінальний Результат ---")
    print(f"Фінальна MSE на навчальних даних: {final_mse:.8f}")
    print(f"Тестова MSE (на нових даних): {test_mse:.8f}")

    FUNCTION_EXPRESSION = r'z = \sin(|x|) \cdot \sin(x+y)'

    plot_results(
        nn=trained_nn,
        mse_log=mse_log,
        architecture=architecture,
        target_func=target_function,
        func_expression=FUNCTION_EXPRESSION
    )