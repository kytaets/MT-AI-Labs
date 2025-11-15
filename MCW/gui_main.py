import tkinter as tk
from tkinter import messagebox, scrolledtext
import numpy as np
from structural_ga import StructuralGeneticAlgorithm
from neural_network import NeuralNetwork
import threading
import math


# --- Допоміжні функції для роботи з рядком функції ---
def parse_and_evaluate_func(func_str, x, y):
    """Парсить та обчислює функцію. Підтримує math.sin, math.cos, math.pi, abs."""
    # Доступні функції та константи
    local_dict = {'x': x, 'y': y, 'sin': math.sin, 'cos': math.cos, 'pi': math.pi, 'abs': abs}
    try:
        # Використовуємо eval з обережністю (тут для навчальних цілей)
        return eval(func_str, {"__builtins__": None}, local_dict)
    except Exception as e:
        # Повернення дуже великого значення у випадку помилки обчислення
        return 1e9


def target_function_gui(x_coords, y_coords, func_str):
    """Обчислює функцію для масиву NumPy, використовуючи parse_and_evaluate_func."""
    results = np.zeros_like(x_coords)
    for i in range(len(x_coords)):
        results[i] = parse_and_evaluate_func(func_str, x_coords[i], y_coords[i])
    return results.reshape(-1, 1)


# --- Клас GUI ---

class StructuralSynthesisApp:
    def __init__(self, master):
        self.master = master
        master.title("Структурний Синтез НМПП (ГА) +20")

        self.create_input_widgets()
        self.create_output_widgets()
        self.create_log_widget()

        self.start_button = tk.Button(master, text="Запустити Синтез", command=self.start_synthesis_thread)
        self.start_button.grid(row=5, column=0, columnspan=2, pady=10)

        self.ga_instance = None
        self.stop_event = threading.Event()

    def create_input_widgets(self):
        # Функція (Ваш варіант 69)
        tk.Label(self.master, text="Функція f(x, y):").grid(row=0, column=0, sticky="w")
        self.func_var = tk.StringVar(value="sin(abs(x)) * sin(x+y)")
        tk.Entry(self.master, textvariable=self.func_var, width=50).grid(row=0, column=1, sticky="w")

        # Інтервал
        tk.Label(self.master, text="Інтервал [a-b]:").grid(row=1, column=0, sticky="w")
        self.interval_var = tk.StringVar(value="0-1")
        tk.Entry(self.master, textvariable=self.interval_var).grid(row=1, column=1, sticky="w")

        # Макс. Нейронів
        tk.Label(self.master, text="Макс. нейронів на шар:").grid(row=2, column=0, sticky="w")
        self.max_neurons_var = tk.IntVar(value=15)
        tk.Entry(self.master, textvariable=self.max_neurons_var).grid(row=2, column=1, sticky="w")

        # Цільова Похибка
        tk.Label(self.master, text="Цільова MSE (навчання):").grid(row=3, column=0, sticky="w")
        self.target_mse_var = tk.DoubleVar(value=0.01)
        tk.Entry(self.master, textvariable=self.target_mse_var).grid(row=3, column=1, sticky="w")

        # Макс. Поколінь
        tk.Label(self.master, text="Макс. Поколінь:").grid(row=4, column=0, sticky="w")
        self.max_gens_var = tk.IntVar(value=500)
        tk.Entry(self.master, textvariable=self.max_gens_var).grid(row=4, column=1, sticky="w")

    def create_output_widgets(self):
        # Вивід структури
        tk.Label(self.master, text="--- Результат Синтезу ---").grid(row=6, column=0, columnspan=2, pady=(10, 0))

        tk.Label(self.master, text="Отримана Структура:").grid(row=7, column=0, sticky="w")
        self.result_struct_var = tk.StringVar(value="Очікування...")
        tk.Label(self.master, textvariable=self.result_struct_var, fg="blue").grid(row=7, column=1, sticky="w")

        # Вивід похибки
        tk.Label(self.master, text="Фінальна MSE:").grid(row=8, column=0, sticky="w")
        self.result_mse_var = tk.StringVar(value="Очікування...")
        tk.Label(self.master, textvariable=self.result_mse_var, fg="red").grid(row=8, column=1, sticky="w")

    def create_log_widget(self):
        # Лог ГА (проміжкові варіанти)
        tk.Label(self.master, text="--- Лог Роботи ГА ---").grid(row=9, column=0, columnspan=2, pady=(10, 0))
        self.log_text = scrolledtext.ScrolledText(self.master, height=10, width=60, state='disabled')
        self.log_text.grid(row=10, column=0, columnspan=2, padx=10, pady=5)

    def log_message(self, message):
        """Додає повідомлення до логу."""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
        # Для коректного оновлення в потоці:
        self.master.after(0, self.master.update)

    def start_synthesis_thread(self):
        """Запускає ГА у окремому потоці, щоб не блокувати GUI."""
        self.start_button.config(state=tk.DISABLED, text="Виконання...")
        self.log_text.configure(state='normal');
        self.log_text.delete(1.0, tk.END);
        self.log_text.configure(state='disabled')
        self.result_struct_var.set("Синтез запущено...")
        self.result_mse_var.set("Синтез запущено...")

        self.synthesis_thread = threading.Thread(target=self.run_synthesis)
        self.synthesis_thread.start()

    def run_synthesis(self):
        try:
            func_str = self.func_var.get()
            interval_str = self.interval_var.get().split('-')
            interval_start = float(interval_str[0])
            interval_end = float(interval_str[1])
            max_neurons = self.max_neurons_var.get()
            max_gens = self.max_gens_var.get()
            target_mse = self.target_mse_var.get()

            # Налаштування ГА
            self.ga_instance = StructuralGeneticAlgorithm(
                population_size=100,
                max_generations=max_gens,
                mutation_rate=0.1,
                crossover_rate=0.8,
                max_neurons=max_neurons,
                max_hidden_layers=5  # Фіксовано
            )

            # Завантаження даних
            X_train, Z_train = self.generate_data_for_gui(func_str, interval_start, interval_end)
            self.ga_instance.X_train = X_train
            self.ga_instance.Z_train = Z_train

            self.log_message(f"--- Параметри ГА ---")
            self.log_message(f"Функція: {func_str}")
            self.log_message(f"Макс. нейронів на шар: {max_neurons}")
            self.log_message(f"--------------------")

            best_individual = None
            final_mse = 1e9

            population = self.ga_instance.initialize_population()

            for gen in range(max_gens):

                fitnesses = np.array([self.ga_instance.calculate_fitness(ind) for ind in population])

                # Найкращий індивід
                best_idx = np.argmax(fitnesses)
                best_individual = population[best_idx]
                best_mse = -fitnesses[best_idx]

                best_arch, _ = self.ga_instance._decode_structure(best_individual)

                if gen % 10 == 0:
                    self.log_message(
                        f"Пок. {gen:03d} | MSE: {best_mse:.6f} | Структура: {'-'.join(map(str, best_arch))}")

                # Критерій зупинки
                if best_mse < target_mse:
                    self.log_message(f"\nКритерій MSE досягнуто! MSE < {target_mse}")
                    final_mse = best_mse
                    break

                # Селекція
                parents_indices = self.ga_instance.selection_indices(fitnesses)

                # Елітизм
                new_population = [best_individual.copy()]

                for i in range(0, self.ga_instance.pop_size - 1, 2):
                    p1 = population[parents_indices[i]]
                    p2 = population[parents_indices[i + 1]] if i + 1 < self.ga_instance.pop_size else population[
                        parents_indices[i]]

                    # Кросинговер та Мутація
                    child1, child2 = self.ga_instance.crossover(p1, p2)

                    new_population.append(self.ga_instance.mutate(child1))
                    if len(new_population) < self.ga_instance.pop_size:
                        new_population.append(self.ga_instance.mutate(child2))

                # Оновлення популяції (видалення надлишку, якщо pop_size непарний)
                population = new_population[:self.ga_instance.pop_size]

            # Фінальний вивід
            final_arch, _ = self.ga_instance._decode_structure(best_individual)
            self.result_struct_var.set(f"{'-'.join(map(str, final_arch))}")
            self.result_mse_var.set(f"{best_mse:.6f}")
            self.log_message(f"\n*** ФІНАЛЬНИЙ РЕЗУЛЬТАТ ***")
            self.log_message(f"Структура: {'-'.join(map(str, final_arch))}")
            self.log_message(f"MSE: {best_mse:.6f}")

        except Exception as e:
            # Обробка помилки та виведення її у GUI
            error_message = f"Виникла помилка під час синтезу: {e}"
            messagebox.showerror("Помилка ГА", error_message)
            self.log_message(f"ПОМИЛКА: {e}")
        finally:
            self.start_button.config(state=tk.NORMAL, text="Запустити Синтез")

    def generate_data_for_gui(self, func_str, start, end, num_samples=1000):
        """Генерує дані, використовуючи функцію з GUI."""
        X = np.random.uniform(start, end, (num_samples, 2))
        x_coords = X[:, 0]
        y_coords = X[:, 1]

        Z = target_function_gui(x_coords, y_coords, func_str)
        return X, Z


if __name__ == '__main__':
    root = tk.Tk()
    app = StructuralSynthesisApp(root)
    root.mainloop()