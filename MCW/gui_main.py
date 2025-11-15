import tkinter as tk
from tkinter import messagebox, scrolledtext
import numpy as np
from structural_ga import StructuralGeneticAlgorithm
from neural_network import NeuralNetwork
import threading
import math
from visualization import plot_results


def parse_and_evaluate_func(func_str, x, y):
    local_dict = {'x': x, 'y': y, 'sin': math.sin, 'cos': math.cos, 'pi': math.pi, 'abs': abs}
    try:
        return eval(func_str, {"__builtins__": None}, local_dict)
    except Exception as e:
        return 1e9


def target_function_gui(x_coords, y_coords, func_str):
    results = np.zeros_like(x_coords)
    for i in range(len(x_coords)):
        results[i] = parse_and_evaluate_func(func_str, x_coords[i], y_coords[i])
    return results.reshape(-1, 1)


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
        tk.Label(self.master, text="Функція f(x, y):").grid(row=0, column=0, sticky="w")
        self.func_var = tk.StringVar(value="sin(abs(x)) * sin(x+y)")
        tk.Entry(self.master, textvariable=self.func_var, width=50).grid(row=0, column=1, sticky="w")

        tk.Label(self.master, text="Інтервал [a-b]:").grid(row=1, column=0, sticky="w")
        self.interval_var = tk.StringVar(value="0-1")
        tk.Entry(self.master, textvariable=self.interval_var).grid(row=1, column=1, sticky="w")

        tk.Label(self.master, text="Макс. нейронів на шар:").grid(row=2, column=0, sticky="w")
        self.max_neurons_var = tk.IntVar(value=15)
        tk.Entry(self.master, textvariable=self.max_neurons_var).grid(row=2, column=1, sticky="w")

        tk.Label(self.master, text="Цільова MSE (навчання):").grid(row=3, column=0, sticky="w")
        self.target_mse_var = tk.DoubleVar(value=0.01)
        tk.Entry(self.master, textvariable=self.target_mse_var).grid(row=3, column=1, sticky="w")

        tk.Label(self.master, text="Макс. Поколінь:").grid(row=4, column=0, sticky="w")
        self.max_gens_var = tk.IntVar(value=500)
        tk.Entry(self.master, textvariable=self.max_gens_var).grid(row=4, column=1, sticky="w")

    def create_output_widgets(self):
        tk.Label(self.master, text="--- Результат Синтезу ---").grid(row=6, column=0, columnspan=2, pady=(10, 0))

        tk.Label(self.master, text="Отримана Структура:").grid(row=7, column=0, sticky="w")
        self.result_struct_var = tk.StringVar(value="Очікування...")
        tk.Label(self.master, textvariable=self.result_struct_var, fg="blue").grid(row=7, column=1, sticky="w")

        tk.Label(self.master, text="Фінальна MSE:").grid(row=8, column=0, sticky="w")
        self.result_mse_var = tk.StringVar(value="Очікування...")
        tk.Label(self.master, textvariable=self.result_mse_var, fg="red").grid(row=8, column=1, sticky="w")

    def create_log_widget(self):
        tk.Label(self.master, text="--- Лог Роботи ГА ---").grid(row=9, column=0, columnspan=2, pady=(10, 0))
        self.log_text = scrolledtext.ScrolledText(self.master, height=10, width=60, state='disabled')
        self.log_text.grid(row=10, column=0, columnspan=2, padx=10, pady=5)

    def log_message(self, message):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
        self.master.after(0, self.master.update)

    def start_synthesis_thread(self):
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

            self.ga_instance = StructuralGeneticAlgorithm(
                population_size=100,
                max_generations=max_gens,
                mutation_rate=0.1,
                crossover_rate=0.8,
                max_neurons=max_neurons
            )

            X_train, Z_train = self.generate_data_for_gui(func_str, interval_start, interval_end)
            self.ga_instance.X_train = X_train
            self.ga_instance.Z_train = Z_train

            self.log_message(f"--- Параметри ГА ---")
            self.log_message(f"Функція: {func_str}, Інтервал: [{interval_start}, {interval_end}]")

            best_individual = None
            final_mse = 1e9
            mse_log = []

            population = self.ga_instance.initialize_population()

            for gen in range(max_gens):

                fitnesses = np.array([self.ga_instance.calculate_fitness(ind) for ind in population])

                best_idx = np.argmax(fitnesses)
                best_individual = population[best_idx]
                best_mse = -fitnesses[best_idx]

                best_arch, _ = self.ga_instance._decode_structure(best_individual)
                mse_log.append(best_mse)

                if gen % 10 == 0:
                    self.log_message(
                        f"Пок. {gen:03d} | MSE: {best_mse:.6f} | Структура: {'-'.join(map(str, best_arch))}")

                if best_mse < target_mse:
                    self.log_message(f"\nКритерій MSE досягнуто! MSE < {target_mse}")
                    final_mse = best_mse
                    break

                parents_indices = self.ga_instance.selection_indices(fitnesses)
                new_population = [best_individual.copy()]

                for i in range(0, self.ga_instance.pop_size - 1, 2):
                    p1 = population[parents_indices[i]]
                    p2 = population[parents_indices[i + 1]] if i + 1 < self.ga_instance.pop_size else population[
                        parents_indices[i]]

                    child1, child2 = self.ga_instance.crossover(p1, p2)

                    new_population.append(self.ga_instance.mutate(child1))
                    if len(new_population) < self.ga_instance.pop_size:
                        new_population.append(self.ga_instance.mutate(child2))

                population = new_population[:self.ga_instance.pop_size]

            final_arch, weights_start_idx = self.ga_instance._decode_structure(best_individual)
            final_weights = best_individual[weights_start_idx:]

            self.result_struct_var.set(f"{'-'.join(map(str, final_arch))}")
            self.result_mse_var.set(f"{best_mse:.6f}")
            self.log_message(f"\n*** ФІНАЛЬНИЙ РЕЗУЛЬТАТ ***")
            self.log_message(f"Структура: {'-'.join(map(str, final_arch))}")
            self.log_message(f"MSE: {best_mse:.6f}")

            final_nn = NeuralNetwork(final_arch)
            final_nn.set_weights(final_weights)

            self.master.after(0, lambda: self.show_final_plots(final_nn, final_arch, mse_log, func_str, interval_start,
                                                               interval_end))


        except Exception as e:
            error_message = f"Виникла помилка під час синтезу: {e}"
            messagebox.showerror("Помилка ГА", error_message)
            self.log_message(f"ПОМИЛКА: {e}")
        finally:
            self.start_button.config(state=tk.NORMAL, text="Запустити Синтез")

    def generate_data_for_gui(self, func_str, start, end, num_samples=1000):
        X = np.random.uniform(start, end, (num_samples, 2))
        x_coords = X[:, 0]
        y_coords = X[:, 1]

        Z = target_function_gui(x_coords, y_coords, func_str)
        return X, Z

    def show_final_plots(self, nn, arch, mse_log, func_str, start, end):
        def compatible_target_func(x, y):
            return target_function_gui(x, y, func_str).ravel()

        plot_results(
            nn=nn,
            mse_log=mse_log,
            architecture=arch,
            target_func=compatible_target_func,
            func_expression=func_str,
            interval_start=start,
            interval_end=end
        )


if __name__ == '__main__':
    root = tk.Tk()
    app = StructuralSynthesisApp(root)
    root.mainloop()