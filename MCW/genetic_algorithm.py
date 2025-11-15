import numpy as np
from neural_network import NeuralNetwork
from data_generator import generate_training_data


class GeneticAlgorithm:
    def __init__(self, nn_architecture, population_size=150, max_generations=3000,
                 mutation_rate=0.1, crossover_rate=0.8, target_mse=0.005):

        self.architecture = nn_architecture
        self.nn = NeuralNetwork(nn_architecture)
        self.weights_size = self.nn.total_weights_size

        self.pop_size = population_size
        self.max_gens = max_generations
        self.mut_rate = mutation_rate
        self.cross_rate = crossover_rate
        self.target_mse = target_mse

        self.X_train, self.Z_train = generate_training_data(num_samples=1000, interval_start=0, interval_end=1)

    def initialize_population(self):
        """Ініціалізація популяції випадковими вагами."""
        return np.random.uniform(-1.0, 1.0, (self.pop_size, self.weights_size))

    def calculate_fitness(self, individual):
        """Функція придатності: негативна Середньоквадратична Похибка (MSE)."""
        self.nn.set_weights(individual)
        predictions = self.nn.forward_pass(self.X_train)
        mse = np.mean((self.Z_train - predictions) ** 2)
        return -mse

    def selection_indices(self, fitnesses):
        """Селекція (турнірна) та повернення індексів переможців."""
        k = 5
        selected_indices = []
        for _ in range(self.pop_size):
            indices = np.random.choice(self.pop_size, k, replace=False)
            winner_global_index = indices[np.argmax(fitnesses[indices])]
            selected_indices.append(winner_global_index)
        return np.array(selected_indices)

    def crossover(self, parent1, parent2):
        """Кросинговер (одноточковий)."""
        if np.random.rand() < self.cross_rate:
            crossover_point = np.random.randint(1, self.weights_size - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        """Мутація (додавання Гауссівського шуму)."""
        if np.random.rand() < self.mut_rate:
            mutation_indices = np.random.choice(self.weights_size, size=int(self.weights_size * 0.1), replace=False)
            individual[mutation_indices] += np.random.normal(0, 0.1, len(mutation_indices))
        return individual

    def run(self):
        """Запуск генетичного алгоритму."""
        population = self.initialize_population()
        mse_log = []  # Ініціалізація логу MSE

        best_individual = None
        best_mse = float('inf')

        for gen in range(self.max_gens):
            fitnesses = np.array([self.calculate_fitness(ind) for ind in population])

            current_best_idx = np.argmax(fitnesses)
            current_best_individual = population[current_best_idx]
            current_best_mse = -fitnesses[current_best_idx]

            mse_log.append(current_best_mse)  # Додавання поточного MSE до логу

            if current_best_mse < best_mse:
                best_mse = current_best_mse
                best_individual = current_best_individual

            if gen % 100 == 0:
                print(f"Покоління {gen:4d}: Найкраща MSE = {current_best_mse:.8f} (Ціль: {self.target_mse})")

            if best_mse < self.target_mse:
                print(f"\nНавчання завершено в поколінні {gen}. Досягнута цільова MSE = {best_mse:.8f}")
                break

            parents_indices = self.selection_indices(fitnesses)
            new_population = [best_individual.copy()]

            for i in range(0, self.pop_size - 1, 2):
                p1 = population[parents_indices[i]]
                p2 = population[parents_indices[i + 1]] if i + 1 < self.pop_size else population[parents_indices[i]]

                child1, child2 = self.crossover(p1, p2)

                new_population.append(self.mutate(child1))
                if len(new_population) < self.pop_size:
                    new_population.append(self.mutate(child2))

            population = np.array(new_population)

        # ПОВЕРНЕННЯ ТРЬОХ ЗНАЧЕНЬ: ваги, фінальна MSE, лог MSE
        return best_individual, best_mse, mse_log