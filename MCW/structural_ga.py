import numpy as np
from neural_network import NeuralNetwork
from data_generator import generate_training_data  # Буде перевизначено в GUI


class StructuralGeneticAlgorithm:
    def __init__(self, population_size=100, max_generations=2000,
                 mutation_rate=0.1, crossover_rate=0.8,
                 max_neurons=15, max_hidden_layers=5):

        self.pop_size = population_size
        self.max_gens = max_generations
        self.mut_rate = mutation_rate
        self.cross_rate = crossover_rate
        self.max_neurons = max_neurons
        self.max_hidden_layers = max_hidden_layers

        # Навчальні дані будуть встановлені з GUI
        self.X_train = None
        self.Z_train = None

    def _decode_structure(self, individual):
        """Декодує структурну частину хромосоми в архітектуру НМ.
           Структурна частина - це перші max_hidden_layers генів."""

        # Перші гени кодують кількість нейронів у прихованих шарах
        structure_genes = individual[:self.max_hidden_layers]

        # Обмеження нейронів
        structure_genes = np.clip(structure_genes, 0, self.max_neurons).astype(int)

        # Створення архітектури (Ігноруємо шари з 0 нейронами)
        hidden_layers = [n for n in structure_genes if n > 0]

        # Забезпечення мінімальної архітектури (2-1-1), якщо всі приховані шари були видалені
        if not hidden_layers:
            hidden_layers = [1]

        architecture = [2] + hidden_layers + [1]

        # Індекс, звідки починаються ваги
        weights_start_idx = self.max_hidden_layers

        return architecture, weights_start_idx

    def _encode_individual(self, architecture, weights):
        """Створює хромосому (структура + ваги) з архітектури та ваг."""

        # Структурна частина (для фіксованої довжини)
        structure_genes = np.zeros(self.max_hidden_layers)
        num_hidden = len(architecture) - 2

        # Встановлюємо кількість нейронів у прихованих шарах
        for i in range(num_hidden):
            # Перевірка на обмеження
            if i < self.max_hidden_layers:
                structure_genes[i] = architecture[i + 1]

        # Повна хромосома
        return np.concatenate((structure_genes, weights))

    def calculate_fitness(self, individual):
        """Оцінка придатності: тепер включає декодування структури та штраф за складність."""

        architecture, weights_start_idx = self._decode_structure(individual)
        flat_weights = individual[weights_start_idx:]

        if self.X_train is None or self.Z_train is None:
            return -1e10  # Повернення дуже поганої придатності, якщо дані не завантажені

        temp_nn = NeuralNetwork(architecture)

        # Перевірка на відповідність розміру ваг (дуже важливо після операторів ГА)
        if temp_nn.total_weights_size != len(flat_weights):
            # Якщо розмір ваг не відповідає структурі (наприклад, після схрещування різних структур),
            # ми повинні переініціалізувати ваги, щоб індивід не помер, але отримати штраф.
            # Або, для простоти:
            return -1e8  # Повернення дуже поганої придатності

        temp_nn.set_weights(flat_weights)
        predictions = temp_nn.forward_pass(self.X_train)

        mse = np.mean((self.Z_train - predictions) ** 2)

        # Штраф за складність (принцип Оккама)
        complexity_penalty = 0.0001 * sum(architecture[1:-1])

        return -(mse + complexity_penalty)

    def initialize_population(self):
        """Ініціалізація популяції з випадковою структурою та вагами."""
        population = []
        for _ in range(self.pop_size):
            # Випадкова кількість прихованих шарів (1 до max_hidden_layers)
            num_hidden = np.random.randint(1, self.max_hidden_layers + 1)

            # Випадкова кількість нейронів у кожному прихованому шарі
            hidden_layers = np.random.randint(1, self.max_neurons + 1, size=num_hidden)

            architecture = [2] + list(hidden_layers) + [1]

            # Генерація ваг для цієї структури
            temp_nn = NeuralNetwork(architecture)
            weights = np.random.uniform(-1.0, 1.0, temp_nn.total_weights_size)

            # Створення хромосоми: структура (фіксована довжина) + ваги
            individual = self._encode_individual(architecture, weights)
            population.append(individual)

        return population

    def selection_indices(self, fitnesses):
        """Селекція (турнірна) та повернення індексів переможців."""
        k = 5
        selected_indices = []
        for _ in range(self.pop_size):
            indices = np.random.choice(len(fitnesses), k, replace=False)
            winner_global_index = indices[np.argmax(fitnesses[indices])]
            selected_indices.append(winner_global_index)
        return np.array(selected_indices)

    def crossover(self, parent1, parent2):
        """Кросинговер: схрещуємо структурну частину та ваги, якщо вони сумісні."""
        if np.random.rand() < self.cross_rate:

            arch1, weights_start_idx1 = self._decode_structure(parent1)
            arch2, weights_start_idx2 = self._decode_structure(parent2)

            weights1 = parent1[weights_start_idx1:]
            weights2 = parent2[weights_start_idx2:]

            # 1. Кросинговер структурної частини (перші max_hidden_layers генів)
            cross_point_struct = np.random.randint(1, self.max_hidden_layers)
            struct_child1 = np.concatenate(
                (parent1[:cross_point_struct], parent2[cross_point_struct:self.max_hidden_layers]))
            struct_child2 = np.concatenate(
                (parent2[:cross_point_struct], parent1[cross_point_struct:self.max_hidden_layers]))

            # 2. Кросинговер ваг (якщо вони мають приблизно однакову довжину)
            if len(weights1) == len(weights2) and len(weights1) > 2:
                cross_point_weights = np.random.randint(1, len(weights1) - 1)
                weights_child1 = np.concatenate((weights1[:cross_point_weights], weights2[cross_point_weights:]))
                weights_child2 = np.concatenate((weights2[:cross_point_weights], weights1[cross_point_weights:]))
            else:
                # Якщо довжина різна, використовуємо ваги першого батька
                weights_child1 = weights1.copy()
                weights_child2 = weights2.copy()

            # 3. Створення нового індивіда (структура + ваги)
            # ВАЖЛИВО: Оскільки кросинговер може створити нову структуру,
            # але зберегти ваги, які не відповідають структурі, ці індивіди
            # отримають штраф у calculate_fitness.
            child1 = np.concatenate((struct_child1, weights_child1))
            child2 = np.concatenate((struct_child2, weights_child2))

            return child1, child2
        else:
            # Гарантоване повернення двох індивідів
            return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        """Мутація (ваги + можливість зміни структури)."""

        architecture, weights_start_idx = self._decode_structure(individual)
        flat_weights = individual[weights_start_idx:]

        # 1. Мутація ваг
        if np.random.rand() < self.mut_rate:
            mutation_indices = np.random.choice(len(flat_weights), size=int(len(flat_weights) * 0.1), replace=False)
            flat_weights[mutation_indices] += np.random.normal(0, 0.1, len(mutation_indices))

            # Новий індивід після мутації ваг
            mutated_individual = self._encode_individual(architecture, flat_weights)
        else:
            mutated_individual = individual.copy()

        # 2. Мутація структури (застосовується до вже потенційно мутованого індивіда)
        if np.random.rand() < 0.05:
            new_architecture = list(architecture)

            # Випадкова зміна: додати нейрон (1), видалити нейрон (-1)
            struct_change = np.random.choice([-1, 1])

            if struct_change == 1 and len(new_architecture) < self.max_hidden_layers + 2:
                # Додати нейрон
                layer_to_change = np.random.randint(1, len(new_architecture) - 1)
                new_architecture[layer_to_change] = min(new_architecture[layer_to_change] + 1, self.max_neurons)

            elif struct_change == -1 and len(new_architecture) > 3:  # Мінімум 2-1-1
                # Видалити нейрон
                layer_to_change = np.random.randint(1, len(new_architecture) - 1)
                new_architecture[layer_to_change] -= 1

                # Видалити шар, якщо кількість нейронів стала 0
                if new_architecture[layer_to_change] == 0:
                    new_architecture.pop(layer_to_change)

            # Створення нової НМ та ініціалізація нових ваг
            if new_architecture != architecture:
                new_nn = NeuralNetwork(new_architecture)
                # Повна реініціалізація ваг, оскільки структура змінилася
                new_flat_weights = np.random.uniform(-1.0, 1.0, new_nn.total_weights_size)

                # Гарантоване повернення індивіда
                return self._encode_individual(new_architecture, new_flat_weights)

        # Якщо структурна мутація не відбулася або не призвела до зміни,
        # повертаємо індивіда після мутації ваг (або оригінального, якщо мутація ваг не відбулася).
        return mutated_individual