import numpy as np


class NeuralNetwork:
    def __init__(self, architecture):
        self.architecture = architecture
        self.num_layers = len(architecture)

        self.activation = lambda x: np.maximum(0, x)
        self.output_activation = lambda x: x  # Linear

        self.total_weights_size = self._calculate_total_weights()

    def _calculate_total_weights(self):
        total_size = 0
        for i in range(1, self.num_layers):
            weights_size = self.architecture[i - 1] * self.architecture[i]
            bias_size = self.architecture[i]
            total_size += weights_size + bias_size
        return total_size

    def set_weights(self, flat_weights):
        self.weights = []
        self.biases = []
        current_index = 0

        for i in range(1, self.num_layers):
            prev_size = self.architecture[i - 1]
            curr_size = self.architecture[i]

            weights_len = prev_size * curr_size
            W = flat_weights[current_index: current_index + weights_len].reshape(prev_size, curr_size)
            current_index += weights_len
            self.weights.append(W)

            bias_len = curr_size
            b = flat_weights[current_index: current_index + bias_len].reshape(1, curr_size)
            current_index += bias_len
            self.biases.append(b)

    def forward_pass(self, X):
        a = X
        for i in range(self.num_layers - 2):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.activation(z)

        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        output = self.output_activation(z)

        return output