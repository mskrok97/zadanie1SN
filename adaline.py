import numpy as np
import math


class Adaline:
    def __init__(self, learning_factor: float, weights, acceptable_error, bias: float = 0):
        self.learning_factor = learning_factor
        self.weights = weights
        self.acceptable_error = acceptable_error
        self.bias = bias
        self.func = lambda value: 1.0 if value > 0.0 else -1.0
        # print(weights)

    def learn(self, data):
        epochs = 0
        errors = []
        error = self.acceptable_error + 1

        while error > self.acceptable_error:
            # print("Epoka: ", epochs + 1)

            for row in data:
                output = self.predict(row)
                expected = float(self.func(row[2]))
                diff = expected - output
                delta = math.pow(diff, 2)
                errors.append(delta)
                self.weights = self.weights + 2*self.learning_factor*diff*row[0:2]
                self.bias += 2*self.learning_factor*diff

            # print(self.weights)
            error = np.sum(errors) / np.size(errors)
            # print("Blad: ", error)
            epochs += 1

            if epochs >= 1000:
                print("Osiagnieto maksymalna liczbe epok, nie udalo sie wyuczyc Adaline")
                break
        print("Liczba epok: ", epochs)

    def predict(self, data_row) -> float:
        activation_sum = data_row[0] * self.weights[0] + data_row[1] * self.weights[1] + self.bias
        return self.func(activation_sum)

    def test(self, input_1, input_2):
        print("Wynik: ", self.predict(np.array([input_1, input_2])))
