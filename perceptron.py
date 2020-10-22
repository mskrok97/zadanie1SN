import numpy as np


class Perceptron:
    def __init__(self, learning_factor: float, weights,
                 func_type, threshold: float = 0, bias: float = 0):
        self.learning_factor = learning_factor
        self.weights = weights
        self.threshold = threshold
        self.bias = bias
        if func_type == "u":
            self.func = lambda value: 1.0 if value > self.threshold else 0.0
        else:
            self.func = lambda value: 1.0 if value > self.threshold else -1.0
        # print(weights)

    def predict(self, data_row) -> float:
        activation_sum = data_row[0]*self.weights[0] + data_row[1]*self.weights[1] + self.bias
        return self.func(activation_sum)

    def learn(self, data):
        epoch_count = 0
        was_error = True
        last_weights = self.weights
        last_error = 0.0
        while was_error:
            # print("Epoka: ", epoch_count + 1)
            was_error = False
            error_sum = 0

            for row in data:
                output = self.predict(row)
                expected = float(self.func(row[2]))  # zmiana wartości do rodzaju funkcji aktywacji
                error = expected - output
                if error == -0.0 or error == +0.0:
                    error = 0.0
                weight_change = error*row[0:2]
                self.bias += error*self.learning_factor
                self.weights = self.weights + self.learning_factor*weight_change

                if error != 0.0:
                    was_error = True
                    error_sum += 1

            new_error = float(error_sum)/len(data)*100

            if last_error != 0.0 and last_error < new_error:
                self.weights = last_weights
            else:
                last_weights = self.weights

            # print("blad: ", float(error_sum)/len(data)*100, "%")
            epoch_count += 1
            if epoch_count >= 1000:
                print("Osiagnieto maksymalna liczbe epok, nie udalo sie wyuczyc perceptronu")
                break

        print("Liczba epok: ", epoch_count)

    def test(self, input_1, input_2):
        print("Wynik: ", self.predict(np.array([input_1, input_2])))
