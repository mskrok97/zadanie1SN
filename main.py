import data_generator
from perceptron import Perceptron
from adaline import Adaline


def testowanie(perceptron: Perceptron):
    print("Testowanie")
    do_exit = False
    while not do_exit:
        print("Aby zakonczyc testowanie, zamiast podawac pierwsza liczbe, wpisz: exit")
        token = input("Podaj pierwsza liczbe: ")
        if token != "exit":
            input_1 = float(token)
            input_2 = float(input("Podaj druga liczbe: "))
            print(perceptron.test(input_1, input_2))
        else:
            do_exit = True

if __name__ == '__main__':
    print("Witaj w programie!")
    #learning_factor = float(input("Podaj wspolczynnik uczenia: "))
    #weight_range = float(input("Podaj gorna wartosc zakresu wag (dolna zostanie utworzona analogicznie): "))
    #func_type = input("Podaj rodzaj fukcji aktywacji (bipolarna - b / unipolarna - u : ")
    #threshold = float(input("Podaj wartosc progu: "))
    #bias = float(input("Podaj wartosc bias: "))
    print("Generowanie danych...")
    data = data_generator.generate_data(50)
    weights_p = data_generator.generate_weights(0.1)
    weights_a = weights_p
    print("Uczenie...")
    #perceptron = Perceptron(learning_factor, data_generator.generate_weights(weight_range),
    #                        func_type, threshold)

    perceptron = Perceptron(0.01, weights_p,
                            "u", 0.0, 0.05)
    perceptron.learn(data)

    adaline = Adaline(0.01, data_generator.generate_weights(0.1), 0.1, 0.05)
    adaline.learn(data)
    #for row in data:
    #    print(row)  #, "    Predykcja: ", perceptron.predict(row), "   error: ", perceptron.predict(row) - row[2],
    #          "     zmiana wag: ", (perceptron.predict(row) - row[2])*row[0:2])

    #for row in data:
    #    output = perceptron.predict(row)
    #    expected = perceptron.func(row[2])
    #    error = output - expected
    #    if error != +0.0 and error != -0.0 and error != 0.0:
    #        was_error = True
    #        #errors += 1
    #    weight_change = error * row[0:2]
    #    #self.bias = error * self.learning_factor
    #    perceptron.weights = perceptron.weights + perceptron.learning_factor * weight_change
    #    print(row, "    Predykcja: ", output, "   error: ", error,
    #          "     zmiana wag: ", weight_change, "    nowe wagi: ", perceptron.weights)
    #token = input("Czy chcesz przetestowac perceptron? (t/n): ")
    #if token == "t":
    #    testowanie(perceptron)
