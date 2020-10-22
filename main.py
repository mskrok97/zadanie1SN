import data_generator
from perceptron import Perceptron
from adaline import Adaline


def test(model):
    do_exit = False
    while not do_exit:
        token = input("Czy chcesz przetestowac perceptron? (t/n): ")
        if token == "t":
            input_1 = float(input("Podaj pierwsza liczbe: "))
            input_2 = float(input("Podaj druga liczbe: "))
            print(model.test(input_1, input_2))
        else:
            do_exit = True


if __name__ == '__main__':
    print("Witaj w programie!")
    p_type = input("Podaj perceptron czy adaline? (p/a): ")
    learning_factor = float(input("Podaj wspolczynnik uczenia: "))
    weight_range = float(input("Podaj gorna wartosc zakresu wag (dolna zostanie utworzona analogicznie): "))
    print("Generowanie danych...")
    data = data_generator.generate_data(50)
    weights = data_generator.generate_weights(weight_range)
    bias = float(input("Podaj wartosc bias: "))

    if p_type == "p":
        func_type = input("Podaj rodzaj fukcji aktywacji (bipolarna - b / unipolarna - u : ")
        threshold = float(input("Podaj wartosc progu: "))
        perceptron = Perceptron(learning_factor, weights,
                                func_type, threshold, bias)
        print("Uczenie...")
        perceptron.learn(data)
        test(perceptron)
    elif p_type == "a":
        acceptable_error = input("Podaj dopuszczalny blad: ")
        adaline = Adaline(learning_factor, weights, acceptable_error, bias)
        print("Uczenie...")
        adaline.learn(data)
        test(adaline)
    else:
        print("Nierozpoznawany model")

    # perceptron = Perceptron(0.01, weights_p,
    #                        "u", 0.0, 0.05)

    # adaline = Adaline(0.01, data_generator.generate_weights(0.1), 0.1, 0.05)
    # adaline.learn(data)

    #token = input("Czy chcesz przetestowac perceptron? (t/n): ")
    #if token == "t":
    #    testowanie(perceptron)
