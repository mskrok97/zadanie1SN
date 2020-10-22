import random
import numpy as np


RANGE = 0.3
false_data = [[0, 0], [0, 1], [1, 0]]
true_data = [[1, 1]]


def generate_data(size):
    data = []
    for i in range(size):
        false_sample = false_data[random.randint(0, len(false_data) - 1)]
        true_sample = true_data[random.randint(0, len(true_data) - 1)]

        # np.append(data, [random.uniform(false_sample[0] - RANGE, false_sample[0] + RANGE),
        #                 random.uniform(false_sample[1] - RANGE, false_sample[1] + RANGE),
        #                 0], axis=0)
        # np.append(data, [random.uniform(true_sample[0] - RANGE, true_sample[0] + RANGE),
        #                 random.uniform(true_sample[1] - RANGE, true_sample[1] + RANGE),
        #                 1], axis=0)
        data.append(np.array(
                    [random.uniform(false_sample[0] - RANGE, false_sample[0] + RANGE),
                     random.uniform(false_sample[1] - RANGE, false_sample[1] + RANGE),
                     0.0]))
        data.append(np.array(
                    [random.uniform(true_sample[0] - RANGE, true_sample[0] + RANGE),
                     random.uniform(true_sample[1] - RANGE, true_sample[1] + RANGE),
                     1.0]))

    # print(data)
    random.shuffle(data)
    return data


def generate_weights(weight_range):
    return np.array([random.uniform(0 - weight_range, weight_range), random.uniform(0 - weight_range, weight_range)])
