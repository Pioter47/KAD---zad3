from math import pi, cos, sin, sqrt
# from random import random
import random
import math
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random.mtrand import rand

rand2 = np.random.RandomState(2)


def get_random_point(center: Tuple[float, float], radius: float) -> Tuple[float, float]:
    shift_x, shift_y = center

    a = rand2.random() * 2 * pi
    # a = random.random() * 2 * pi
    # r = radius * sqrt(random.random())
    r = radius * sqrt(rand2.random())

    return r * cos(a) + shift_x, r * sin(a) + shift_y


list_only_one_circle = [get_random_point((0, 0), 2) for x in range(200)]
df_only_one_circle = pd.DataFrame(list_only_one_circle)

list_first_circle = [get_random_point((-3, 0), 1) for x in range(100)]
list_second_circle = [get_random_point((3, 0), 1) for x in range(100)]
# list_third_circle = [get_random_point((0, 5), 3) for x in range(200)]

df_fist_circle = pd.DataFrame(list_first_circle)
df_second_circle = pd.DataFrame(list_second_circle)
# df_third_circle = pd.DataFrame(list_third_circle)

df_two_combined = df_fist_circle.append(df_second_circle)
# df_three_combined = df_two_combined.append(df_third_circle)
print(df_two_combined)

# tutaj zmieniamy liczbe neuronow (w range)
neurons = [get_random_point((0, 0), 3) for x in range(16)]
df_neurons_only_one_circle = pd.DataFrame(neurons)


neurons1 = [get_random_point((-3, 0), 2) for x in range(10)]
df_neurons1 = pd.DataFrame(neurons1)
neurons2 = [get_random_point((3, 0), 2) for x in range(10)]
df_neurons2 = pd.DataFrame(neurons2)
frames = [df_neurons1, df_neurons2]
df_neurons_combined = pd.concat(frames, ignore_index=True)

# Return index of the BMU in the grid
def find_BMU_my_second_version(SOM, elem):
    min_index = 0
    first_result = sqrt(((SOM[0][0] - elem[0]) ** 2)
                        + ((SOM[1][0]) - elem[1]) ** 2)
    for i in range(len(SOM)):
        f = sqrt(((SOM[0][i] - elem[0]) ** 2) + ((SOM[1][i]) - elem[1]) ** 2)
        # print(f)
        if f < first_result:
            first_result = f
            min_index = i
    return min_index



def update_weights(SOM, train_ex, learn_rate, radius_sq,
                   BMU_index):
    # if radius is close to zero then only BMU is changed
    if radius_sq < 1e-3:
        SOM.iloc[BMU_index] += learn_rate * (train_ex - SOM.iloc[BMU_index])
        return SOM


    # Change all cells in a small neighborhood of BMU
    for neuron in range(len(SOM)):
        dist_sq = sqrt(((SOM[0][neuron] - SOM[0][BMU_index]) ** 2)
                       + ((SOM[1][neuron]) - SOM[1][BMU_index]) ** 2)
        dist_func = np.exp(-dist_sq / 2 / radius_sq)
        SOM.iloc[neuron] += learn_rate * dist_func * (train_ex - SOM.iloc[neuron])

    return SOM


# Main routine for training an SOM. It requires an initialized SOM grid
# or a partially trained grid as parameter
def train_SOM(SOM, train_data, learn_rate=.1, radius_sq=.15,
              lr_decay=.1, radius_decay=.1, epochs=10):
    learn_rate_0 = learn_rate
    radius_0 = radius_sq
    for epoch in np.arange(0, epochs):
        # shuffle train_data
        train_data = train_data.sample(frac=1)
        for cell in range(len(train_data)):
            # dostac sie do wiersza o indeksie [cell]
            one_row = train_data.iloc[cell]
            index = find_BMU_my_second_version(SOM, one_row)
            SOM = update_weights(SOM, one_row,
                                 learn_rate, radius_sq, index)

        # Update learning rate and radius
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        radius_sq = radius_0 * np.exp(-epoch * radius_decay)
    return SOM


def blad_kwant(SOM, elem):
    first_result = sqrt(((SOM[0][0] - elem[0]) ** 2)
                        + ((SOM[1][0]) - elem[1]) ** 2)
    for i in range(len(SOM)):
        f = sqrt(((SOM[0][i] - elem[0]) ** 2) + ((SOM[1][i]) - elem[1]) ** 2)
        if f < first_result:
            first_result = f
    return first_result


if __name__ == '__main__':



    data = []
    epochs = []
    for epoch in range(0, 21):
        SOM = train_SOM(df_neurons_only_one_circle, df_only_one_circle, epochs=epoch)
        bledy = []
        for cell in range(len(df_only_one_circle)):
            one_row = df_two_combined.iloc[cell]
            bledy.append(blad_kwant(df_neurons_only_one_circle, one_row))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(df_only_one_circle[0], df_only_one_circle[1], color='hotpink')
        # ax1.scatter(df_fist_circle[0], df_fist_circle[1])
        # ax1.scatter(df_second_circle[0], df_second_circle[1])
        # ax1.scatter(df_neurons_combined[0], df_neurons_combined[1], c='black')
        ax1.scatter(df_neurons_only_one_circle[0], df_neurons_only_one_circle[1], c='black')
        plt.savefig("wykres dla epoki " + str(epoch), dpi=300)
        plt.close()
        bledy_ = sum(bledy) / len(bledy)
        data.append(bledy_)
        epochs.append(epoch)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(epochs, data)
    plt.xlim(0, 10)
    plt.ylim(0, 4)
    print(data)
    plt.scatter(epochs, data)
    plt.xlabel("liczba epok")
    plt.ylabel("średni błąd kwantyzacji")
    plt.title("wykres błędu kwantyzacji dla 16 neuronów")
    plt.savefig("wykres bledu kwantyzacji dla 16 neuronów", dpi=300)
    # plt.show()

