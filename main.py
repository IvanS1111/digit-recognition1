import math
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def act_func(x):
    x_plus = np.exp(x)
    x_minus = np.exp(-x)
    return (x_plus - x_minus) / (x_plus + x_minus)
    #return 1/(1 + np.exp(x))
    #return max(0, x)


def df(x):
    return 1 - act_func(x) ** 2
    #return act_func(x) * (1 - act_func(x))
    #return 1


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

INPUT_LAYER = 784
MIDDLE_LAYER = 100
OUTPUT_LAYER = 10
SPEED = 0.001
LEFT = -0.02
RIGHT = 0.02
BIAS = 1
STATISTICS = []
stat = 0
SIZE = 5000

#w1 = np.zeros((INPUT_LAYER, MIDDLE_LAYER))
w1 = np.load('w11.npy')
#for i in range(INPUT_LAYER):
    #for j in range(MIDDLE_LAYER):
        #w1[i][j] = random.uniform(LEFT, RIGHT)

#w2 = np.zeros((MIDDLE_LAYER, OUTPUT_LAYER))
w2 = np.load('w22.npy')
#for i in range(MIDDLE_LAYER):
    #for j in range(OUTPUT_LAYER):
        #w2[i][j] = random.uniform(LEFT, RIGHT)

for t in range(45000, 45007):
    x = []
    #x.append(BIAS)
    for i in range(28):
        for j in range(28):
            x.append(x_train[t][i][j])

    x = np.array(x)
    for i in range(len(x)):
        if x[i] > 0:
            x[i] = 1.0
        else:
            x[i] = 0.0
    plt.imshow(x_train[t], cmap=plt.cm.binary)
    plt.show()
    s1 = np.dot(x, w1)

    f1 = np.array([act_func(s1[i]) for i in range(MIDDLE_LAYER)])
    s2 = np.dot(f1, w2)

    f2 = np.array([act_func(s2[i]) for i in range(OUTPUT_LAYER)])
    MAX = -100000.0
    INDEX = 0
    for i in range(len(f2)):
            if f2[i] > MAX:
                MAX = f2[i]
                INDEX  = i
    print('net_ans = ', INDEX)
    print('true_ans = ', y_train[t])
    if INDEX == y_train[t]:
        stat += 1
    if t % 99 == 0:
        STATISTICS.append(stat)
        print('#############################')
        print('t = ', t)
        print('stat = ', stat)
        print('#############################')
        stat = 0
    print(f2)

    true_arr = np.array([0.0 for i in range(OUTPUT_LAYER)])
    true_arr[y_train[t]] = 1.0

    delta2 = np.array([(true_arr[i] - f2[i]) for i in range(OUTPUT_LAYER)])

    delta1 = np.array([0.0 for i in range(MIDDLE_LAYER)])
    for i in range(MIDDLE_LAYER):
        for j in range(OUTPUT_LAYER):
            delta1[i] += delta2[j] * w2[i][j]

    for i in range(INPUT_LAYER):
        for j in range(MIDDLE_LAYER):
            w1[i][j] += SPEED * delta1[j] * df(s1[j]) * x[i]

    for i in range(MIDDLE_LAYER):
        for j in range(OUTPUT_LAYER):
            w2[i][j] += SPEED * delta2[j] * df(s2[j]) * f1[i]

print('statistics = ', STATISTICS)
#np.save('w11', w1)
#np.save('w22', w2)

#plt.imshow(x_test[0], cmap=plt.cm.binary)
#plt.show()
