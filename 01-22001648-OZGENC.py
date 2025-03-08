import random
import matplotlib.pyplot as plt
import numpy as np

x_points = [random.uniform(-2, 2) for _ in range(1000)]
y_points = [random.uniform(-2, 2) for _ in range(1000)]

def step_function(x):
    return 1 if x >= 0 else 0


def neuron1(x, y):
    weights = np.array([1, -1])
    bias = 1
    return step_function(x * weights[0] + y * weights[1] + bias)

def neuron2(x, y):
    weights = np.array([-1, -1])
    bias = 1
    return step_function(x * weights[0] + y * weights[1] + bias)

def neuron3(x):
    weights = np.array([-1])
    bias = 0
    return step_function(x * weights[0] + bias)

def neuron4(h0, h1, h2):
    weights = np.array([1, 1, -1])
    bias = -1.5
    return step_function(h0 * weights[0] + h1 * weights[1] + h2 * weights[2] + bias)

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('x')
plt.ylabel('y')
plt.title("1000 Random Points in the Square [-2, 2]^2")

for x, y in zip(x_points, y_points):
    result = neuron4(neuron1(x, y), neuron2(x, y), neuron3(x))
    if result == 1:
        plt.plot(x, y, 'ro')
    else:
        plt.plot(x, y, 'bo')


x_c = np.linspace(0, 2, 200)
y_c = 1-x_c

x_second = np.linspace(0, 0, 200)
y_second = np.linspace(-2, 1, 200)
plt.plot(x_c, y_c, 'k--', label='Decision Region: y <= 1 - x (for x > 0)')
plt.plot(x_second, y_second, 'k--')
plt.legend()


plt.show()
