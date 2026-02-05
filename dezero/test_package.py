import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import math


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


def testadd(x, y):
    z = (x + y) + (x + y)
    return z


def my_sin(x, threshold = 0.0001):
    y = 0
    for i in range(1000000):
        c = (-1)**i / math.factorial(2*i + 1)
        t = c * x**(2*i + 1)
        y += t
        if abs(t.data) < threshold:
            break
    return y


# x = Variable(np.array(1.0), name='x')
# y = Variable(np.array(1.0), name='y')
# z = goldstein(x, y)
# # z = testadd(x, y)
# z.backward()
# z.name = 'z'

# print(z.data, z.name)
# print(x.grad, y.grad)

# plot_dot_graph(z, verbose=False, to_file='goldstein.png')

x = Variable(np.array(np.pi/4), name='x')
y = my_sin(x)
y.backward()
y.name = 'y'
print(y.data)
print(x.grad)
plot_dot_graph(y, verbose=False, to_file='my_sin.png')
