from turtle import mode
import numpy as np
from numpy.linalg import LinAlgError
from dezero import Variable, MLP
from dezero.NN_layer import loss
import dezero.functions as F
import dezero.layers as L
from dezero.layers import Layer

# np.random.seed(0)
# x = np.random.randn(100, 1)
# y = np.sin(2 * np.pi * x) + np.random.randn(100, 1) 

# l1 = L.Linear(10)
# l2 = L.Linear(1)

# def predict(x):
#     y = l1(x)
#     y = F.sigmoid_simple(y)
#     y = l2(y)
#     return y

# lr = 0.2
# iters = 10000
# hidden_size = 10

# for i in range(iters):
#     y_pred = predict(x)
#     loss = F.mean_squared_error(y_pred, y)
    
#     l1.cleargrads()
#     l2.cleargrads()
#     loss.backward()
    
#     for l in [l1, l2]:
#         for p in l.params():
#             p.data -= lr * p.grad.data

#     if i % 1000 == 0:
#         print(loss)

##layer in layer

# model = Layer()
# model.l1 = L.Linear(10)
# model.l2 = L.Linear(3)

# def predict(model, x):
#     y = model.l1(x)
#     y = F.sigmoid_simple(y)
#     y = model.l2(y)
#     return y

# for p in model.params():
#     print(p)

# model.cleargrads()

# class TwoLayerNet(Model):
#     def __init__(self, hidden_size, out_size):
#         super().__init__()
#         self.l1 = L.Linear(hidden_size)
#         self.l2 = L.Linear(out_size)

#     def forward(self, x):
#         y = self.l1(x)
#         y = F.sigmoid_simple(y)
#         y = self.l2(y)
#         return y

# model = MLP([hidden_size, 1])

# for i in range(iters):
#     y_pred = model(x)
#     loss = F.mean_squared_error(y_pred, y)
    
#     model.cleargrads()
#     loss.backward()
    
#     for p in model.params():
#         p.data -= lr * p.grad.data

#     if i % 1000 == 0:
#         print(loss)

model = MLP([10, 3])
x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
loss = F.soft_cross_entropy_simple(y, t)
print(loss)