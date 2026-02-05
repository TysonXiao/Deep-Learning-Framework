import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph

# def f(x):
#     y = x ** 4 - 2 * x ** 2
#     return y


# x = Variable(np.array(2.0))
# iters = 10

# for i in range(iters):
#     print(i, x.data)
#     y = f(x)
#     x.cleargrad()
#     y.backward(create_graph=True)

#     gx = x.grad
#     x.cleargrad()
#     gx.backward()
#     gx2 = x.grad

#     x.data -= gx.data / gx2.data

# x = Variable(np.array(1.0))
# y = sin(x)
# y.backward(create_graph=True)
# print(x.grad)

# for i in range(4):
#     gx = x.grad
#     x.cleargrad()
#     gx.backward(create_graph=True)
#     print(x.grad)

# #绘制图形
# import matplotlib.pyplot as plt
# x = Variable(np.linspace(-7, 7, 100))
# y = sin(x)
# y.backward(create_graph=True)
 
# logs = [y.data]

# for i in range(3):
#     logs.append(x.grad.data)
#     gx = x.grad
#     x.cleargrad()
#     gx.backward(create_graph=True)
    
# labels = ['y=sin(x)', 'y`', 'y``', 'y```']
# for i, v in enumerate(logs):
#     plt.plot(x.data, v, label=labels[i])
# plt.legend()
# plt.show()

# x = Variable(np.array(1.0))
# y = tanh(x)
# x.name = 'x'
# y.name = 'y'
# y.backward(create_graph=True)

# iters = 1
# for i in range(iters):
#     print(i, x.grad)
#     gx = x.grad
#     x.cleargrad()
#     gx.backward(create_graph=True)

# gx = x.grad
# gx.name = "gx" + str(iters + 1)
# plot_dot_graph(gx, verbose=False, to_file='tanh.png')


# x0 = Variable(np.random.randn(2,3,4))
# x1 = Variable(np.array([2]))
# y = x0 / x1
# y.backward(create_graph=True)
# print(y)
# print(x0.grad)   
# print(x1.grad)         


import numpy as np
import timeit

# np.show_config()

# 准备数据
x = np.random.rand(800000)
y = np.random.rand(800000)

# 定义测试语句
stmt = "np.dot(x, y)"

# 准备环境（避免在循环中重复创建数组）
setup = "import numpy as np; from __main__ import x, y"

number = 1000
# 运行 1,000,000 次，取总时间
total_time = timeit.timeit(stmt, setup=setup, number=number)

print(f"单次 dot 运算平均耗时: {total_time / number * 1e6:.3f} 微秒 (us)")
