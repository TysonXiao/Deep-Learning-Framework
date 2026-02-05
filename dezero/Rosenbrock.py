import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from dezero import Variable


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2)**2 + (x0 - 1)**2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.002
iters = 5000

x0_history = []
x1_history = []
y_history = []

for i in range(iters):
    print(x0.data, x1.data)
    #numpy数组需要copy()才能记录历史值，否则会存一堆引用，指向最后的值，导致list中是同样的值
    x0_history.append(x0.data.copy()) 
    x1_history.append(x1.data.copy())
    y = rosenbrock(x0, x1)
    y_history.append(y.data.copy()) # 记录当前高度

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad


def plot_rosenbrock_3d():
    # 1. 定义数据范围
    # x0 的范围选择 [-2, 2]
    # x1 的范围选择 [-1, 3] 以便更好地展示弯曲的山谷形状
    x0_vals = np.linspace(-0.5, 1.5, 200)
    x1_vals = np.linspace(-0.5, 2.5, 200)

    # 2. 创建网格数据 (Meshgrid)
    X0, X1 = np.meshgrid(x0_vals, x1_vals)

    # 3. 计算函数值 Y
    # 用户的公式: y = 100 * (x1 - x0**2)**2 + (x0 - 1)**2
    # 注意：在代码中，我们将计算结果赋值给 Z_vals 以便在3D图中表示高度
    Z_vals = rosenbrock(X0, X1)

    # 4. 创建 3D 图形设置
    fig = plt.figure(figsize=(12, 9))
    # add_subplot(111, projection='3d') 是创建3D坐标轴的关键
    ax = fig.add_subplot(111, projection='3d')

    # 5. 绘制 3D 曲面图
    # cmap=cm.viridis: 设置颜色映射，viridis 是一种对视觉友好的渐变色
    # rstride=1, cstride=1: 设置行和列的采样步长，越小越精细
    # alpha=0.8: 设置透明度
    surf = ax.plot_surface(X0, X1, Z_vals, cmap=cm.viridis, alpha=0.6,
                           linewidth=0, antialiased=True, 
                           rstride=5, cstride=5)

    # 绘制梯度下降路径 (关键点：使用记录的 history 列表)
    # zorder 确保线显示在曲面上方
    ax.plot(x0_history, x1_history, y_history, color='red', marker='o', 
            markersize=2, linewidth=1.5, label='Gradient Descent Path', zorder=10)

    # 标记起点和终点
    ax.scatter(x0_history[0], x1_history[0], y_history[0], color='blue', s=50, label='Start (0.0, 2.0)')
    ax.scatter(x0_history[-1], x1_history[-1], y_history[-1], color='green', s=50, label='End Point')

    # 6. 添加标签和标题
    # 使用 LaTeX 格式使数学符号更美观
    ax.set_xlabel('$x_0$ Axis', fontsize=12)
    ax.set_ylabel('$x_1$ Axis', fontsize=12)
    ax.set_zlabel('Function Value $y$', fontsize=12)
    ax.set_title('3D Visualization of Rosenbrock Function\n$y = 100(x_1 - x_0^2)^2 + (x0 - 1)^2$', fontsize=14)

    # 7. 添加颜色条 (Colorbar) 以指示数值大小
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Function Value')

    # 8. 调整观察视角 (可选)
    # elev: 仰角, azim: 方位角
    # 这个角度可以较好地观察到弯曲的山谷和位于 (1,1) 的最小值点
    ax.view_init(elev=35, azim=-135)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_rosenbrock_3d()