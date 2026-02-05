import numpy as np
import weakref
import contextlib

import dezero


# 配置类
class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


# 变量类
class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self):
        return dezero.functions.transpose(self)
    
    @property
    def T(self):
        return self.transpose()

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)
    
    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            func = funcs.pop()
            gys = [y().grad for y in func.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = func.backward(*gys)   
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(func.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

                    if not retain_grad:
                        for y in func.outputs:
                            y().grad = None

    def cleargrad(self):
        self.grad = None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        return 'Variable({})'.format(str(self.data).replace('\n', '\n' + ' ' * 9))


# 函数类及其辅助函数
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# 辅助函数：将对象转换为Variable

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [input.data for input in inputs]  #这里确保xs的元素是np.array
        ys = self.forward(*xs)  #这里确保了传入前向计算的是np.array
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            self.outputs = [weakref.ref(output) for output in outputs]
            self.inputs = inputs
            for output in outputs:
                output.set_creator(self)

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError("forward() is not implemented")

    def backward(self, gy):
        raise NotImplementedError("backward() is not implemented")


# 四则运算类
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 + x1

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  #如果进行了广播，需要对梯度进行求和，转换为原始输入的形状
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 - x1

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:  #如果进行了广播，需要对梯度进行求和，转换为原始输入的形状
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy * x1, gy * x0
        if self.x0_shape != self.x1_shape:  #如果进行了广播，需要对梯度进行求和，转换为原始输入的形状
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


# 除法类
class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy / x1, -gy * x0 / (x1 ** 2)
        if self.x0_shape != self.x1_shape:  #如果进行了广播，需要对梯度进行求和，转换为原始输入的形状
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * (x ** (c - 1)) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


# 运算符重载设置
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow