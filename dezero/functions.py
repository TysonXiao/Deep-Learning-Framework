import numpy as np
from dezero.core import Function
from dezero.core import as_variable
from dezero import utils


class Sin(Function):
    def forward(self, x):
        return np.sin(x)
    
    def backward(self, gx):
        x, = self.inputs
        return gx * cos(x)

def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)
    
    def backward(self, gx):
        x, = self.inputs
        return gx * -sin(x)

def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, gy):
        y = self.outputs[0]()
        return gy * (1 - y**2)

def tanh(x):
    return Tanh()(x)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape)  #利用了numpy的reshape方法
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)


class Transpose(Function):
    def forward(self, x):
        return np.transpose(x)
    
    def backward(self, gy):
        return transpose(gy)

def transpose(x):
    return Transpose()(x)


class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape  
    
    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)
    
    def backward(self, gy):
        return sum_to(gy, self.x_shape)

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)
    

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        return utils.sum_to(x, self.shape)
    
    def backward(self, gy):
        return broadcast_to(gy, self.x_shape)

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
