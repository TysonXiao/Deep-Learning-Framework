from dezero.core import Parameter
import dezero.functions as F
import numpy as np
import weakref


class Layer:
    def __init__(self):
        self._params = set()
        
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            if isinstance(self.__dict__[name], Layer):
                yield from self.__dict__[name].params()
            else:
                yield self.__dict__[name]

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()


class Linear(Layer):
    def __init__(self, out_size, in_size=None, nobias=False, dtype=np.float32):
        super().__init__()

        self.in_size, self.out_size = in_size, out_size
        self.dtype = dtype
        self.W = Parameter(None, name='W')

        if self.in_size is not None:
            W_data = np.random.randn(self.in_size, self.out_size).astype(dtype) * np.sqrt(1 / self.in_size)
            self.W.data = W_data
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(self.out_size, dtype=dtype), name='b')
            
    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            W_data = np.random.randn(self.in_size, self.out_size).astype(self.dtype) * np.sqrt(1 / self.in_size)
            self.W.data = W_data

        y = F.linear_simple(x, self.W, self.b)
        return y
