# https://github.com/geohot/tinygrad/blob/f4e0cb5945f9327e0b2d3aac54d94f84081d8676/tinygrad/tensor.py

import numpy as np 

# ====== TENSOR CLASS ========

class Tensor:
    def __init__(self, data, requires_grad = True, dtype=np.float32):
        self.data = np.array(data, dtype=dtype)
        self.grad = None
        self.requires_grad = requires_grad

        # This is used for the autograd graph (toposort)
        self._ctx = None 
        
    def __repr__(self): 
        return f"<Tensor {self.data} , shape={self.shape},  grad={self.grad}>"
    
    # === BACKPROP ==
    # toposort from tinygrad/micrograd
    def deepwalk(self):

        def _deepwalk(node, visited, nodes):
            visited.add(node)
            if node._ctx:
                [_deepwalk(i, visited, nodes) for i in node._ctx.parents if i not in visited]
                nodes.append(node)
            return nodes

        return _deepwalk(self, set(), [])
    
    # backprop
    def backward(self):
            print("need to backward pass...")
            return

    def apply(self):
        return 

    # === property method === 
    @property
    def shape(self): return self.data.shape
    
    # === creation helpers === 
    @classmethod
    def linear_layer(cls, ins, outs): return cls(1/np.sqrt(ins*outs) * np.random.uniform(-1,1,size=(ins,outs)))

# ====== FUNCTION CLASS ========

class Function:
    def __init__(self, *tensors:Tensor):
        self.parents = tensors
        return
