import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'test'))

from simplegrad.tensor import Tensor

class BobNet():
    def __init__(self):
        self.l1 = Tensor.linear_layer(28*28, 128)
        self.l2 = Tensor.linear_layer(128, 10)


    def forward(self, x):
        print('the forward pass')
        return  


if __name__ == '__main__':

    model = BobNet()
    print(model.l1)

    