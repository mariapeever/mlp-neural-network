import numpy as np
from activation import Activation

class Softmax(Activation):
  
    def __init__(self, layer):
        super(Softmax, self).__init__(layer)
        self.name = 'softmax'
        self.layer = layer
    
    def activation_func(self, s, w, b):
        # Sigmoid
        # print('sigmoid=1 / (1 + np.exp(-s))',1,'/(',1,'+',np.exp(-s),')=',(1 / (1 + np.exp(-s))))
        
        return (np.exp(s) / np.sum([np.exp(self.layer.sigma(a, w, b)) for a in self.layer.model.inputs]))
    
