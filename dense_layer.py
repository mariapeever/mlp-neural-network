from layer import Layer
import numpy as np
from activation import Activation

class DenseLayer(Layer):
    # Dense
    
    def __init__(self, hidden_units, activation, initializer, bias, name, model):
      """
      @param HiddenUnits - Hidden units for weights
      @param ActivationFunc - Activation function (e.g. sigmoid)
      @param Initializer - weights
      @param Bias - bias
      @param InputShape - InputShape
      @param Name - Name
      @param Model - Model ref
      """
      super(DenseLayer, self).__init__(initializer, bias, name, model)
      self.hidden_units = hidden_units
      self.activation = activation if activation != None else Activation
          
    def build(self, inputs):
      
      self.kernel = self.initializer((self.hidden_units,) + inputs.X.shape)
      return self.forward_pass(inputs)

    def sum(self, X, w, b):
      output = np.zeros(w.shape[0], dtype=np.float32)
      for i in range(len(w)):
        sigma = self.sigma(X, w[i], b)
        output[i] = self.activation(self).activation_func(sigma, w[i], b) if self.activation != None else sigma
        
      return output

    def beta_func(self, *args):
      # Sum with activation
      if self.activation == None:
        return super().beta_func(*args)
      return self.activation(self).beta_func(*args)
        
    def beta_h_func(self, *args):
      # Sum with activation
      if self.activation == None:
        return super().beta_h_func(*args)
      return self.activation(self).beta_h_func(*args)
