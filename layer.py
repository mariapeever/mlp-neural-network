import numpy as np
from output import Output


class Layer():
    def __init__(self, initializer, bias, name, model):
        """
        @param initializer - weights
        @param bias - bias
        @param input_shape - the input shape
        @param name - name
        @param model - model ref
        """
        self.activation = None
        self.initializer = initializer
        self.kernel = None
        self.bias = bias
        self.model = model
        self.name = name
        self.beta = None
        self.X = None

    def forward_pass(self, X):
        # ForwardPass
        # print('forward pass')
        self.input = X
        if self.model.epoch == 0:
            self.model.connections[self.name] = X.name
        # Weighted Sum
        s = self.sum(X.X, self.kernel, self.bias)
        output = Output(s, self.name, self.model)
        output.set_layer(self)
        
        self.X = output
        return output
          
    def sum(self, X, w, b):
        # Sum without activation
        xs
        output = np.zeros(w.shape[0], dtype=np.float32)
        for i in range(len(w)):
          output[i] = self.sigma(X, w[i], b)
        return output
      
    def sigma(self, X, w, b):
        # Weighted sum
        s = b
        for i in range(len(X)):
            # print('s=s+w*x=', s,'+',w[i],'*',X[i],'=',(s + w[i] * X[i]), (s + np.dot(w[i], X[i])))
            s = s + w[i] * X[i]

        return s
