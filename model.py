from datetime import datetime
import h5py
from dense import Dense
from input import Input
from sigmoid import Sigmoid
from softmax import Softmax
from initializers import initializers
from activation_check import activation_check
import os
import numpy as np
from score import mean_squared_error 

class Model():
    
    def __init__(self, input_shape):

        self.input_shape = input_shape
        self.connections = {}
        self.layers = {}
        self.weights = {}
        self.learning_rate = 1
        self.beta = None
        self.idx = 0
        self.epoch = 0
        self.bias = 0

        # fw_pass_d1, dense1 = Dense(64, Softmax, initializers('random'), self.bias, 'dense1', self)
        fw_pass_d2, dense2 = Dense(1, Sigmoid, initializers('random'), self.bias, 'dense2', self)
      
        # self.layers['dense1'] = (fw_pass_d1, dense1)
        self.layers['dense2'] = (fw_pass_d2, dense2)

    def compile(self, learning_rate, bias):
        self.bias = bias
        self.learning_rate = learning_rate
    
    def fit(self, inputs, labels, epochs):
        self.inputs = inputs
        self.X = Input(inputs[0], 'x1', self)

        self.build(self.X)

        self.targets = labels
        self.epochs = epochs
        
        for i in range(epochs):
            self.epoch = i
            
            for j in range(self.inputs.shape[0]):
                
                self.X = Input(inputs[j], 'x1', self)
                self.y = labels[j]
                self.backpropagation()

                print('Epoch: {} - Input: {} - Outputs: {} - Beta: {}'.format(i, j, self.X.X[0], self.beta[0]))
            
            if i == self.epochs-1:
                timestamp = datetime.now()
                timestamp = timestamp.strftime("%Y-%m-%d-%H-%M-%S")
                t = timestamp
                self.save_path = os.path.join('', 'epoch-{}-weights-{}.h5'.format(self.epoch, t))
                f = h5py.File(self.save_path, 'w')
                print('Weights saved to path: {}.'.format(self.save_path))
                data = self.extract_weights()
                for layer in data.keys():
                  f.create_dataset(layer, data=data[layer])
                
                f.close()

            print('Epoch: {} - Beta: {}'.format(i, self.beta[0]))
       
    def build(self, x):
        for _, layer in list(self.layers.values()):
          x = layer.build(x)

    def backpropagation(self):

        # Backpropagation
        # Forward pass
        self.forward_pass()
      
        # Backward pass
        self.backward_pass()
        
    def forward_pass(self):
        # Forward pass
        # dense1 = self.layers['dense1'][0]
        dense2 = self.layers['dense2'][0]
        
        # Call
        # x = dense1(self.X)
        x = dense2(self.X)
        self.X = x

    def backward_pass(self):
        # BackwardPass
  
        y = self.y
        X = self.X

        keys = list(self.layers.keys())
        keys.reverse()
        # Connections
        connections = self.connections
        
        # Output layer
        
        key = keys[0]
        layer = self.layers[key]
        # Activation check
        try:
            assert(layer[1].activation == None or activation_check(layer[1].activation(layer[1]).name))
        except:
            print('There is currently no support for {} activation. Please use Sigmoid or None.'.format(layer[1].activation().name))
            beta = None
            layers = {}
            return beta, layers

        # BetaOut
        beta_out = layer[1].beta_func(layer[1].X.X, y)
        layer[1].beta = beta_out
        self.beta = beta_out
        # Recursive backward pass for MLPs
        layers = self.rec_bkw_pass(self.layers, keys, key, layer)
        keys.reverse()
        self.layers = dict([(layers[key][1].name, layers[key]) for key in keys])
        self.layers[key] = layer

    def rec_bkw_pass(self, layers, keys, key, layer):
        """
        Recursive backward pass for MLPs
        @param Layers - All layers as a cell struct
        @param Keys - Keys for all layers
        @param Key - Current key
        @param Layer - Current layer
        """
        # increment idx
        X = self.X
        
        # connections
        connections = self.connections
        # lr
        lr = self.learning_rate
        # Init kernel (weights)
        kernel = layer[1].kernel
        # Inputs to current layer as cells
        inputs = connections[key].split(' ')
        
        key = inputs[0]
        for i in range(len(kernel)):
            
            if key in keys:
              input = layers[key] 
              out = input[1].X.X
            else:
              out = X.X
            kernel[i] = self.update_weights(lr, layer[1].beta[i], kernel[i], out)
            if len(inputs) == 0:
                continue
            
            # Activation check
            try:
                assert(layer[1].activation == None or activation_check(layer[1].activation(layer[1]).name))
            except AssertionError:
                print('There is currently no support for {} activation. Please use Sigmoid.'.format(layer[1].activation().name))
                return layers
                
            # Beta - hidden
            if key in keys:
              beta_h = input[1].beta_h_func(layer[1].X.X[i], layer[1].beta[i], kernel[i]) 
              input[1].beta = beta_h
              
              # Recursive call
              layers = self.rec_bkw_pass(layers, keys, key, input) 

        layer[1].kernel = kernel
        layers[layer[1].name] = layer
        return layers
    def update_weights(self, lr, beta, w, X):
        # Update weights
        return  w + lr * beta * X 
        
    def load_weights(self, file):
        f = h5py.File(file, 'r')

        for key in f.keys():
          self.layers[key][1].kernel = f[key]

        print('Weights loaded successfully.')

    def extract_weights(self):
        layers = self.layers
        weights = {}
        
        for key in list(layers.keys()):
            weights[key] = np.array(layers[key][1].kernel, dtype=np.float32)
        return weights

    # def evaluate(self, test_data, test_labels):
    #     self.X = Input(test_data, 'x1', self)
    #     self.forward_pass()
    #     print('mse:', mean_squared_error(test_labels, self.X.X))
        
    def predict(self, x):

        self.X = Input(x, 'x1', self)
        self.forward_pass()
        return self.X.X

    def summary(self): # summary
        # Print Summary
        self.X = Input(np.zeros((self.input_shape[1:])), 'x1', self)
        self.build(self.X)
        self.forward_pass()
        outputs= self.X
        layers = self.layers 
        keys = list(layers.keys())
        connections = self.connections
        
        summary = ''
        for i in range(80):
            summary += '='
        summary += '\n'
        summary += 'Name                              Inputs\n' 
        for i in range(80):
            summary += '='
        
        summary += '\n'
        
        for key in keys:
            layer = layers[key][1]
            inputs = connections[layer.name]
            inputs = inputs.split(' ')
            
            summary += '{}                            {}\n'.format(layer.name, inputs[0])
            if i < len(keys):
                for j in range(80):
                    summary += '-'
                
            summary += '\n'
        for i in range(80):
            summary += '='
        
        print(summary)
     