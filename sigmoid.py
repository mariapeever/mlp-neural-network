import numpy as np
from activation import Activation

class Sigmoid(Activation):
  
    def __init__(self, layer=None):
        super(Sigmoid, self).__init__(layer)
        self.name = 'sigmoid'
    
    def activation_func(self, s, w=None, b=None):
        # Sigmoid
        # print('sigmoid=1 / (1 + np.exp(-s))',1,'/(',1,'+',np.exp(-s),')=',(1 / (1 + np.exp(-s))))
        return (1 / (1 + np.exp(-s)))
        
    def beta_func(self, out, y):
        # Sigmoid
        # print('beta=out*(1-out)*(y-out)=', out,'*','(',1,'-',out,')*','(',y,'-',out,')=',out * (1 - out) * (y - out), np.dot(np.dot(out, (1 - out)), (y - out))) 
        return out * (1 - out) * (y - out)
        # return np.dot(np.dot(out, (1 - out)), (y - out))

    def beta_h_func(self, out, beta, w):
        # Sigmoid
         
        err = np.zeros(w.shape, dtype=np.float32)
        for i in range(w.shape[0]):
          # print('beta=out*(1-out)*(beta*w)=', out,'*','(',1,'-',out,')*','(',beta,'*',w[i],')=', (out*(1-out)*(beta*w[i])), np.dot(np.dot(out, (1 - out)),(np.dot(beta, w[i]))), np.float32(out) * np.float32((1 - out)) * np.float32((beta * w[i])), np.float32(out) * (np.float32(1) - np.float32(out)) * (np.float32(beta) * np.float32(w[i])))
          # print('out', np.float32(out) * (np.float32(1) - np.float32(out)) * (np.float32(beta) * np.float32(w[i])))
          err[i] = out * (1 - out) * (beta * w[i])
          # err[i] = np.float32(out) * np.float32((1 - out)) * np.float32((beta * w[i]))
          # err[i] = np.dot(np.dot(out, (1 - out)),(np.dot(beta, w[i])))
        # print('err', err)
        return err
