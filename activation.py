import numpy as np

class Activation():
  
    def __init__(self, layer=None):
        self.name = None
    
    def activation_func(self, s, w=None, b=None):
        # Sigmoid
        # print('sigmoid=1 / (1 + np.exp(-s))',1,'/(',1,'+',np.exp(-s),')=',(1 / (1 + np.exp(-s))))
        return s
        
    def beta_func(self, out, y):
        # Sigmoid
        # print('beta=out*(1-out)*(y-out)=', out,'*','(',1,'-',out,')*','(',y,'-',out,')=',out * (1 - out) * (y - out), np.dot(np.dot(out, (1 - out)), (y - out))) 
        return y - out
        # return np.dot(np.dot(out, (1 - out)), (y - out))

    def beta_h_func(self, out, beta, w):
        # Sigmoid
         
        err = np.zeros(w.shape, dtype=np.float32)
        # print('beta_h_func kernel', w)
        for i in range(w.shape[0]):
          # print('beta=out*beta*w=', out,'*',beta,'*',w[i],'=',(out * beta * w[i]), np.dot(out, np.dot(beta, w[i])))
          err[i] = out * beta * w[i]
          # err[i] = np.dot(out, np.dot(beta, w[i]))
        return err
