import numpy as np

def initializers(name):
    # Initializers - custom weights
    def random(input_shape):
      return np.random.rand(*input_shape)
    
    def ones(input_shape):
      return np.ones(input_shape, dtype=np.float32)
    
    def zeros(input_shape):
      return np.zeros(input_shape, dtype=np.float32)
      
    switcher = {
        'random': random,
        'ones': ones,
        'zeros': zeros }
    try:
      assert(name in list(switcher.keys()))
      return switcher[name]
    except:
      print('Kernel is required.')

    