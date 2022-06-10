class Input():
    # Input layer
    def __init__(self, X, name, model):
      """
      @param name
      @param model
      """
      self.name = name
      self.X = X
      self.input_shape = X.shape
      self.model = model
    
    def set_layer(self, layer):
      self.layer = layer
      
      