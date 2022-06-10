from input import Input

class Output(Input):
    # Input layer
    def __init__(self, X, name, model):
      """
      @param X
      @param name
      @param model
      """
      super(Output, self).__init__(X, name, model)
