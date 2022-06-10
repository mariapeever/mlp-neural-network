from dense_layer import DenseLayer 

def Dense(hidden_units, activation, initializer, bias, name, model):
    # Dense funct
    Dense = DenseLayer(hidden_units, activation, initializer, bias, name, model)
    def forward_pass(inputs):
        return Dense.forward_pass(inputs)
    return forward_pass, Dense