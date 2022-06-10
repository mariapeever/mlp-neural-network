def activation_check(activation):
  if activation in ['sigmoid', 'softmax', None]:
    return True
  return False