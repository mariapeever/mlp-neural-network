def mean_squared_error(labels, targets):
	sigma = 0
	for label, target in zip(labels, targets):
		sigma += (target - label) ** 2
	return sigma / len(labels)