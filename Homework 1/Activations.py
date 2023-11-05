import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x, keepdims=True))
    return e_x / e_x.sum(keepdims=True)