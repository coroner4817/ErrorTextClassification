import numpy as np


def sigmoid(x):
    x = 1 / (np.exp(-x) + 1)
    return x


def sigmoid_grad(f):
    f = f * (1 - f)
    return f


def softmax(x):
    if len(x.shape) > 1:
        x_max = np.max(x, axis=1)
        x -= np.reshape(x_max, (x.shape[0], 1))
        x = np.exp(x)
        exp_sum = np.sum(x, axis=1)
        x /= exp_sum.reshape((x.shape[0], 1))
    else:
        x_max = np.max(x)
        x -= x_max
        exp_sum = np.sum(np.exp(x))
        x = np.exp(x) / exp_sum

    return x
