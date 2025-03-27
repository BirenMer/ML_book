import numpy as np

#Blow data points are common for all algo and will be used throughout
def data_points():
    X = [0.5, 2.5]
    Y = [0.2, 0.0]
    return X,Y

# Sigmoid function
def f(w, b, x):
    return 1.0 / (1.0 + np.exp(-(w * x + b)))

# Error function
def error(w, b):
    X,Y=data_points()
    err = 0.0
    for x, y in zip(X, Y):
        fx = f(w, b, x)
        err += 0.5 * (fx - y) ** 2
    return err

# Gradients
def grad_b(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx)

def grad_w(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx) * x
