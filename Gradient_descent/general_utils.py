import numpy as np

# Generate data points
def data_points():
    X = np.linspace(-4, 4, 100)
    Y = 1 / (1 + np.exp(-X))  # Sigmoid function as ground truth
    return X, Y

def f(w, b, x):
    z = w * x + b
    # Use a numerically stable sigmoid computation
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

# Error function
def error(w, b):
    return 0.1 * w**2 + b**2

# Gradients
def grad_b(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx)

def grad_w(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx) * x
