import numpy as np

from trajectory_plotting_utils import plot_contour
from general_utils import data_points, grad_w, grad_b, error

def do_line_search_gradient_descent(init_w=-2, init_b=2, max_epochs=1000):
    w, b = init_w, init_b
    etas = [0.1, 0.5, 1.0, 5.0, 10.0]
    X, Y = data_points()
    trajectory = [(w, b)]

    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        dw /= len(X)
        db /= len(X)

        min_error = float('inf')
        best_w, best_b = w, b

        for eta in etas:
            tmp_w = w - eta * dw
            tmp_b = b - eta * db
            current_error = error(tmp_w, tmp_b)
            if current_error < min_error:
                min_error = current_error
                best_w, best_b = tmp_w, tmp_b

        w, b = best_w, best_b
        trajectory.append((w, b))

    return trajectory

trajectory=do_line_search_gradient_descent()

plot_contour(trajectory=trajectory,label="Line Search GD")