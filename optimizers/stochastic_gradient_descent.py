import numpy as np

from trajectory_plotting_utils import plot_contour
from general_utils import data_points, grad_w,grad_b

#this code updates the parameters at evey point,
def do_stochastic_gradient_descent(init_w=-2,init_b=2,lr=0.1,max_epochs=1000):
    w, b = init_w,init_b
    eta = lr

    X,Y = data_points()
    trajectory = [(w, b)]

    for epoch in range(max_epochs):
        dw,db=0,0
        for x,y in zip(X,Y):
            dw = grad_w(w,b,x,y)
            db = grad_b(w,b,x,y)
           
            #Updating weights at every data point
            w = w - eta * dw
            b = b - eta * db
            
            trajectory.append((w, b))
    return trajectory

def main():
    trajectory=do_stochastic_gradient_descent()

    plot_contour(trajectory=trajectory,label="SGD")

if __name__ == "__main__":
    main()