#vannila gradient descent
import numpy as np

from general_utils import data_points, grad_w,grad_b,error
from trajectory_plotting_utils import plot_error_contour



def do_gragient_descent():
    #trying to converge gradient descent for the below two point i.e. 
    # GD will optimized it self to reach convergence while trying to include this points in it trajectory. 
    # Data points
    X,Y=data_points()

    w,b,eta,max_epochs=-2,-2,1.0,1000

    trajectory = [(w, b)]

    for i in range(max_epochs):
        dw,db=0,0
        for x,y in zip(X,Y):
            dw+=grad_w(w,b,x,y)
            db+=grad_b(w,b,x,y)
            
        #updating parameters after going thorugh each and every data point
        w=w-eta*dw
        b=b-eta*db
        trajectory.append((w,b))

    return trajectory

gradient_descent_trajectory=do_gragient_descent()

plot_error_contour(error_function=error,trajectory=gradient_descent_trajectory,algo_name="Gradient Descent")