
#WIP code
import numpy as np
#Sigmoid function f(x)

#datapoints
X=[0.5,2.5]
Y=[0.2,0.0]


def f(w,b,x):
    return 1.0/(1.0 * np.exp(-(w*x+b)))


#Error function
def error(w,b):
    err=0.0
    for x,y in zip(X,Y):
        fx=f(w,b,x)
        err+=0.5*(fx-y)**2
    return err


def grad_b(w,b,x,y):
    fx=f(w,b,x)
    return (fx-y) * (fx) * (1-fx)

def grad_w(w,b,x,y):
    fx=f(w,b,x)
    return (fx-y) * (fx) * (1-fx) * x

def do_line_search_gradient_descent(init_w,init_b,max_epochs):
    
    w,b,etas = init_w,init_b,[0.1,0.5,1.0,5.0,10.0]
    
    prev_v_w,prev_v_b,gamma=0,0,0.9
    
    for i in range(max_epochs):
        dw,db=0,0
        for x,y in zip(X,Y):
            dw+=grad_w(w,b,x,y)
            db+=grad_b(w,b,x,y)
        min_error=1000 #some large value
        best_w,best_b=w,b
        for eta in etas:
            tmp_w=w-eta*dw
            tmp_b=b-eta*db
            if error(tmp_w,tmp_b)<min_error:
                best_w=tmp_w
                best_b=tmp_b
                min_error=error(tmp_w,tmp_b)
        w,b=best_w,best_b
        
            