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

def do_nestrove_accelerated_gradient_descent(init_w,init_b,max_epochs):
    w,b,eta=init_w,init_b,1.0
    prev_v_w,prev_v_b,gamma=0,0,0.9
    for i in range(max_epochs):
        dw,db=0,0
        #do partial updates
        v_w=gamma*prev_v_w
        v_b=gamma*prev_v_b
        for x,y in zip(X,Y):
            #calculating gradients after partial update
            dw+=grad_w(w-v_w,b-v_b,x,y)
            db+=grad_b(w-v_w,b-v_b,x,y)

        #now do the full update 
        v_w=gamma*prev_v_w+eta*dw
        v_b=gamma*prev_v_b+eta*db

        w=w-v_w
        b=b-v_b

        prev_v_w=v_w
        prev_v_b=v_b
        