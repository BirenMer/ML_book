import math
import numpy as np

#datapoints (Need dataset with 1000 points and need at lest 80% as 0 only X points)

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

def do_adam(init_w,init_b,max_epochs):
    w_b_dw_db=[(init_w,init_b,0,0)]
    
    w_history,b_history,error_history=[],[],[]
    
    w,b,eta,mini_batch_size,num_points_seen=init_w,init_b,0.1,10,0

    m_w, m_b, v_w, v_b, eps, beta1, beta2=0,0,0,0,1e-8,0.9,0.999

    for i in range(max_epochs):
        dw,db=0,0
        for x,y in zip(X,Y):
            dw+=grad_w(w,b,x,y)
            db+=grad_b(w,b,x,y)
        m_w=beta1*m_w+(1-beta1)*dw
        m_b=beta1*m_b+(1-beta1)*db

        v_w=beta2*v_w+(1-beta2)*dw**2
        v_b=beta2*v_b+(1-beta2)*db**2

        m_w=m_w/(1-math.pow(beta1,i+1))
        m_b=m_b/(1-math.pow(beta1,i+1))

        v_w=v_w/(1-math.pow(beta2,i+1))
        v_b=v_b/(1-math.pow(beta2,i+1))

        w=w-(eta/np.sqrt(v_w+eps))*m_w
        b=b-(eta/np.sqrt(v_b+eps))*m_b