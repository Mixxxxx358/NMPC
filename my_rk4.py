def my_rk4(x,u,f,dt):
    # function MY_RK4
    # used for a discretized simulation of next time-step
    # of the state variables using f(x,u)
    
    k1 = dt*f(x,u)
    k2 = dt*f(x+1*k1/2,u)
    k3 = dt*f(x+1*k2/2,u)
    k4 = dt*f(x+1*k3,u)
    x_next = x + 1/6*1*(k1+2*k2+2*k3+k4)
    
    return x_next