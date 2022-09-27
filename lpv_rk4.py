def lpv_rk4(x,u,A,B,dt, correction):
    # function LPV_RK4
    # used for discretized simulation of next state variable
    # using A and B matrices obtained by embedding
    k1 = dt*(A@x+B@u + correction)
    k2 = dt*(A@(x+1*k1/2)+B@u + correction)
    k3 = dt*(A@(x+1*k2/2)+B@u + correction)
    k4 = dt*(A@(x+1*k3)+B@u + correction)
    x_next = x + 1/6*1*(k1+2*k2+2*k3+k4)
    
    return x_next