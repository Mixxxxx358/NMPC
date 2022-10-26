import numpy as np
from casadi import *

def par_lpv_int(x,nx,u,nu,Jfx,Jfu,dlam,stages,L):
    # FUNCTION LPV_INT
    # RK4 integrator with chosen resolution and internal stages
    # used to get A,B matrices symbolically to be used at gridpoints
    
    A = np.zeros([nx,nx])
    #B = np.zeros([nx,nu])

    Lx = np.kron(L,x)
    Lu = L*u

    Fx = Jfx.map(stages)
    kx = Fx(Lx.T, Lu.T)
    Fu = Jfu.map(stages)
    ku = Fu(Lx.T, Lu.T)

    A = np.zeros((2,2))
    A[:,0] = reshape(sum2(kx[:,0:stages*2:2]), (1,2)) # k1
    A[:,1] = reshape(sum2(kx[:,1:stages*2:2]), (1,2)) # k1
    A[:,0] = A[:,0] + 4*reshape(sum2(kx[:,stages*2:stages*4:2]), (1,2)) # k2
    A[:,1] = A[:,1] + 4*reshape(sum2(kx[:,stages*2+1:stages*4:2]), (1,2)) # k2
    A[:,0] = A[:,0] + reshape(sum2(kx[:,stages*4:stages*6:2]), (1,2)) # k4
    A[:,1] = A[:,1] + reshape(sum2(kx[:,stages*4+1:stages*6:2]), (1,2)) # k4
    A = 1/6*dlam*A

    B = np.zeros((2,1))
    B[:,0] = reshape(sum2(ku[:,0:stages:1]), (1,2)) # k1
    B[:,0] = B[:,0] + 4*reshape(sum2(ku[:,stages:2*stages:1]), (1,2)) # k2
    B[:,0] = B[:,0] + reshape(sum2(ku[:,2*stages:3*stages:1]), (1,2)) # k4
    B = 1/6*dlam*B
            
    return A, B