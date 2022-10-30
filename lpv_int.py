import numpy as np
from casadi import *

def lpv_int(x,nx,u,nu,Jfx,Jfu,dlam,stages):
    # FUNCTION LPV_INT
    # RK4 integrator with chosen resolution and internal stages
    # used to get A,B matrices symbolically to be used at gridpoints
    
    A = np.zeros([nx,nx])
    B = np.zeros([nx,nu])
    #C = np.zeros([ny,nx])
    lam = 0

    for i in np.arange(stages):
        k1 = Jfx(lam*x,lam*u)
        j1 = Jfu(lam*x,lam*u)
        #l1 = Jhx(lam*x,lam*u)

        k2 = Jfx((lam+dlam/2)*x,(lam+dlam/2)*u)
        j2 = Jfu((lam+dlam/2)*x,(lam+dlam/2)*u)
        #l2 = Jhx((lam+dlam/2)*x,(lam+dlam/2)*u)

        k4 = Jfx((lam+dlam)*x,(lam+dlam)*u)
        j4 = Jfu((lam+dlam)*x,(lam+dlam)*u)
        #l4 = Jhx((lam+dlam)*x,(lam+dlam)*u)

        A = A + 1/6*dlam*(k1 + 4*k2 + k4)
        B = B + 1/6*dlam*(j1 + 4*j2 + j4)
        #C = C + 1/6*dlam*(l1 + 4*l2 + l4)
        lam = lam + dlam
            
    return A,B

def lpv_int_C(x,nx,u,nu,ny,Jfx,Jfu,Jhx,dlam,stages):
    # FUNCTION LPV_INT
    # RK4 integrator with chosen resolution and internal stages
    # used to get A,B matrices symbolically to be used at gridpoints
    
    A = np.zeros([nx,nx])
    B = np.zeros([nx,nu])
    C = np.zeros([ny,nx])
    lam = 0

    for i in np.arange(stages):
        k1 = Jfx(lam*x,lam*u)
        j1 = Jfu(lam*x,lam*u)
        l1 = Jhx(lam*x,lam*u)

        k2 = Jfx((lam+dlam/2)*x,(lam+dlam/2)*u)
        j2 = Jfu((lam+dlam/2)*x,(lam+dlam/2)*u)
        l2 = Jhx((lam+dlam/2)*x,(lam+dlam/2)*u)

        k4 = Jfx((lam+dlam)*x,(lam+dlam)*u)
        j4 = Jfu((lam+dlam)*x,(lam+dlam)*u)
        l4 = Jhx((lam+dlam)*x,(lam+dlam)*u)

        A = A + 1/6*dlam*(k1 + 4*k2 + k4)
        B = B + 1/6*dlam*(j1 + 4*j2 + j4)
        C = C + 1/6*dlam*(l1 + 4*l2 + l4)
        lam = lam + dlam
            
    return A,B,C
