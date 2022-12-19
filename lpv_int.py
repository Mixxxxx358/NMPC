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

def lambda_simpson(x,u,nx,nu,ny,Jfx,Jfu,Jhx,stages):
    # FUNCTION LAMBDA_SIMPSON
    # Simpson rule integrator between 0 and 1 with chosen resolution (stages)
    # used to get A,B matrices symbolically to be used at gridpoints
    
    A = np.zeros([nx,nx])
    B = np.zeros([nx,nu])
    C = np.zeros([ny,nx])
    lambda0 = 0
    dlam = 1/stages

    for i in np.arange(stages):
        A = A + dlam*1/6*(Jfx(lambda0*x,lambda0*u) + 4*Jfx((lambda0+dlam/2)*x,(lambda0+dlam/2)*u) + Jfx((lambda0+dlam)*x,(lambda0+dlam)*u))
        B = B + dlam*1/6*(Jfu(lambda0*x,lambda0*u) + 4*Jfu((lambda0+dlam/2)*x,(lambda0+dlam/2)*u) + Jfu((lambda0+dlam)*x,(lambda0+dlam)*u))
        C = C + dlam*1/6*(Jhx(lambda0*x,lambda0*u) + 4*Jhx((lambda0+dlam/2)*x,(lambda0+dlam/2)*u) + Jhx((lambda0+dlam)*x,(lambda0+dlam)*u))
        lambda0 = lambda0 + dlam
            
    return A,B,C

def lambda_trap(x,u,nx,nu,ny,Jfx,Jfu,Jhx,stages):
    # FUNCTION LAMBDA_SIMPSON
    # Simpson rule integrator between 0 and 1 with chosen resolution (stages)
    # used to get A,B matrices symbolically to be used at gridpoints
    
    A = np.zeros([nx,nx])
    B = np.zeros([nx,nu])
    C = np.zeros([ny,nx])
    lambda0 = 0
    dlam = 1/stages

    for i in np.arange(stages):
        A = A + dlam*1/2*(Jfx(lambda0*x,lambda0*u) + Jfx((lambda0+dlam)*x,(lambda0+dlam)*u))
        B = B + dlam*1/2*(Jfu(lambda0*x,lambda0*u) + Jfu((lambda0+dlam)*x,(lambda0+dlam)*u))
        C = C + dlam*1/2*(Jhx(lambda0*x,lambda0*u) + Jhx((lambda0+dlam)*x,(lambda0+dlam)*u))
        lambda0 = lambda0 + dlam
            
    return A,B,C

