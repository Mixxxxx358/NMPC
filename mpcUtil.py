import numpy as np
import itertools
import qpsolvers as qp
from casadi import *

def CasADiExp(ss_enc, x, u):
    n_hidden_layers = ss_enc.f_n_hidden_layers
    nu = ss_enc.nu if ss_enc.nu is not None else 1

    params = {}
    for name, param in ss_enc.fn.named_parameters():
        params[name] = param.detach().numpy()
    params_list = list(params.values())
    
    xu = vertcat(x,u)

    temp_nn = xu
    for i in range(n_hidden_layers):
        W_NL = params_list[2+i*2]
        b_NL = params_list[3+i*2]
        temp_nn = mtimes(W_NL, temp_nn)+b_NL
        temp_nn = tanh(temp_nn)
    W_NL = params_list[2+n_hidden_layers*2]
    b_NL = params_list[3+n_hidden_layers*2]
    nn_NL = mtimes(W_NL, temp_nn)+b_NL

    W_Lin = params_list[0]
    b_Lin = params_list[1]
    nn_Lin = mtimes(W_Lin,xu) + b_Lin

    return nn_NL + nn_Lin

def CasADiHn(ss_enc, x):
    n_hidden_layers = 2#ss_enc.h_n_hidden_layers

    params = {}
    for name, param in ss_enc.hn.named_parameters():
        params[name] = param.detach().numpy()
    params_list = list(params.values())

    temp_nn = x
    for i in range(n_hidden_layers):
        W_NL = params_list[2+i*2]
        b_NL = params_list[3+i*2]
        temp_nn = mtimes(W_NL, temp_nn)+b_NL
        temp_nn = tanh(temp_nn)
    W_NL = params_list[2+n_hidden_layers*2]
    b_NL = params_list[3+n_hidden_layers*2]
    nn_NL = mtimes(W_NL, temp_nn)+b_NL

    W_Lin = params_list[0]
    b_Lin = params_list[1]
    nn_Lin = mtimes(W_Lin,x) + b_Lin

    return nn_NL + nn_Lin

def CasADiFn(ss_enc, x, u):
    n_hidden_layers = 2#ss_enc.f_n_hidden_layers
    nu = ss_enc.nu if ss_enc.nu is not None else 1

    params = {}
    for name, param in ss_enc.fn.named_parameters():
        params[name] = param.detach().numpy()
    params_list = list(params.values())
    
    xu = vertcat(x,u)

    temp_nn = xu
    for i in range(n_hidden_layers):
        W_NL = params_list[2+i*2]
        b_NL = params_list[3+i*2]
        temp_nn = mtimes(W_NL, temp_nn)+b_NL
        temp_nn = tanh(temp_nn)
    W_NL = params_list[2+n_hidden_layers*2]
    b_NL = params_list[3+n_hidden_layers*2]
    nn_NL = mtimes(W_NL, temp_nn)+b_NL

    W_Lin = params_list[0]
    b_Lin = params_list[1]
    nn_Lin = mtimes(W_Lin,xu) + b_Lin

    return nn_NL + nn_Lin

def getPhi(list_A, Nc, nx, nu):
    Phi = np.zeros([nx*Nc, nx])
    for i in range(Nc):
        temp = np.eye(nx)
        for j in range(i,-1,-1):
            temp = np.matmul(temp, list_A[(nx*j):(nx*j+nx),:])
        Phi[i*nx:(i+1)*nx, :] = temp
    return Phi

def getGamma(list_A, list_B, Nc, nx, nu):
    Gamma = np.zeros([nx*Nc, nu*Nc])
    for i in range(Nc):
        for j in range(0,i+1):
            temp = np.eye(nx)
            for l in range(i-j,-1,-1):
                if l == 0:
                    temp = np.matmul(temp, list_B[(nx*j):(nx*j+nx),:])
                else:
                    temp = np.matmul(temp, list_A[(nx*l):(nx*l+nx),:])
            Gamma[i*nx:nx*(i+1),j*nu:(j+1)*nu] = temp
    return Gamma

def getPsi(Nc, R):
    return np.kron(np.eye(Nc), R)

def getOmega(Nc, Q):
    return np.kron(np.eye(Nc), Q)

def getDEMc(x_min, x_max, u_min, u_max, Nc, nx, nu):
    bi = np.array([list(itertools.chain([-u_min, u_max], [x*-1 for x in x_min],  x_max))])
    bN = np.array([list(itertools.chain([x*-1 for x in x_min],  x_max))])
    c = np.hstack((np.tile(bi, Nc), bN)).T

    In = np.eye(nx)
    Im = np.eye(nu)
    Zn = np.zeros((nu,nx))
    Zm = np.zeros((nx,nu))

    Mi = np.vstack((Zn, Zn, -In, In))
    Mn = np.vstack((-In, In))
    M = (np.zeros((Nc*2*(nx+nu)+2*nx, Nc*nx)))
    M[Nc*2*(nx+nu):,(Nc-1)*nx:] = Mn
    M[2*(nx+nu):Nc*2*(nx+nu),:(Nc-1)*nx] = np.kron(np.eye(Nc-1), Mi)

    Ei = np.vstack((-Im, Im, Zm, Zm))
    E = np.vstack((np.kron(np.eye(Nc), Ei), np.zeros((nx*2, Nc*nu))))

    D = np.zeros((Nc*2*(nx+nu)+2*nx, nx))
    D[:2*(nx+nu),:] = Mi

    return D, E, M, c

def getABlist(x,u,Nc,nx,nu,Get_A,Get_B, list_A, list_B):
    pA = Get_A(np.vstack(np.split(x,Nc)).T,u)
    for i in range(Nc):
        list_A[(nx*i):(nx*i+nx),:] = pA[:,i*nx:(i+1)*nx]
    pB = Get_B(np.vstack(np.split(x,Nc)).T,u)
    for i in range(Nc):
        list_B[(nx*i):(nx*i+nx),:] = pB[:,i*nu:(i+1)*nu]
    
    return list_A, list_B

# Fix use of Xk-1 in getting C, should be Xk
def getABClist(x,u,Nc,nx,nu,ny,Get_A,Get_B,Get_C, list_A, list_B, list_C):
    pA = Get_A(np.vstack(np.split(x,Nc)).T,u)
    for i in range(Nc):
        list_A[(nx*i):(nx*i+nx),:] = pA[:,i*nx:(i+1)*nx]
    pB = Get_B(np.vstack(np.split(x,Nc)).T,u)
    for i in range(Nc):
        list_B[(nx*i):(nx*i+nx),:] = pB[:,i*nu:(i+1)*nu]
    # pC = Get_C(np.vstack(np.split(x,Nc)).T,u)
    # for i in range(Nc):
    #     list_C[(ny*i):(ny*i+ny),:] = pC[:,i*nx:(i+1)*nx]
    
    return list_A, list_B, list_C

def getClist(x,u,Nc,nx,ny,Get_C, list_C):
    pC = Get_C(np.vstack(np.split(x,Nc)).T,u)
    for i in range(Nc):
        list_C[(ny*i):(ny*i+ny),:] = pC[:,i*nx:(i+1)*nx]
    
    return list_C

def getXsUs(y_reference_list_normalize, nx, nu, ny, Nsim, u_min, u_max, x_min, x_max, get_A, get_B, C, f0, h0):
    ne = 1 #number of variables in epsilon
    Q = np.eye(ny) # add this as variable of function
    R = np.eye(nu) # add this as variable of function
    lam = 100
    
    In = np.eye(nx)
    Im = np.eye(nu)
    Zn = np.zeros((nu,nx))
    Zm = np.zeros((nx,nu))

    Mi = np.vstack((Zn, Zn, -In, In))
    Ei = np.vstack((-Im, Im, Zm, Zm))
    h = np.array([list(itertools.chain([-u_min, u_max], [x*-1 for x in x_min],  x_max))]).T

    T = np.zeros((2*(nx+nu), nx+nu+ne))
    T[:,:nx] = Mi
    T[:,nx:nx+nu] = Ei
    T[:,nx+nu:] = -np.ones((2*(nx+nu),ne))

    b = np.zeros((nx+ny, 1))
    b[:nx] = f0

    P = np.zeros((nx+nu+ne, nx+nu+ne))
    P[:nx, :nx] = C.T@Q@C
    P[nx:nx+nu, nx:nx+nu] = R
    P[nx+nu:, nx+nu:] = lam

    q = np.zeros((nx+nu+ne,1))
    
    xs = np.zeros(nx)
    us = np.zeros(nu)
    xue = np.zeros(nx+nu+ne)
    As = np.zeros((nx,nx))
    Bs = np.zeros((nx,nu))
    A = np.zeros((nx+ny, nx+nu+ne))
    A[nx:nx+ny,:nx] = C #change this to getC from xs us when needed

    x_reference_list_normalized = np.zeros((nx, Nsim))
    u_reference_list_normalized = np.zeros((nu, Nsim))
    e_reference_list_normalized = np.zeros((ne, Nsim))

    for j in range(Nsim):

        b[nx:nx+ny] = y_reference_list_normalize[j] - h0 #+ correction_h #add h0 here when needed
        q[:nx,0] = C.T@Q@(h0 - y_reference_list_normalize[j])

        for i in range(10):
            As[:,:] = get_A(xs, us)
            Bs[:,:] = get_B(xs, us)

            A[:nx,:nx] = np.eye(nx) - As
            A[:nx,nx:nx+nu] = -Bs
            #A[nx:,:nx] = C
            #q[:nx,0] = C.T@Q@h0 - C.T@Q@

            #xu[:] = (np.linalg.inv(A)@b)[:,0]
            xue[:] = (qp.solve_qp(P,q,T,h[:,0],A,b[:,0],solver="osqp"))

            xold = xs
            uold = us
            xs = xue[:nx]
            us = xue[nx:nx+nu]
            e = xue[nx+nu:]

            if np.linalg.norm(xs-xold) <= 1e-5 and np.linalg.norm(us-uold) <= 1e-5:
                break

        x_reference_list_normalized[:,j] = xs
        u_reference_list_normalized[:,j] = us
        e_reference_list_normalized[:,j] = e
        
    return x_reference_list_normalized, u_reference_list_normalized, e_reference_list_normalized

def getXsUs_Cs(y_reference_list_normalize, nx, nu, ny, Nsim, u_min, u_max, y_min, y_max, get_A, get_B, get_C, f0, h0):
    ne = 1 #number of variables in epsilon
    Q = 1*np.eye(ny) # add this as variable of function
    R = 1*np.eye(nu) # add this as variable of function
    lam = 1000
    
    In = np.eye(ny)
    Im = np.eye(nu)
    Zn = np.zeros((nu,ny))
    Zm = np.zeros((ny,nu))

    Mi = np.vstack((Zn, Zn, -In, In))
    Ei = np.vstack((-Im, Im, Zm, Zm))
    h = np.array([list(itertools.chain([-u_min, u_max], [x*-1 for x in y_min],  y_max))]).T - Mi@h0

    T = np.zeros((2*(ny+nu), nx+nu+ne))
    #T[:,:ny] = Mi
    T[:,nx:nx+nu] = Ei
    T[:,nx+nu:] = -np.ones((2*(ny+nu),ne))

    b = np.zeros((nx+ny, 1))
    b[:nx] = f0

    P = np.zeros((nx+nu+ne, nx+nu+ne))
    P[nx:nx+nu, nx:nx+nu] = R
    P[nx+nu:, nx+nu:] = lam

    q = np.zeros((nx+nu+ne,1))
    
    xs = np.zeros(nx)
    us = np.zeros(nu)
    xue = np.zeros(nx+nu+ne)
    As = np.zeros((nx,nx))
    Bs = np.zeros((nx,nu))
    Cs = np.zeros((ny,nx))

    A = np.zeros((nx+ny, nx+nu+ne))

    x_reference_list_normalized = np.zeros((nx, Nsim))
    u_reference_list_normalized = np.zeros((nu, Nsim))
    e_reference_list_normalized = np.zeros((ne, Nsim))


    for j in range(Nsim):

        b[nx:nx+ny] = y_reference_list_normalize[j] - h0

        for i in range(20):
            As[:,:] = get_A(xs, us)
            Bs[:,:] = get_B(xs, us)
            Cs[:,:] = get_C(xs, us)

            T[:,:nx] = Mi@Cs

            A[:nx,:nx] = np.eye(nx) - As
            A[:nx,nx:nx+nu] = -Bs
            A[nx:nx+ny,:nx] = Cs

            q[:nx,0] = Cs.T@Q@(h0 - y_reference_list_normalize[j])
            P[:nx, :nx] = Cs.T@Q@Cs

            xue[:] = (qp.solve_qp(P,q,T,h[:,0],A,b[:,0],solver="osqp"))

            xold = np.copy(xs)
            uold = np.copy(us)
            xs = xue[:nx]
            us = xue[nx:nx+nu]
            e = xue[nx+nu:]

            if np.linalg.norm(xs-xold) <= 1e-4 and np.linalg.norm(us-uold) <= 1e-4:
                print("Target Selector iteration: " + str(j))
                print("LPV counter: " + str(i))
                break
        

        x_reference_list_normalized[:,j] = xs
        u_reference_list_normalized[:,j] = us
        e_reference_list_normalized[:,j] = e
        
    return x_reference_list_normalized, u_reference_list_normalized, e_reference_list_normalized

def getF0(list_A, f0, Nc, nx):
    F0 = np.zeros([nx*Nc, nx])
    for i in range(Nc):
        for j in range(i+1):
            temp = np.eye(nx)
            for l in range(1,j+1):
                #print(str(i) + ", " + str(j) + ", " + str(l))
                temp = np.matmul(list_A[(nx*l):(nx*l+nx),:], temp)
            F0[i*nx:(i+1)*nx, :] = F0[i*nx:(i+1)*nx, :] + temp
    return F0@f0

def getZ(list_C, Nc, ny, nx):
    Z = np.zeros((Nc*ny, Nc*nx))
    for i in range(Nc):
        Z[i*ny:(i+1)*ny,i*nx:(i+1)*nx] = list_C[i,:]

    return Z

def getDEMc_out(y_min, y_max, u_min, u_max, Nc, ny, nu):
    bi = np.array([list(itertools.chain([-u_min, u_max], [y*-1 for y in y_min],  y_max))])
    bN = np.array([list(itertools.chain([y*-1 for y in y_min],  y_max))])
    c = np.hstack((np.tile(bi, Nc), bN)).T

    In = np.eye(ny)
    Im = np.eye(nu)
    Zn = np.zeros((nu,ny))
    Zm = np.zeros((ny,nu))

    Mi = np.vstack((Zn, Zn, -In, In))
    Mn = np.vstack((-In, In))
    M = (np.zeros((Nc*2*(ny+nu)+2*ny, Nc*ny)))
    M[Nc*2*(ny+nu):,(Nc-1)*ny:] = Mn
    M[2*(ny+nu):Nc*2*(ny+nu),:(Nc-1)*ny] = np.kron(np.eye(Nc-1), Mi)

    Ei = np.vstack((-Im, Im, Zm, Zm))
    E = np.vstack((np.kron(np.eye(Nc), Ei), np.zeros((ny*2, Nc*nu))))

    D = np.zeros((Nc*2*(ny+nu)+2*ny, ny))
    D[:2*(ny+nu),:] = Mi

    return D, E, M, c

# no difference with getXsUs_Cs currently
def getXsUs_Cs_test(y_reference_list_normalize, nx, nu, ny, Nsim, u_min, u_max, y_min, y_max, get_A, get_B, get_C, f0, h0):
    ne = 1 #number of variables in epsilon
    Q = 1*np.eye(ny) # add this as variable of function
    R = 1*np.eye(nu) # add this as variable of function
    lam = 1000
    
    In = np.eye(ny)
    Im = np.eye(nu)
    Zn = np.zeros((nu,ny))
    Zm = np.zeros((ny,nu))

    Mi = np.vstack((Zn, Zn, -In, In))
    Ei = np.vstack((-Im, Im, Zm, Zm))
    h = np.array([list(itertools.chain([-u_min, u_max], [x*-1 for x in y_min],  y_max))]).T - Mi@h0

    T = np.zeros((2*(ny+nu), nx+nu+ne))
    #T[:,:ny] = Mi
    T[:,nx:nx+nu] = Ei
    T[:,nx+nu:] = -np.ones((2*(ny+nu),ne))

    b = np.zeros((nx+ny, 1))
    b[:nx] = f0

    P = np.zeros((nx+nu+ne, nx+nu+ne))
    #P[:nx, :nx] = C.T@Q@C
    P[nx:nx+nu, nx:nx+nu] = R
    P[nx+nu:, nx+nu:] = lam

    q = np.zeros((nx+nu+ne,1))
    
    xs = np.zeros(nx)
    us = np.zeros(nu)
    xue = np.zeros(nx+nu+ne)
    As = np.zeros((nx,nx))
    Bs = np.zeros((nx,nu))
    Cs = np.zeros((ny,nx))

    A = np.zeros((nx+ny, nx+nu+ne))
    #A[nx:nx+ny,:nx] = C #change this to getC from xs us when needed

    x_reference_list_normalized = np.zeros((nx, Nsim))
    u_reference_list_normalized = np.zeros((nu, Nsim))
    e_reference_list_normalized = np.zeros((ne, Nsim))


    for j in range(Nsim):

        b[nx:nx+ny] = y_reference_list_normalize[j] - h0 #+ correction_h #add h0 here when needed
        #q[:nx,0] = C.T@Q@(h0 - y_reference_list_normalize[j])

        for i in range(10):
            As[:,:] = get_A(xs, us)
            Bs[:,:] = get_B(xs, us)
            Cs[:,:] = get_C(xs, us)

            T[:,:nx] = Mi@Cs

            A[:nx,:nx] = np.eye(nx) - As
            A[:nx,nx:nx+nu] = -Bs
            A[nx:nx+ny,:nx] = Cs
            #b[nx:nx+ny] = Cs - h0

            q[:nx,0] = Cs.T@Q@(h0 - y_reference_list_normalize[j])
            P[:nx, :nx] = Cs.T@Q@Cs

            #xu[:] = (np.linalg.inv(A)@b)[:,0]
            #xue[:] = (qp.solve_qp(P,q,T,h[:,0],A,b[:,0],solver="osqp"))
            xue[:] = (qp.solve_qp(P,q,T,h[:,0],A,b[:,0],solver="osqp"))

            xold = xs
            uold = us
            xs = xue[:nx]
            us = xue[nx:nx+nu]
            e = xue[nx+nu:]

            if np.linalg.norm(xs-xold) <= 1e-8 and np.linalg.norm(us-uold) <= 1e-8:
                break

        x_reference_list_normalized[:,j] = xs
        u_reference_list_normalized[:,j] = us
        e_reference_list_normalized[:,j] = e
        
    return x_reference_list_normalized, u_reference_list_normalized, e_reference_list_normalized


# def getABgrad(x,u,Nc,nx,nu,Get_A,Get_B, list_A, list_B, Lambda):
#     mult_A = np.stack((np.ones((nx,nx)), np.ones((nx,nx))*4, np.ones((nx,nx))))
#     mult_B = np.stack((np.ones((nx,nu)), np.ones((nx,nu))*4, np.ones((nx,nu))))
    
#     for j in range(Nc):
#         A = np.zeros((nx,nx))
#         B = np.zeros([nx,nu]) 

#         lambda0 = 0

#         for i in range(n_stages):
#             an = fA[n_int_comp*n_stages*j+(i)*n_int_comp:n_int_comp*n_stages*j+(i+1)*n_int_comp,0,:,0,:].detach().numpy()
#             A = A + dlam*1/6*np.sum(np.multiply(mult_A, an), axis=0)

#             bn = fB[n_int_comp*n_stages*j+(i)*n_int_comp:n_int_comp*n_stages*j+(i+1)*n_int_comp,0,:,0,:].detach().numpy()
#             B = B + dlam*1/6*np.sum(np.multiply(mult_B, bn), axis=0)

#             lambda0 = lambda0 + dlam

#         list_A[nx*(j):nx*(j+1),:] = A.copy()
#         list_B[nx*(j):nx*(j+1),:] = B.copy()

#     return list_A, list_B

# def getCgrad(x,u,Nc,nx,ny,Get_C, list_C, Lambda):
#     mult_C = np.vstack((np.ones((ny,nx)), np.ones((ny,nx))*4, np.ones((ny,nx))))
    
#     for j in range(Nc):
#         C = np.zeros((ny,nx))

#         lambda0 = 0

#         for i in range(n_stages):
#             cn = fC[n_int_comp*n_stages*j+(i)*n_int_comp:n_int_comp*n_stages*j+(i+1)*n_int_comp,0,0,:].detach().numpy()
#             C = C + dlam*1/6*np.sum(np.multiply(mult_C, cn), axis=0)[np.newaxis]

#             lambda0 = lambda0 + dlam

#         list_C[ny*(j):ny*(j+1),:] = C.copy()

#     return list_C

# add new code here