import numpy as np
import itertools
import qpsolvers as qp

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

def getXsUs_Cs(y_reference_list_normalize, nx, nu, ny, Nsim, u_min, u_max, x_min, x_max, get_A, get_B, get_C, C, f0, h0):
    ne = 1 #number of variables in epsilon
    Q = np.eye(ny) # add this as variable of function
    R = np.eye(nu) # add this as variable of function
    lam = 1000
    
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
        q[:nx,0] = C.T@Q@(h0 - y_reference_list_normalize[j])

        for i in range(20):
            As[:,:] = get_A(xs, us)
            Bs[:,:] = get_B(xs, us)
            Cs[:,:] = get_C(xs, us)

            A[:nx,:nx] = np.eye(nx) - As
            A[:nx,nx:nx+nu] = -Bs
            A[nx:nx+ny,:nx] = Cs
            q[:nx,0] = Cs.T@Q@(h0 - y_reference_list_normalize[j])
            P[:nx, :nx] = Cs.T@Q@Cs

            #xu[:] = (np.linalg.inv(A)@b)[:,0]
            xue[:] = (qp.solve_qp(P,q,T,h[:,0],A,b[:,0],solver="osqp"))

            xold = xs
            uold = us
            xs = xue[:nx]
            us = xue[nx:nx+nu]
            e = xue[nx+nu:]

            if np.linalg.norm(xs-xold) <= 1e-6 and np.linalg.norm(us-uold) <= 1e-6:
                break

        x_reference_list_normalized[:,j] = xs
        u_reference_list_normalized[:,j] = us
        e_reference_list_normalized[:,j] = e
        
    return x_reference_list_normalized, u_reference_list_normalized, e_reference_list_normalized