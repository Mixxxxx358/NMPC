import time
import qpsolvers as qp
import deepSI
import numpy as np
from casadi import *
from torch import nn
from matplotlib import pyplot as plt

import Systems
from my_rk4 import *
from lpv_int import *
from par_lpv_int import *
from lpv_rk4 import *
from mpcUtil import *
import torch

import torch
from functorch import vmap, jacrev

# -------------------------  NMPC function  ------------------------

def output_NMPC_linear(system, encoder, u_min, u_max, y_min, y_max, x0, Q, R, dt, dlam, stages, y_reference_list, Nc=5, Nsim=30, max_iterations=1):
    # declared sym variables
    nx = encoder.nx
    n_states = nx
    x = MX.sym("x",nx,1)
    nu = encoder.nu if I_enc.nu is not None else 1
    n_controls = nu
    u = MX.sym("u",nu,1)
    ny = I_enc.ny if I_enc.ny is not None else 1

    # convert torch nn to casadi function
    rhs = CasADiFn(I_enc, x, u)
    f = Function('f', [x, u], [rhs])
    y_rhs = CasADiHn(I_enc, x)
    h = Function('h', [x], [y_rhs])

    correction = f(np.zeros((nx,1)), 0)
    #correction = f([0,0,0,0], 0)
    rhs_c = rhs - correction
    correction_h = h(np.zeros((nx,1)))
    #correction_h = h([0,0,0,0])
    y_rhs_c = y_rhs - correction_h

    # normalize reference list
    norm = encoder.norm
    y_reference_list_normalized = ((y_reference_list.T - norm.y0)/norm.ystd).T
    x0_norm = (x0 - norm.y0)/norm.ystd
    u0 = 0
    u0_norm = (u0 - norm.u0)/norm.ustd

    # determine getA and getB functions
    Jfx = Function("Jfx", [x, u], [jacobian(rhs_c,x)])
    Jfu = Function("Jfu", [x, u], [jacobian(rhs_c,u)])
    Jhx = Function("Jhx", [x, u], [jacobian(y_rhs_c,x)])
    # Jhu = Function("Jhu", [x, u], [jacobian(y_rhs_c,u)])
    
    # [A_sym, B_sym, C_sym] = lpv_int_C(x,nx,u,nu,ny,Jfx,Jfu,Jhx,dlam,stages)
    [A_sym, B_sym, C_sym] = lambda_simpson(x,u,nx,nu,ny,Jfx,Jfu,Jhx,stages)
    get_A = Function("get_A",[x,u],[A_sym])
    get_B = Function("get_B",[x,u],[B_sym])
    get_C = Function("get_C",[x,u],[C_sym])
    Get_A = get_A.map(Nc, "thread", 10)
    Get_B = get_B.map(Nc, "thread", 10)
    Get_C = get_C.map(Nc, "thread", 10)
    
    # declare bounds of system
    # x_max_norm = (x_max - norm.y0)/norm.ystd
    # x_min_norm = (x_min - norm.y0)/norm.ystd
    y_max_norm = (y_max - norm.y0)/norm.ystd
    y_min_norm = (y_min - norm.y0)/norm.ystd
    u_min_norm = (u_min - norm.u0)/norm.ustd
    u_max_norm = (u_max - norm.u0)/norm.ustd

    h0 = np.array(correction_h.elements())
    y_reference_list_normalized = np.clip(y_reference_list_normalized, y_min_norm, y_max_norm)

    #C = np.array([[0, 1]])
    x_reference_list_normalized, u_reference_list_normalized, e_reference_list_normalized = getXsUs_Cs(y_reference_list_normalized,\
         nx, nu, 1, Nsim+Nc, u_min_norm, u_max_norm, y_min_norm, y_max_norm, get_A, get_B, get_C, correction, h0) # fix the Nsim for Xs

    # plt.plot(np.arange(Nsim+Nc), e_reference_list_normalized[0,:])
    # plt.show()

    # logging list
    u_log = np.zeros(Nsim*n_controls)
    x_log = np.zeros((Nsim+1)*n_states)
    x_log[:nx] = x0
    y_log = np.zeros((Nsim+1))
    y_est_log = np.zeros((Nsim+1)*ny)
    e_log = np.zeros([n_states,Nsim])
    t = np.zeros(Nsim)
    t0 = 0
    comp_t_log = np.zeros(Nsim)
    start = time.time()
    lpv_counter = np.zeros(Nsim,int)
    components_time = np.zeros((4, Nsim*max_iterations))

    # set initial values for x
    x = np.tile(x0_norm, Nc) #Xk
    u = np.tile(u0_norm, Nc) #Uk

    list_A = np.zeros([Nc*nx, nx])
    list_B = np.zeros([Nc*nx, nu])
    list_C = np.zeros([Nc*ny, nx])
    Psi = getPsi(Nc, R)
    Omega = getOmega(Nc, Q)
    #D, E, M, c = getDEMc(x_min_norm, x_max_norm, u_min_norm, u_max_norm, Nc, nx, nu)
    D, E, M, c = getDEMc_out(y_min_norm, y_max_norm, u_min_norm, u_max_norm, Nc, ny, nu)

    ne = 1
    Ge = np.zeros((Nc+ne, Nc+ne))

    # Observer variables
    nb = encoder.na
    uhist = torch.zeros((1,nb))
    na = encoder.nb
    yhist = torch.zeros((1,na+1))

    f0 = np.array(correction.elements())[np.newaxis].T
    H0 = np.tile(h0,Nc)[np.newaxis].T
    y_norm = np.zeros((1,1))

    system.reset_state()
    fig7 = plt.figure(figsize=[8.9, 8])


    Lambda = np.array([])
    lambda0 = 0
    dlam = 1/stages
    for i in np.arange(stages):
        Lambda = np.hstack((Lambda, lambda0, lambda0 + dlam/2, lambda0 + dlam)) # Simpson
        # Lambda = np.hstack((Lambda, lambda0, lambda0 + dlam)) # Trapezium
        lambda0 = lambda0 + dlam

    JacF = vmap(jacrev(encoder.fn, argnums=(0,1)))
    JacH = vmap(jacrev(encoder.hn))
    n_int_comp = 3
    n_stages = stages
    batch_size = Nc*stages*n_int_comp

    mult_A = np.stack((np.ones((nx,nx)), np.ones((nx,nx))*4, np.ones((nx,nx))))
    mult_B = np.stack((np.ones((nx,nu)), np.ones((nx,nu))*4, np.ones((nx,nu))))
    mult_C = np.vstack((np.ones((ny,nx)), np.ones((ny,nx))*4, np.ones((ny,nx))))


    for mpciter in range(Nsim):
        start_time_iter = time.time()
        Xs = np.reshape(x_reference_list_normalized[:,mpciter+1:mpciter+Nc+1].T, (nx*Nc,1))
        Us = u_reference_list_normalized[:,mpciter:mpciter+Nc].T

        while True:
            
            #list_A, list_B = getABlist(x,u,Nc,nx,nu,Get_A,Get_B, list_A, list_B)
            #list_A, list_B, list_C = getABClist(x,u,Nc,nx,nu,ny,Get_A,Get_B,Get_C, list_A, list_B, list_C)

            # list_A, list_B, list_C = getABClist(np.hstack((x0_norm, x[:-nx])),u,Nc,nx,nu,ny,Get_A,Get_B,Get_C, list_A, list_B, list_C)
            # #temporary fix to get C with pk+1 instead of pk above
            # list_C = getClist(x,np.hstack((u[1:],u[-1])),Nc,nx,ny,Get_C, list_C)


            # component_start = time.time()
            Xlam = np.kron(np.hstack((x0_norm, x[:-nx])), Lambda)
            Ulam = np.kron(u, Lambda)
            Xlam_C = np.kron(x, Lambda)

            x_tens = torch.reshape(torch.Tensor(Xlam[np.newaxis].T),(batch_size,1,2))
            u_tens = torch.reshape(torch.Tensor(Ulam[np.newaxis].T),(batch_size,1,1))
            x_tens_C = torch.reshape(torch.Tensor(Xlam_C[np.newaxis].T),(batch_size,1,2))

            component_start = time.time()
            fA, fB = JacF(x_tens,u_tens)
            fC = JacH(x_tens_C)
            components_time[0, max_iterations*mpciter + lpv_counter[mpciter]] = components_time[0, max_iterations*mpciter + lpv_counter[mpciter]] + time.time() - component_start

            for j in range(Nc):
                A = np.zeros((nx,nx))
                B = np.zeros([nx,nu])
                C = np.zeros((ny,nx))

                lambda0 = 0

                for i in range(n_stages):
                    an = fA[n_int_comp*n_stages*j+(i)*n_int_comp:n_int_comp*n_stages*j+(i+1)*n_int_comp,0,:,0,:].detach().numpy()
                    A = A + dlam*1/6*np.sum(np.multiply(mult_A, an), axis=0)

                    bn = fB[n_int_comp*n_stages*j+(i)*n_int_comp:n_int_comp*n_stages*j+(i+1)*n_int_comp,0,:,0,:].detach().numpy()
                    B = B + dlam*1/6*np.sum(np.multiply(mult_B, bn), axis=0)

                    cn = fC[n_int_comp*n_stages*j+(i)*n_int_comp:n_int_comp*n_stages*j+(i+1)*n_int_comp,0,0,:].detach().numpy()
                    C = C + dlam*1/6*np.sum(np.multiply(mult_C, cn), axis=0)[np.newaxis]

                    lambda0 = lambda0 + dlam

                list_A[nx*(j):nx*(j+1),:] = A.copy()
                list_B[nx*(j):nx*(j+1),:] = B.copy()
                list_C[ny*(j):ny*(j+1),:] = C.copy()

            # components_time[0, max_iterations*mpciter + lpv_counter[mpciter]] = components_time[0, max_iterations*mpciter + lpv_counter[mpciter]] + time.time() - component_start

            
            component_start = time.time()
            F0 = getF0(list_A, f0, Nc, nx)
            Phi = getPhi(list_A, Nc, nx, nu)
            Gamma = getGamma(list_A, list_B, Nc, nx, nu)
            Z = getZ(list_C,Nc,ny,nx)
            G = 2*(Psi+(Gamma.T@Omega@Gamma))
            #F = 2*(Gamma.T@Omega@(Phi@(x[:2][np.newaxis].T) - Xs) - Psi@Us + Gamma.T@Omega@F0)
            F = 2*(Gamma.T@Omega@(Phi@(x0_norm[np.newaxis].T) - Xs) - Psi@Us + Gamma.T@Omega@F0)
            #L = (M@Gamma) + E
            L = (M@Z@Gamma) + E
            #W = -D - (M@Phi)
            W = -(D@y_norm) - (M@Z@(Phi@x0_norm +F0[:,0])) - (M@H0)[:,0]
            # print((D@y_norm).shape)
            # print((M@Z@(Phi@x0_norm +F0[:,0])).shape)
            # print(((M@H0)[:,0]).shape)

            #Le = np.hstack((L, -np.ones((Nc*2*(nx+nu)+2*nx,1))))
            Le = np.hstack((L, -np.ones((Nc*2*(ny+nu)+2*ny,1))))
            Ge[:Nc, :Nc] = G
            Ge[Nc:,Nc:] = 1000
            Fe = np.vstack((F, np.zeros((ne,ne))))

            u_old = np.copy(u)
            x_old = np.copy(x)

            #ue = qp.solve_qp(Ge,Fe,Le,(W@(x0_norm)) + c[:,0],solver="osqp")
            # print((W + c[:,0]).shape)
            ue = qp.solve_qp(Ge,Fe,Le,W + c[:,0],solver="osqp")
            # ue = qp.solve_qp(Ge,Fe,solver="osqp")
            u[:] = ue[:Nc]
            e = ue[Nc:]

            x[:] = ((Phi@x0_norm) + Gamma@u)[:] + F0[:,0]

            # lpv iteration output and input plot
            # if mpciter == 0:
            #     str_label = "iter " + str(lpv_counter[mpciter] + 1)

            #     y_plot = norm.ystd*(Z@x + H0[:,0]) + norm.y0
            #     plt.subplot(2,1,1)
            #     plt.plot(y_plot, label=str_label)

            #     u_plot = norm.ustd*u + norm.u0
            #     plt.subplot(2,1,2)
            #     plt.plot(u_plot, label=str_label)
            
            lpv_counter[mpciter] += 1
            # print((np.linalg.norm(u-u_old)))
            # print(x)
            # print(x_old)
            # print(((np.subtract(x, x_old))))
            if (lpv_counter[mpciter] >= max_iterations) or (np.linalg.norm(u-u_old) < 1e-1):
                components_time[1, max_iterations*mpciter + lpv_counter[mpciter]-1] = components_time[1, max_iterations*mpciter + lpv_counter[mpciter]-1] + time.time() - component_start
                break
            components_time[1, max_iterations*mpciter + lpv_counter[mpciter]-1] = components_time[1, max_iterations*mpciter + lpv_counter[mpciter]-1] + time.time() - component_start

        print("MPC iteration: ", mpciter+1)
        print("LPV counter: ", lpv_counter[mpciter])
        
        component_start = time.time()
        t[mpciter] = t0
        t0 = t0 + dt
        
        # denormalize x and u
        #x_denormalized = norm.ystd*x0_norm + norm.y0
        u_denormalized = norm.ustd*u[0] + norm.u0
        components_time[2, max_iterations*mpciter + lpv_counter[mpciter]-1] = components_time[2, max_iterations*mpciter + lpv_counter[mpciter]-1] + time.time() - component_start

        # make system step and normalize
        component_start = time.time()
        system.x = system.f(system.x, u_denormalized)
        y_measured = system.h(system.x, u_denormalized)
        components_time[3, max_iterations*mpciter + lpv_counter[mpciter]-1] = components_time[3, max_iterations*mpciter + lpv_counter[mpciter]-1] + time.time() - component_start
        component_start = time.time()
        y_norm[:] = (y_measured - norm.y0)/norm.ystd

        # apply observer
        for j in range(nb-1):
            uhist[0,j] = uhist[0,j+1]
        uhist[0,nb-1] = torch.Tensor([u[0]])

        for j in range(na):
            yhist[0,j] = yhist[0,j+1]
        yhist[0,na] = torch.Tensor([y_norm])

        zest = encoder.encoder(uhist,yhist)
        yest = encoder.hn(zest)
        try:
            yest = (yest.detach()).numpy()
        except:
            yest = yest
        yest_denorm = norm.ystd*yest + norm.y0

        # log measurements and variables
        x_log[(mpciter+1)*nx:(mpciter+2)*nx] = zest.detach()
        y_log[mpciter+1] = y_measured
        y_est_log[mpciter+1] = yest_denorm
        u_log[mpciter] = u_denormalized#u[0]# 
        e_log[0,mpciter] = e
        
        # shift mpc variables left one step
        x[:] = np.hstack((x[nx:(Nc)*nx],x[-nx:]))
        #x[:nx] = zest.detach()
        x0_norm[:] = zest.detach()
        u = np.hstack((u[nu:Nc*nu],u[-nu:]))

        # finished mpc time measurement
        end_time_iter = time.time()
        comp_t_log[mpciter] = end_time_iter - start_time_iter
        components_time[2, max_iterations*mpciter + lpv_counter[mpciter]-1] = components_time[2, max_iterations*mpciter + lpv_counter[mpciter]-1] + time.time() - component_start

    end = time.time()
    runtime = end - start

    return x_log, u_log, e_log, comp_t_log, t, runtime, lpv_counter, components_time, y_log, y_est_log, x_reference_list_normalized, u_reference_list_normalized

# -------------------------  Main function  -------------------------

if __name__ == "__main__":
    # MPC parameters
    dt = 0.1
    Nc = 10
    Nsim = 450
    #dlam = 0.05
    stages = 20
    dlam = 1.0/stages
    max_iterations = 4
    nx = 2

    # Weight matrices for the cost function
    # Q = 10*np.eye(nx)
    Q = np.matrix('1000,0;0,10')
    R = 1
    
    # Box constraints
    u_min = -4
    u_max = 4
    y_min = [-1.2]
    y_max = [1.2]

    # Initial and final values
    x0 = np.zeros((nx,))

    # determine state references
    # x_reference_list = 1*np.load("references/randomLevelTime5_10Range-1_1Nsim500.npy")
    x_reference_list = 1*np.load("references/randomLevelTime25_30Range-1_1Nsim500.npy")
    # x_reference_list = np.load("references/multiSineInOutNsim500_2.npy")/2
    #x_reference_list = np.vstack((np.zeros(300),0.3*np.ones(300)))
    #x_reference_list = np.vstack((np.zeros(40),np.hstack((1*np.ones(20),-1*np.ones(20)))))
    
    sys_unblanced = Systems.OutputUnbalancedDisc(dt=dt, sigma_n=[0.0]) #0.045, 0.014
    # sys_unblanced = Systems.LTI()

    #I_enc = deepSI.load_system("systems/ObserverUnbalancedDisk_dt01_nab_4_e200")
    #I_enc = deepSI.load_system("systems/ObserverUnbalancedDisk_dt01_e300")
    #I_enc = deepSI.load_system("systems/ObserverUnbalancedDisk_dt01_nab_4_SNR_20_e100")
    I_enc = deepSI.load_system("systems/ObserverUnbalancedDisk_dt01_nab_4_SNR_30_e250")
    # I_enc = deepSI.load_system("systems/LtiTanh_nab_4_Ru_4_SNR_25_e20")

    x_reference_list = np.clip(x_reference_list, y_min, y_max)

    x_log, u_log, e_log, comp_t_log, t, runtime, lpv_counter, components_time, y_log, y_est_log, Xs, Us \
         = output_NMPC_linear(sys_unblanced, I_enc, u_min=u_min, u_max=u_max, y_min=y_min, y_max=y_max, x0=x0, \
            Q=Q, R=R, dt=dt, dlam=dlam, stages=stages, y_reference_list=x_reference_list[1,:], Nc=Nc, Nsim=Nsim, max_iterations=max_iterations)

    print("Runtime:" + str(runtime))

    np.save("Experiments/components_time_long.npy", components_time)
    np.save("Experiments/comp_t_log_long.npy", comp_t_log)
    np.save("Experiments/lpv_counter_long.npy", lpv_counter)

# ------------------------------  Plots  -------------------------------

    nx = I_enc.nx
    nu = I_enc.nu if I_enc.nu is not None else 1

    # continuation of LPV iteration output and input plot
    Us_denorm = Us*I_enc.norm.ustd + I_enc.norm.u0
    plt.subplot(2,1,1)
    plt.grid()
    # plt.xlabel("time [s]")
    plt.ylabel("angle [rad]")
    # plt.plot(np.arange(Nc+1), np.hstack((np.zeros(1),x_reference_list[1,:Nc])), '--', label='reference')
    # plt.plot(np.arange(Nc+1), np.ones(Nc+1)*y_max[0], 'r-.', label='max')
    # plt.plot(np.arange(Nc+1), np.ones(Nc+1)*y_min[0], 'r-.', label='min')
    plt.legend()

    plt.subplot(2,1,2)
    plt.grid()
    plt.xlabel("steps")
    plt.ylabel("input [V]")
    # plt.plot(np.arange(Nc), Us_denorm[0,:Nc], '--', label='steady state')
    # plt.plot(np.arange(Nc), np.ones(Nc)*u_max, 'r-.', label='max')
    # plt.plot(np.arange(Nc), np.ones(Nc)*u_min, 'r-.', label='min')
    plt.legend()

    plt.show()
    
    

    fig1 = plt.figure(figsize=[8.9, 8])

    # plt.subplot(4,1,1)
    # plt.plot(np.arange(Nsim+1)*dt, x_log[0::nx], label='state 1')
    # plt.plot(np.arange(Nsim+1)*dt, np.hstack((np.zeros(1),Xs[0,:Nsim])), '--', label='steady state')
    # plt.xlabel("time [s]")
    # plt.grid()
    # plt.legend()

    # plt.subplot(4,1,2)
    # plt.plot(np.arange(Nsim+1)*dt, x_log[1::nx], label='state 2')
    # plt.plot(np.arange(Nsim+1)*dt, np.hstack((np.zeros(1),Xs[1,:Nsim])), '--', label='steady state')
    # plt.xlabel("time [s]")
    # plt.grid()
    # plt.legend()

    log_q = np.load("Experiments/log_q.npy")
    log_w = np.load("Experiments/log_w.npy")

    Us_denorm = Us*I_enc.norm.ustd + I_enc.norm.u0
    plt.subplot(2,1,1)
    plt.plot(np.arange(Nsim)*dt, u_log[:], label='input')
    plt.plot(np.arange(Nsim)*dt, Us_denorm[0,:Nsim], '--', label='steady state')
    plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*u_max, 'r-.', label='max')
    plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*u_min, 'r-.', label='min')
    # plt.plot(np.arange(Nsim)*dt, u_log[:] - log_w[0,:], label='LPV-NMPC')
    plt.ylabel("input [V]", fontsize=12)
    # plt.xlabel("time [s]", fontsize=12)
    plt.grid()
    plt.legend(loc='upper right', fontsize=12);

    plt.subplot(2,1,2)
    plt.plot(np.arange(Nsim+1)*dt, y_log, label='output')
    plt.plot(np.arange(Nsim+1)*dt, np.hstack((np.zeros(1),x_reference_list[1,:Nsim])), '--', label='reference')
    plt.plot(np.arange(Nsim+1)*dt, y_est_log, 'y:', label='obsv est')
    plt.plot(np.arange(Nsim+1)*dt, np.ones(Nsim+1)*y_max[0], 'r-.', label='max')
    plt.plot(np.arange(Nsim+1)*dt, np.ones(Nsim+1)*y_min[0], 'r-.', label='min')
    plt.plot(np.arange(Nsim+1)*dt, y_log[:] - np.hstack((0,log_q[0,:])), label='LPV-NMPC')
    #plt.plot(np.arange(Nsim+1)*dt, y_log - np.hstack((np.zeros(1),x_reference_list[1,:Nsim])), label='error')
    plt.ylabel("angle [rad]", fontsize=12)
    plt.xlabel("time [s]", fontsize=12)
    plt.grid()    
    plt.legend(loc='upper right', fontsize=12);

    plt.show()

    # fig6 = plt.figure(figsize=[5.0, 5.0])
    # #plt.plot(np.arange(Nsim+1)*dt, y_log - np.hstack((np.zeros(1),x_reference_list[1,:Nsim])), label='error')
    # plt.plot(np.arange(Nsim+1)*dt, y_log, label='output')
    # plt.plot(np.arange(Nsim+1)*dt, y_est_log, label='obsv est')
    # plt.plot(np.arange(Nsim+1)*dt, np.ones(Nsim+1)*y_max[0], 'r-.', label='max')
    # plt.plot(np.arange(Nsim+1)*dt, np.ones(Nsim+1)*y_min[0], 'r-.', label='min')
    # plt.plot(np.arange(Nsim+1)*dt, np.hstack((np.zeros(1),x_reference_list[1,:Nsim])), '--', label='reference')
    # plt.ylabel("angle [rad]")
    # plt.xlabel("time [s]")
    # plt.grid()    
    # plt.legend();
    # plt.show()

    fig4 = plt.figure(figsize=[8.9, 4.0])
    #plt.subplot(2,1,1)
    plt.step(np.arange(Nsim), lpv_counter, label='lpv counter')
    plt.plot(np.arange(Nsim), np.ones(Nsim)*max_iterations, '--', label='max iter')
    plt.ylabel("lpv counter")
    plt.xlabel("mpciter")
    plt.grid()
    plt.legend();
    # plt.show()

    fig4 = plt.figure(figsize=[8.9, 4.0])
    #plt.subplot(2,1,2)
    plt.step(np.arange(Nsim), comp_t_log, label='computation time')
    plt.plot(np.arange(Nsim), np.ones(Nsim)*dt, '--', label='dt')
    plt.ylabel("computation time")
    plt.xlabel("mpciter")
    plt.grid()
    plt.legend()
    # plt.show()

    fig2 = plt.figure(figsize=[8.9, 4.0])#['getAB', 'solve', 'overhead', 'sim']
    data1 = np.trim_zeros(components_time[0,:])
    data2 = np.trim_zeros(components_time[1,:])
    data3 = np.trim_zeros(components_time[2,:])
    data4 = np.trim_zeros(components_time[3,:])
    data = [data1, data2, data3, data4]
    plt.boxplot(data)
    plt.xticks([1, 2, 3, 4],  ['getAB', 'solve', 'overhead', 'sim'])
    plt.grid(axis='y')
    plt.ylabel("time [s]")
    plt.xlabel("components")
    # plt.show()

    fig3 = plt.figure(figsize=[8.9, 4.0])#['getAB', 'solve', 'overhead', 'sim']
    plt.bar(['getAB', 'solve', 'overhead', 'sim'], np.sum(components_time, axis=1))
    plt.ylabel("time [s]")
    plt.xlabel("components")
    plt.show()