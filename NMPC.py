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

# -------------------------  Util functions  -------------------------

class I_encoder(deepSI.fit_systems.SS_encoder):
    def __init__(self, nx = 2, na=2, nb=2, feedthrough=False) -> None:
        super().__init__(nx=nx, na=na, nb=nb, feedthrough=feedthrough)

    def init_nets(self, nu, ny): # a bit weird
        ny = ny if ny is not None else 1
        nu = nu if nu is not None else 1
        self.encoder = self.e_net(self.nb*nu+self.na*ny, self.nx, n_nodes_per_layer=self.e_n_nodes_per_layer, n_hidden_layers=self.e_n_hidden_layers, activation=self.e_activation)
        self.fn =      self.f_net(self.nx+nu,            self.nx, n_nodes_per_layer=self.f_n_nodes_per_layer, n_hidden_layers=self.f_n_hidden_layers, activation=self.f_activation)
        hn_in = self.nx + nu if self.feedthrough else self.nx
        self.hn =      nn.Identity(hn_in)

# -------------------------  NMPC functions  ------------------------

def NMPC(system, encoder, x_min, x_max, u_min, u_max, x0, u_ref, Q, R, dt, dlam, stages, x_reference_list, Nc=5, Nsim=30, max_iterations=1):
    # declared sym variables
    nx = encoder.nx
    x = MX.sym("x",nx,1)
    nu = encoder.nu if I_enc.nu is not None else 1
    u = MX.sym("u",nu,1)

    # convert torch nn to casadi function
    rhs = CasADiExp(I_enc, x, u)
    f = Function('f', [x, u], [rhs])
    correction = f([0,0], 0)
    rhs_c = rhs - correction
    #f_c = Function('f_c', [x, u], [rhs_c])

    opti = Opti()

    # declare variables and parameters of states and inputs
    n_states = np.shape(x)[0]
    states = opti.variable(n_states,Nc+1)    
    x_initial = opti.parameter(n_states,1)

    n_controls = np.shape(u)[0]
    controls = opti.variable(n_controls,Nc)

    reference = opti.parameter(n_states,1)

    epsilon = opti.variable(n_states,1)
    opti.subject_to(epsilon[0,0] == epsilon[1,0])

    # normalize reference list
    norm = encoder.norm
    reference_list_normalized = ((x_reference_list.T - norm.y0)/norm.ystd).T
    x0_norm = (x0 - norm.y0)/norm.ystd

    # determine getA and getB functions
    Jfx = Function("Jfx", [x, u], [jacobian(rhs_c,x)])
    Jfu = Function("Jfu", [x, u], [jacobian(rhs_c,u)])
    
    # LPV generation
    # Jfx = SX.sym("JFx",n_states,n_states)
    # Jfu = SX.sym("JFu",n_controls,n_states)
    # for i in np.arange(n_states):
    #     Jfx[:,i] = gradient(rhs[i],x)
    #     Jfu[:,i] = gradient(rhs[i],u)
    # Jfx = Jfx.T
    # Jfu = Jfu.T
    # Jfx = Function("Jfx", [x, u], [Jfx])
    # Jfu = Function("Jfu", [x, u], [Jfu])
    
    [A_sym, B_sym] = lpv_int(x,n_states,u,n_controls,Jfx,Jfu,dlam,stages)
    get_A = Function("get_A",[x,u],[A_sym])
    get_B = Function("get_B",[x,u],[B_sym])
    Get_A = get_A.map(Nc, "thread", 32)
    Get_B = get_B.map(Nc, "thread", 32)
    list_A = opti.parameter(Nc*n_states,n_states)
    list_B = opti.parameter(Nc*n_states,n_controls)

    # declare bounds of system
    x_max_norm = (x_max - norm.y0)/norm.ystd
    x_min_norm = (x_min - norm.y0)/norm.ystd
    #opti.subject_to(opti.bounded(x_min_norm,states,x_max_norm))
    u_min_norm = (u_min - norm.u0)/norm.ustd
    u_max_norm = (u_max - norm.u0)/norm.ustd
    opti.subject_to(opti.bounded(u_min_norm,controls,u_max_norm))
    opti.subject_to(states[:,0] == x_initial)

    # opts = {'print_time' : 0, 'ipopt': {'print_level': 0}}
    # opti.solver("ipopt",opts)
    opts = {'print_time' : 0, 'ipopt': {'print_level': 0}}
    opti.solver("ipopt", opts)

    objective = 0 # add some soft bounds on the states and inputs
    for i in np.arange(Nc):
        opti.subject_to(states[:,i+1] == list_A[(n_states*i):(n_states*i+n_states),:]@states[:,i] \
            + list_B[(n_states*i):(n_states*i+n_states),:]@controls[:,i] + correction)
        objective = (objective + 
                        mtimes(mtimes((states[:,i]-reference).T,Q),(states[:,i]-reference)) +
                        mtimes(mtimes((controls[:,i]-u_ref).T,R),(controls[:,i]-u_ref)))
        opti.subject_to(opti.bounded(x_min_norm - epsilon, states[:,i], x_max_norm + epsilon))
    objective = objective + epsilon.T @ epsilon *10000
    opti.minimize(objective)

    # logging list
    t = np.zeros(Nsim)
    t0 = 0
    u_log = np.zeros([n_controls,Nsim])
    x_log = np.zeros([n_states,Nsim+1])
    e_log = np.zeros([n_states,Nsim])
    comp_t_log = np.zeros(Nsim)
    components_total_time = np.zeros(4) # getAB, solve, overhead, sim
    x_log[:,0] = x0
    start = time.time()
    lpv_counter = np.zeros(Nsim,int)

    # set initial values for x
    x = np.zeros([n_states,Nc+1])
    u = np.zeros([n_controls,Nc]) # change this to u0 instead of 0 maybe
    opti.set_initial(states, x)
    opti.set_initial(controls, u)
    opti.set_initial(epsilon, [0,0])
    opti.set_value(x_initial,x0_norm)

    for mpciter in np.arange(Nsim):
        start_time_iter = time.time()
        component_start = time.time()
        # solve for u and x
        opti.set_value(reference,[reference_list_normalized[0,mpciter],reference_list_normalized[1,mpciter]])
        components_total_time[2] = components_total_time[2] + time.time() - component_start

        # MPC loop
        while True:
            component_start = time.time()
            # determine A,B
            # opti.set_value(list_A, reshape(Get_A(x,u), (Nc*nx, nx)))
            # opti.set_value(list_B, reshape(Get_B(x,u), (Nc*nx, nu)))
            for i in np.arange(Nc):
                opti.set_value(list_A[(n_states*i):(n_states*i+n_states),:],get_A(x[:,i],u[:,i]))
                opti.set_value(list_B[(n_states*i):(n_states*i+n_states),:],get_B(x[:,i],u[:,i]))
            components_total_time[0] = components_total_time[0] + time.time() - component_start
            
            component_start = time.time()
            # solve for u and x
            sol = opti.solve()
            u_old = u
            u = np.reshape(sol.value(controls),[n_controls,Nc])
            x = np.reshape(sol.value(states),[n_states,Nc+1]) # this relies on internal simulation in solution, this is x=Ax+Bu
            # simulate next step using rk4 over non correction casadi function
            #for i in np.arange(Nc):
            #    x[:,i+1] = np.ravel(my_rk4(x[:,i],u[:,i],f,dt),order='F')
            x = np.reshape(sol.value(states),[n_states,Nc+1]) # change this to nn maybe
            components_total_time[1] = components_total_time[1] + time.time() - component_start
            
            component_start = time.time()
            # set new x and u values into optimizer
            opti.set_initial(states, x)
            opti.set_initial(controls, u)

            lpv_counter[mpciter] += 1

            # Stop MPC loop if max iteration reached or input converged
            if (lpv_counter[mpciter] >= max_iterations) or (np.linalg.norm(u-u_old) < 1e-5):
                components_total_time[2] = components_total_time[2] + time.time() - component_start
                break
            components_total_time[2] = components_total_time[2] + time.time() - component_start

        print("MPC iteration: ", mpciter+1)
        print("LPV counter: ", lpv_counter[mpciter])

        component_start = time.time()
        t[mpciter] = t0
        t0 = t0 + dt
        try:
            x = x.full()
        except:
            x = x
        try:
            u = u.full()
        except:
            u = u
        
        # denormalize x and u
        x_denormalized = norm.ystd*x0_norm + norm.y0
        u_denormalized = norm.ustd*u[0,0] + norm.u0
        components_total_time[2] = components_total_time[2] + time.time() - component_start

        # make system step and normalize
        component_start = time.time()
        x_denormalized = system.f(x_denormalized, u_denormalized)
        x_measured = system.h(x_denormalized, u_denormalized)
        components_total_time[3] = components_total_time[3] + time.time() - component_start

        component_start = time.time()
        x0_norm = (x_measured - norm.y0)/norm.ystd

        x_log[:,mpciter+1] = x_measured
        u_log[:,mpciter] = u_denormalized
        e_log[:,mpciter] = np.reshape(sol.value(epsilon),[n_states,])
        
        x = horzcat(x[:,1:(Nc+1)],x[:,-1])
        x[:,0] = x0_norm
        u = horzcat(u[:,1:Nc],u[:,-1])
        opti.set_value(x_initial, x0_norm)
        opti.set_initial(states, x)
        opti.set_initial(controls, u)
        opti.set_initial(epsilon, e_log[:,mpciter])

        # finished mpc time measurement
        end_time_iter = time.time()
        comp_t_log[mpciter] = end_time_iter - start_time_iter
        components_total_time[2] = components_total_time[2] + time.time() - component_start

    end = time.time()
    runtime = end - start

    return x_log, u_log, e_log, comp_t_log, t, runtime, lpv_counter, reference_list_normalized, components_total_time

def NMPC_nonLPV(system, encoder, x_min, x_max, u_min, u_max, x0, u_ref, Q, R, dt, dlam, stages, x_reference_list, Nc=5, Nsim=30, max_iterations=1):
    # declared sym variables
    nx = encoder.nx
    x = MX.sym("x",nx,1)
    nu = I_enc.nu if I_enc.nu is not None else 1
    u = MX.sym("u",nu,1)

    # convert torch nn to casadi function
    rhs = CasADiExp(I_enc, x, u)
    f = Function('f', [x, u], [rhs])

    opti = Opti()

    # declare variables and parameters of states and inputs
    n_states = np.shape(x)[0]
    states = opti.variable(n_states,Nc+1)    
    x_initial = opti.parameter(n_states,1)

    n_controls = np.shape(u)[0]
    controls = opti.variable(n_controls,Nc)

    reference = opti.parameter(n_states,1)

    epsilon = opti.variable(n_states,1)
    opti.subject_to(epsilon[0,0] == epsilon[1,0])

    # normalize reference list
    norm = encoder.norm
    reference_list_normalized = ((x_reference_list.T - norm.y0)/norm.ystd).T
    x0_norm = (x0 - norm.y0)/norm.ystd

    # declare bounds of system
    x_max_norm = (x_max - norm.y0)/norm.ystd
    x_min_norm = (x_min - norm.y0)/norm.ystd
    #opti.subject_to(opti.bounded(x_min_norm,states,x_max_norm))
    u_min_norm = (u_min - norm.u0)/norm.ustd
    u_max_norm = (u_max - norm.u0)/norm.ustd
    opti.subject_to(opti.bounded(u_min_norm,controls,u_max_norm))
    opti.subject_to(states[:,0] == x_initial)

    opts = {'print_time' : 0, 'ipopt': {'print_level': 0}}
    opti.solver("ipopt",opts)

    objective = 0 # add some soft bounds on the states and inputs
    for i in np.arange(Nc):
        opti.subject_to(states[:,i+1] == f(states[:,i], controls[:,i]))
        objective = (objective + 
                        mtimes(mtimes((states[:,i]-reference).T,Q),(states[:,i]-reference)) +
                        mtimes(mtimes((controls[:,i]-u_ref).T,R),(controls[:,i]-u_ref)))
        opti.subject_to(opti.bounded(x_min_norm - epsilon, states[:,i], x_max_norm + epsilon))
    objective = objective + epsilon.T @ epsilon *10000
    opti.minimize(objective)

    # logging list
    t = np.zeros(Nsim)
    t0 = 0
    u_log = np.zeros([n_controls,Nsim])
    x_log = np.zeros([n_states,Nsim+1])
    e_log = np.zeros([n_states,Nsim])
    comp_t_log = np.zeros(Nsim)
    x_log[:,0] = x0
    start = time.time()
    lpv_counter = np.zeros(Nsim,int)

    # set initial values for x
    x = np.zeros([n_states,Nc+1])
    u = np.zeros([n_controls,Nc]) # change this to u0 instead of 0 maybe
    opti.set_initial(states, x)
    opti.set_initial(controls, u)
    opti.set_initial(epsilon, [0,0])
    opti.set_value(x_initial,x0_norm)

    for mpciter in np.arange(Nsim):
        start_time_iter = time.time()

        # solve for u and x
        opti.set_value(reference,[reference_list_normalized[0,mpciter],reference_list_normalized[1,mpciter]])

        # MPC loop
        while True:
            # solve for u and x            
            sol = opti.solve()

            u_old = u
            u = np.reshape(sol.value(controls),[n_controls,Nc])
            x = np.reshape(sol.value(states),[n_states,Nc+1]) # this relies on internal simulation in solution, this is x=Ax+Bu
            # simulate next step using rk4 over non correction casadi function
            #for i in np.arange(Nc):
            #    x[:,i+1] = np.ravel(my_rk4(x[:,i],u[:,i],f,dt),order='F')
            x = np.reshape(sol.value(states),[n_states,Nc+1]) # change this to nn maybe
            
            # set new x and u values into optimizer
            opti.set_initial(states, x)
            opti.set_initial(controls, u)

            lpv_counter[mpciter] += 1  

            # Stop MPC loop if max iteration reached or input converged
            if (lpv_counter[mpciter] >= max_iterations) or (np.linalg.norm(u-u_old) < 1e-5):
                break

        print("MPC iteration: ", mpciter+1)
        print("LPV counter: ", lpv_counter[mpciter])

        t[mpciter] = t0
        t0 = t0 + dt
        try:
            x = x.full()
        except:
            x = x
        try:
            u = u.full()
        except:
            u = u
        
        # denormalize x and u
        x_denormalized = norm.ystd*x0_norm + norm.y0
        u_denormalized = norm.ustd*u[0,0] + norm.u0

        # make system step and normalize
        x_denormalized = system.f(x_denormalized, u_denormalized)
        x_measured = system.h(x_denormalized, u_denormalized)
        x0_norm = (x_measured - norm.y0)/norm.ystd

        x_log[:,mpciter+1] = x_measured
        u_log[:,mpciter] = u_denormalized
        e_log[:,mpciter] = np.reshape(sol.value(epsilon),[n_states,])
        
        x = horzcat(x[:,1:(Nc+1)],x[:,-1])
        x[:,0] = x0_norm
        u = horzcat(u[:,1:Nc],u[:,-1])
        opti.set_value(x_initial, x0_norm)
        opti.set_initial(states, x)
        opti.set_initial(controls, u)
        opti.set_initial(epsilon, e_log[:,mpciter])

        # finished mpc time measurement
        end_time_iter = time.time()
        comp_t_log[mpciter] = end_time_iter - start_time_iter

    end = time.time()
    runtime = end - start

    return x_log, u_log, e_log, comp_t_log, t, runtime, lpv_counter, reference_list_normalized

def NMPC_linear(system, encoder, x_min, x_max, u_min, u_max, x0, u_ref, Q, R, dt, dlam, stages, x_reference_list, Nc=5, Nsim=30, max_iterations=1):
    # declared sym variables
    nx = encoder.nx
    n_states = nx
    x = MX.sym("x",nx,1)
    nu = encoder.nu if I_enc.nu is not None else 1
    n_controls = nu
    u = MX.sym("u",nu,1)

    # convert torch nn to casadi function
    rhs = CasADiExp(I_enc, x, u)
    f = Function('f', [x, u], [rhs])
    correction = f([0,0], 0)
    rhs_c = rhs - correction

    # normalize reference list
    norm = encoder.norm
    reference_list_normalized = ((x_reference_list.T - norm.y0)/norm.ystd).T
    x0_norm = (x0 - norm.y0)/norm.ystd

    # determine getA and getB functions
    Jfx = Function("Jfx", [x, u], [jacobian(rhs_c,x)])
    Jfu = Function("Jfu", [x, u], [jacobian(rhs_c,u)])
    # Jhx = Function("Jhx", [x, u], [jacobian(y_rhs_c,x)])
    # Jhu = Function("Jhu", [x, u], [jacobian(y_rhs_c,u)])
    
    [A_sym, B_sym] = lpv_int(x,n_states,u,n_controls,Jfx,Jfu,dlam,stages)
    get_A = Function("get_A",[x,u],[A_sym])
    get_B = Function("get_B",[x,u],[B_sym])
    Get_A = get_A.map(Nc, "thread", 32)
    Get_B = get_B.map(Nc, "thread", 32)

    # declare bounds of system
    x_max_norm = (x_max - norm.y0)/norm.ystd
    x_min_norm = (x_min - norm.y0)/norm.ystd
    #opti.subject_to(opti.bounded(x_min_norm,states,x_max_norm))
    u_min_norm = (u_min - norm.u0)/norm.ustd
    u_max_norm = (u_max - norm.u0)/norm.ustd

    # logging list
    u_log = np.zeros(Nsim*n_controls)
    x_log = np.zeros((Nsim+1)*n_states)
    x_log[:nx] = x0
    e_log = np.zeros([n_states,Nsim])
    t = np.zeros(Nsim)
    t0 = 0
    comp_t_log = np.zeros(Nsim)
    start = time.time()
    lpv_counter = np.zeros(Nsim,int)
    components_total_time = np.zeros(4) # getAB, solve, overhead, sim

    # set initial values for x
    x = np.tile(x0_norm, Nc)
    u = np.zeros(Nc*nu)

    list_A = np.zeros([Nc*nx, nx])
    list_B = np.zeros([Nc*nx, nu])
    Psi = getPsi(Nc, R)
    Omega = getPsi(Nc, Q)
    D, E, M, c = getDEMc(x_min_norm, x_max_norm, u_min_norm, u_max_norm, Nc, nx, nu)

    for mpciter in range(Nsim):
        start_time_iter = time.time()
        
        while True:
            component_start = time.time()
            for i in np.arange(Nc):
                list_A[(n_states*i):(n_states*i+n_states),:] = get_A(x[i*nx:(i+1)*nx],u[i*nu:(i+1)*nu])
                list_B[(n_states*i):(n_states*i+n_states),:] = get_B(x[i*nx:(i+1)*nx],u[i*nu:(i+1)*nu])
            components_total_time[0] = components_total_time[0] + time.time() - component_start
            
            component_start = time.time()
            Phi = getPhi(list_A, Nc, nx, nu)
            Gamma = getGamma(list_A, list_B, Nc, nx, nu)
            G = 2*(Psi+(Gamma.T@Omega@Gamma))
            F = 2*(Gamma.T@Omega@Phi)
            L = (M@Gamma) + E
            W = -D - (M@Phi)

            u_old = u
            #x_ss = x[:2] - [0, 1]
            #u = -np.linalg.inv(G)@F@x[:2]

            #component_start = time.time()
            u = qp.solve_qp(G,F@x[:2],L,(W@x[:2]) + c[:,0], solver="quadprog")
            #components_total_time[1] = components_total_time[1] + time.time() - component_start
            
            #component_start = time.time()
            x[nx:Nc*nx] = ((Phi@x[:2]) + Gamma@u)[:(Nc-1)*nx]# + np.tile(np.array(correction_f)[:,0], (Nc-1))
            
            lpv_counter[mpciter] += 1
            if (lpv_counter[mpciter] >= 5) or (np.linalg.norm(u-u_old) < 1e-7):
                components_total_time[1] = components_total_time[1] + time.time() - component_start
                break
            components_total_time[1] = components_total_time[1] + time.time() - component_start

        print("MPC iteration: ", mpciter+1)
        print("LPV counter: ", lpv_counter[mpciter])
        
        component_start = time.time()
        t[mpciter] = t0
        t0 = t0 + dt
        
        # denormalize x and u
        x_denormalized = norm.ystd*x0_norm + norm.y0
        u_denormalized = norm.ustd*u[0] + norm.u0
        components_total_time[2] = components_total_time[2] + time.time() - component_start

        # make system step and normalize
        component_start = time.time()
        x_denormalized = system.f(x_denormalized, u_denormalized)
        x_measured = system.h(x_denormalized, u_denormalized)
        components_total_time[3] = components_total_time[3] + time.time() - component_start
        component_start = time.time()
        x0_norm = (x_measured - norm.y0)/norm.ystd

        x_log[(mpciter+1)*nx:(mpciter+2)*nx] = x_measured
        u_log[mpciter] = u_denormalized
        
        x = np.hstack((x[nx:(Nc+1)*nx],x[-2:]))
        x[:nx] = x0_norm
        u = np.hstack((u[nx:Nc*nx],u[-2:]))

        # finished mpc time measurement
        end_time_iter = time.time()
        comp_t_log[mpciter] = end_time_iter - start_time_iter
        components_total_time[2] = components_total_time[2] + time.time() - component_start

    end = time.time()
    runtime = end - start

    return x_log, u_log, e_log, comp_t_log, t, runtime, lpv_counter, reference_list_normalized, components_total_time

def par_NMPC_linear(system, encoder, x_min, x_max, u_min, u_max, x0, u_ref, Q, R, dt, dlam, stages, x_reference_list, Nc=5, Nsim=30, max_iterations=1):
    # declared sym variables
    nx = encoder.nx
    n_states = nx
    x = MX.sym("x",nx,1)
    nu = encoder.nu if I_enc.nu is not None else 1
    n_controls = nu
    u = MX.sym("u",nu,1)

    # convert torch nn to casadi function
    rhs = CasADiExp(I_enc, x, u)
    f = Function('f', [x, u], [rhs])
    correction = f([0,0], 0)
    rhs_c = rhs - correction

    # normalize reference list
    norm = encoder.norm
    reference_list_normalized = ((x_reference_list.T - norm.y0)/norm.ystd).T
    x0_norm = (x0 - norm.y0)/norm.ystd
    u0 = 0
    u0_norm = (u0 - norm.u0)/norm.ustd

    # determine getA and getB functions
    Jfx = Function("Jfx", [x, u], [jacobian(rhs_c,x)])
    Jfu = Function("Jfu", [x, u], [jacobian(rhs_c,u)])

    [A_sym, B_sym] = lpv_int(x,n_states,u,n_controls,Jfx,Jfu,dlam,stages)
    get_A = Function("get_A",[x,u],[A_sym])
    get_B = Function("get_B",[x,u],[B_sym])
    Get_A = get_A.map(Nc, "thread", 32)
    Get_B = get_B.map(Nc, "thread", 32)

    y_reference_list_normalized = reference_list_normalized[1,:]
    C = np.array([[0, 1]])
    x_reference_list_normalized, u_reference_list_normalized, e_reference_list_normalized = getXsUs(y_reference_list_normalized,\
         nx, nu, 1, Nsim+Nc, u_min, u_max, x_min, x_max, get_A, get_B, C, correction, np.zeros(1)) # fix the Nsim for Xs
    
    # declare bounds of system
    x_max_norm = (x_max - norm.y0)/norm.ystd
    x_min_norm = (x_min - norm.y0)/norm.ystd
    u_min_norm = (u_min - norm.u0)/norm.ustd
    u_max_norm = (u_max - norm.u0)/norm.ustd

    # logging list
    u_log = np.zeros(Nsim*n_controls)
    x_log = np.zeros((Nsim+1)*n_states)
    x_log[:nx] = x0
    e_log = np.zeros([n_states,Nsim])
    t = np.zeros(Nsim)
    t0 = 0
    comp_t_log = np.zeros(Nsim)
    start = time.time()
    lpv_counter = np.zeros(Nsim,int)
    components_time = np.zeros((4, Nsim*max_iterations))

    # set initial values for x
    x = np.tile(x0_norm, Nc)
    u = np.tile(u0_norm, Nc)

    list_A = np.zeros([Nc*nx, nx])
    list_B = np.zeros([Nc*nx, nu])
    Psi = getPsi(Nc, R)
    Omega = getPsi(Nc, Q)
    D, E, M, c = getDEMc(x_min_norm, x_max_norm, u_min_norm, u_max_norm, Nc, nx, nu)

    ne = 1
    Ge = np.zeros((Nc+ne, Nc+ne))

    system.reset_state()

    for mpciter in range(Nsim):
        start_time_iter = time.time()
        Xs = np.reshape(x_reference_list_normalized[:,mpciter+1:mpciter+Nc+1].T, (2*Nc,1))
        Us = u_reference_list_normalized[:,:Nc].T

        while True:
            component_start = time.time()
            list_A, list_B = getABlist(x,u,Nc,nx,nu,Get_A,Get_B, list_A, list_B)
            components_time[0, mpciter + lpv_counter[mpciter]] = components_time[0, mpciter + lpv_counter[mpciter]] + time.time() - component_start
            
            component_start = time.time()
            Phi = getPhi(list_A, Nc, nx, nu)
            Gamma = getGamma(list_A, list_B, Nc, nx, nu)
            G = 2*(Psi+(Gamma.T@Omega@Gamma))
            F = 2*(Gamma.T@Omega@(Phi@(x[:2][np.newaxis].T) - Xs) + Psi.T@Us)
            L = (M@Gamma) + E
            W = -D - (M@Phi)

            Le = np.hstack((L, -np.ones((Nc*2*(nx+nu)+2*nx,1))))
            Ge[:Nc, :Nc] = G
            Ge[Nc:,Nc:] = 10000
            Fe = np.vstack((F, np.zeros((ne,ne))))

            u_old = u

            ue = qp.solve_qp(Ge,Fe,Le,(W@(x[:2])) + c[:,0],solver="osqp")
            u = ue[:Nc]
            e = ue[Nc:]

            x[nx:Nc*nx] = ((Phi@x[:2]) + Gamma@u)[:(Nc-1)*nx]
            
            lpv_counter[mpciter] += 1
            if (lpv_counter[mpciter] >= max_iterations) or (np.linalg.norm(u-u_old) < 1e-5):
                components_time[1, mpciter + lpv_counter[mpciter]-1] = components_time[1, mpciter + lpv_counter[mpciter]-1] + time.time() - component_start
                break
            components_time[1, mpciter + lpv_counter[mpciter]-1] = components_time[1, mpciter + lpv_counter[mpciter]-1] + time.time() - component_start

        print("MPC iteration: ", mpciter+1)
        print("LPV counter: ", lpv_counter[mpciter])
        
        component_start = time.time()
        t[mpciter] = t0
        t0 = t0 + dt
        
        # denormalize x and u
        x_denormalized = norm.ystd*x0_norm + norm.y0
        u_denormalized = norm.ustd*u[0] + norm.u0
        components_time[2, mpciter + lpv_counter[mpciter]-1] = components_time[2, mpciter + lpv_counter[mpciter]-1] + time.time() - component_start

        # make system step and normalize
        component_start = time.time()
        system.x = system.f(system.x, u_denormalized)
        x_measured = system.h(system.x, u_denormalized)
        components_time[3, mpciter + lpv_counter[mpciter]-1] = components_time[3, mpciter + lpv_counter[mpciter]-1] + time.time() - component_start
        component_start = time.time()
        x0_norm = (x_measured - norm.y0)/norm.ystd

        x_log[(mpciter+1)*nx:(mpciter+2)*nx] = x_measured
        u_log[mpciter] = u_denormalized
        e_log[0,mpciter] = e
        
        x = np.hstack((x[nx:(Nc+1)*nx],x[-2:]))
        x[:nx] = x0_norm
        u = np.hstack((u[nx:Nc*nx],u[-2:]))

        # finished mpc time measurement
        end_time_iter = time.time()
        comp_t_log[mpciter] = end_time_iter - start_time_iter
        components_time[2, mpciter + lpv_counter[mpciter]-1] = components_time[2, mpciter + lpv_counter[mpciter]-1] + time.time() - component_start

    end = time.time()
    runtime = end - start

    return x_log, u_log, e_log, comp_t_log, t, runtime, lpv_counter, reference_list_normalized, components_time

def output_NMPC_linear(system, encoder, x_min, x_max, u_min, u_max, x0, u_ref, Q, R, dt, dlam, stages, x_reference_list, Nc=5, Nsim=30, max_iterations=1):
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

    correction = f([0,0], 0)
    rhs_c = rhs - correction
    correction_h = h([0,0])
    y_rhs_c = y_rhs - correction_h

    # normalize reference list
    norm = encoder.norm
    reference_list_normalized = ((x_reference_list.T - norm.y0)/norm.ystd).T
    x0_norm = (x0 - norm.y0)/norm.ystd
    u0 = 0
    u0_norm = (u0 - norm.u0)/norm.ustd

    # determine getA and getB functions
    Jfx = Function("Jfx", [x, u], [jacobian(rhs_c,x)])
    Jfu = Function("Jfu", [x, u], [jacobian(rhs_c,u)])
    Jhx = Function("Jhx", [x, u], [jacobian(y_rhs_c,x)])
    # Jhu = Function("Jhu", [x, u], [jacobian(y_rhs_c,u)])
    
    [A_sym, B_sym, C_sym] = lpv_int_C(x,nx,u,nu,ny,Jfx,Jfu,Jhx,dlam,stages)
    get_A = Function("get_A",[x,u],[A_sym])
    get_B = Function("get_B",[x,u],[B_sym])
    get_C = Function("get_C",[x,u],[C_sym])
    Get_A = get_A.map(Nc, "thread", 32)
    Get_B = get_B.map(Nc, "thread", 32)

    y_reference_list_normalized = reference_list_normalized[1,:]
    #C = np.array([[0, 1]])
    x_reference_list_normalized, u_reference_list_normalized, e_reference_list_normalized = getXsUs_Cs(y_reference_list_normalized,\
         nx, nu, 1, Nsim+Nc, u_min, u_max, x_min, x_max, get_A, get_B, get_C, correction, np.array(correction_h)[:,0]) # fix the Nsim for Xs
    
    # declare bounds of system
    x_max_norm = (x_max - norm.y0)/norm.ystd
    x_min_norm = (x_min - norm.y0)/norm.ystd
    u_min_norm = (u_min - norm.u0)/norm.ustd
    u_max_norm = (u_max - norm.u0)/norm.ustd

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
    x = np.tile(x0_norm, Nc)
    u = np.tile(u0_norm, Nc)

    list_A = np.zeros([Nc*nx, nx])
    list_B = np.zeros([Nc*nx, nu])
    Psi = getPsi(Nc, R)
    Omega = getPsi(Nc, Q)
    D, E, M, c = getDEMc(x_min_norm, x_max_norm, u_min_norm, u_max_norm, Nc, nx, nu)

    ne = 1
    Ge = np.zeros((Nc+ne, Nc+ne))

    # Observer variables
    nb = 4
    uhist = torch.zeros((1,nb))#torch.tensor([[0,0]],dtype=torch.float32)
    na = 4
    yhist = torch.zeros((1,na+1))#torch.tensor([[0,0,0]],dtype=torch.float32)

    f0 = np.array(correction.elements())[np.newaxis].T

    system.reset_state()

    for mpciter in range(Nsim):
        start_time_iter = time.time()
        Xs = np.reshape(x_reference_list_normalized[:,mpciter+1:mpciter+Nc+1].T, (2*Nc,1))
        Us = u_reference_list_normalized[:,mpciter:mpciter+Nc].T

        while True:
            component_start = time.time()
            list_A, list_B = getABlist(x,u,Nc,nx,nu,Get_A,Get_B, list_A, list_B)
            components_time[0, mpciter + lpv_counter[mpciter]] = components_time[0, mpciter + lpv_counter[mpciter]] + time.time() - component_start
            
            component_start = time.time()
            F0 = getF0(list_A, f0, Nc, nx)
            Phi = getPhi(list_A, Nc, nx, nu)
            Gamma = getGamma(list_A, list_B, Nc, nx, nu)
            G = 2*(Psi+(Gamma.T@Omega@Gamma))
            F = 2*(Gamma.T@Omega@(Phi@(x[:2][np.newaxis].T) - Xs) - Psi@Us + Gamma.T@Omega@F0)
            L = (M@Gamma) + E
            W = -D - (M@Phi)

            Le = np.hstack((L, -np.ones((Nc*2*(nx+nu)+2*nx,1))))
            Ge[:Nc, :Nc] = G
            Ge[Nc:,Nc:] = 10000
            Fe = np.vstack((F, np.zeros((ne,ne))))

            u_old = u

            ue = qp.solve_qp(Ge,Fe,Le,(W@(x[:2])) + c[:,0],solver="osqp")
            u[:] = ue[:Nc]
            e = ue[Nc:]

            x[nx:Nc*nx] = ((Phi@x[:2]) + Gamma@u)[:(Nc-1)*nx] + F0[:(Nc-1)*nx,0]# + np.tile(correction.elements(), Nc-1)
            
            lpv_counter[mpciter] += 1
            if (lpv_counter[mpciter] >= max_iterations) or (np.linalg.norm(u-u_old) < 1e-5):
                components_time[1, mpciter + lpv_counter[mpciter]-1] = components_time[1, mpciter + lpv_counter[mpciter]-1] + time.time() - component_start
                break
            components_time[1, mpciter + lpv_counter[mpciter]-1] = components_time[1, mpciter + lpv_counter[mpciter]-1] + time.time() - component_start

        print("MPC iteration: ", mpciter+1)
        print("LPV counter: ", lpv_counter[mpciter])
        
        component_start = time.time()
        t[mpciter] = t0
        t0 = t0 + dt
        
        # denormalize x and u
        #x_denormalized = norm.ystd*x0_norm + norm.y0
        u_denormalized = norm.ustd*u[0] + norm.u0
        components_time[2, mpciter + lpv_counter[mpciter]-1] = components_time[2, mpciter + lpv_counter[mpciter]-1] + time.time() - component_start

        # make system step and normalize
        component_start = time.time()
        system.x = system.f(system.x, u_denormalized)
        y_measured = system.h(system.x, u_denormalized)
        components_time[3, mpciter + lpv_counter[mpciter]-1] = components_time[3, mpciter + lpv_counter[mpciter]-1] + time.time() - component_start
        component_start = time.time()
        y_norm = (y_measured - norm.y0)/norm.ystd

        # apply observer
        for j in range(nb-1):
            uhist[0,j] = uhist[0,j+1]
        uhist[0,nb-1] = torch.Tensor([u[0]])

        for j in range(na):
            yhist[0,j] = yhist[0,j+1]
        yhist[0,na] = torch.Tensor([y_norm])

        zest = encoder.encoder(uhist,yhist)
        yest = encoder.hn(zest)
        yest_denorm = norm.ystd*yest + norm.y0

        # log measurements and variables
        x_log[(mpciter+1)*nx:(mpciter+2)*nx] = zest.detach()
        y_log[mpciter+1] = y_measured
        y_est_log[mpciter+1] = yest_denorm
        u_log[mpciter] = u[0]#u_denormalized
        e_log[0,mpciter] = e
        
        # shift mpc variables left one step
        x = np.hstack((x[nx:(Nc+1)*nx],x[-2:]))
        x[:nx] = zest.detach()
        u = np.hstack((u[nx:Nc*nx],u[-2:]))

        # finished mpc time measurement
        end_time_iter = time.time()
        comp_t_log[mpciter] = end_time_iter - start_time_iter
        components_time[2, mpciter + lpv_counter[mpciter]-1] = components_time[2, mpciter + lpv_counter[mpciter]-1] + time.time() - component_start

    end = time.time()
    runtime = end - start

    return x_log, u_log, e_log, comp_t_log, t, runtime, lpv_counter, reference_list_normalized, components_time, y_log, y_est_log, x_reference_list_normalized, u_reference_list_normalized


# -------------------------  Main function  -------------------------

if __name__ == "__main__":
    # MPC parameters
    dt = 0.1
    Nc = 10
    Nsim = 400
    dlam = 0.05
    stages = 20
    max_iterations = 5

    # Weight matrices for the cost function
    Q = np.matrix('100,0;0,100')
    R = 1
    
    # Box constraints
    x_min_out = [-100, -100]
    x_max_out = [100, 100]
    x_min = [-8, -2]
    x_max = [8, 2]
    u_min = -6
    u_max = 6

    # Initial and final values
    x0 = [0,0]
    u_ref = 0

    # determine state references
    #x_reference_list = np.load("references/randomLevelTime5_10Range-1_1Nsim500.npy")
    x_reference_list = np.load("references/randomLevelTime25_30Range-1_1Nsim500.npy")
    #x_reference_list = np.vstack((np.zeros(300),np.zeros(300)))

    # Weight matrices for the cost function
    Q = np.matrix('1,0;0,1000')
    R = 1

    # Initialize system and load corresponding encoder
    # sys_unblanced = Systems.NoisyUnbalancedDisc(dt=dt, sigma_n=[0,0])#0.47, 0.044
    # I_enc = deepSI.load_system("systems/UnbalancedDisk_dt01_e100_SNR_100")
    
    sys_unblanced = Systems.OutputUnbalancedDisc(dt=dt, sigma_n=[0.0])
    I_enc = deepSI.load_system("systems/ObserverUnbalancedDisk_dt01_nab_4_e200")

    # x_log, u_log, e_log, comp_t_log, t, runtime, lpv_counter, reference_list_normalized, components_time = par_NMPC_linear(sys_unblanced, I_enc, \
    #     x_min=x_min, x_max= x_max, u_min=u_min, u_max=u_max, x0=x0, u_ref=u_ref, Q=Q, R=R, dt=dt, dlam=dlam, stages=stages, \
    #     x_reference_list=x_reference_list, Nc=Nc, Nsim=Nsim, max_iterations=max_iterations)

    x_log, u_log, e_log, comp_t_log, t, runtime, lpv_counter, reference_list_normalized, components_time, y_log, y_est_log, Xs, Us \
         = output_NMPC_linear(sys_unblanced, I_enc, x_min=x_min_out, x_max= x_max_out, u_min=u_min, u_max=u_max, x0=x0, u_ref=u_ref, \
            Q=Q, R=R, dt=dt, dlam=dlam, stages=stages, x_reference_list=x_reference_list, Nc=Nc, Nsim=Nsim, max_iterations=max_iterations)

    print("Runtime:" + str(runtime))

# ------------------------------  Plots  -------------------------------

    nx = I_enc.nx
    nu = I_enc.nu if I_enc.nu is not None else 1

    fig1 = plt.figure(figsize=[14.0, 3.0])

    plt.subplot(2,3,1)
    #plt.plot(np.arange(Nsim+1)*dt, x_log[0,:], label='angluar velocity')
    plt.plot(np.arange(Nsim+1)*dt, x_log[0::nx], label='state 1')
    plt.plot(np.arange(Nsim+1)*dt, np.hstack((np.zeros(1),Xs[0,:Nsim])), '--', label='steady state')
    # plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*x_max[0], 'r-.', label='max')
    # plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*x_min[0], 'r-.', label='min')
    #plt.ylabel("angular velocity [rad/s]") # not sure about the unit
    plt.xlabel("time [s]")
    plt.grid()
    plt.legend()

    plt.subplot(2,3,2)
    #plt.plot(np.arange(Nsim+1)*dt, x_log[1,:], label='angle')
    plt.plot(np.arange(Nsim+1)*dt, x_log[1::nx], label='state 2')
    plt.plot(np.arange(Nsim+1)*dt, np.hstack((np.zeros(1),Xs[1,:Nsim])), '--', label='steady state') # figure out what the correct hstack should be here
    # plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*x_max[1], 'r-.', label='max')
    # plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*x_min[1], 'r-.', label='min')
    #plt.ylabel("angle [rad]") # not sure about the unit
    plt.xlabel("time [s]")
    plt.grid()
    plt.legend()

    plt.subplot(2,3,3)
    #plt.plot(np.arange(Nsim)*dt, u_log[0,:], label='input')
    plt.plot(np.arange(Nsim)*dt, u_log[:], label='input')
    plt.plot(np.arange(Nsim)*dt, Us[0,:Nsim], '--', label='steady state')
    # plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*u_ref, '--', label='reference')
    plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*u_max, 'r-.', label='max')
    plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*u_min, 'r-.', label='min')
    plt.ylabel("input [V]")
    plt.xlabel("time [s]")
    plt.grid()
    plt.legend();

    plt.subplot(2,3,4)
    plt.step(np.arange(Nsim), lpv_counter, label='lpv counter')
    plt.plot(np.arange(Nsim), np.ones(Nsim)*max_iterations, '--', label='max iter')
    plt.ylabel("lpv counter")
    plt.xlabel("mpciter")
    plt.grid()
    plt.legend();

    plt.subplot(2,3,5)
    plt.step(np.arange(Nsim), comp_t_log, label='computation time')
    plt.plot(np.arange(Nsim), np.ones(Nsim)*dt, '--', label='dt')
    plt.ylabel("computation time")
    plt.xlabel("mpciter")
    plt.grid()
    plt.legend()

    plt.subplot(2,3,6)
    # plt.step(np.arange(Nsim), e_log[0,:], label='epsilon')
    # plt.plot(np.arange(Nsim), np.zeros(Nsim), '--', label='0 value')
    # plt.ylabel("epsilon")
    # plt.xlabel("mpciter")
    plt.plot(np.arange(Nsim+1)*dt, y_log, label='output')
    plt.plot(np.arange(Nsim+1)*dt, y_est_log, label='obsv est')
    plt.plot(np.arange(Nsim+1)*dt, np.hstack((np.zeros(1),x_reference_list[1,:Nsim])), '--', label='reference') # figure out what the correct hstack should be here
    # plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*x_max[1], 'r-.', label='max')
    # plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*x_min[1], 'r-.', label='min')
    plt.ylabel("angle [rad]")
    plt.xlabel("time [s]")
    plt.grid()    
    plt.legend();

    plt.show()

    fig2 = plt.figure(figsize=[10.0, 10.0])#['getAB', 'solve', 'overhead', 'sim']
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

    fig3 = plt.figure(figsize=[10.0, 10.0])#['getAB', 'solve', 'overhead', 'sim']
    plt.bar(['getAB', 'solve', 'overhead', 'sim'], np.sum(components_time, axis=1))
    plt.ylabel("time [s]")
    plt.xlabel("components")
    # plt.show()