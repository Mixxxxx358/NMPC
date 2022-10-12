import deepSI
import numpy as np
from casadi import *
from torch import nn
from Systems import DuffingOscillator
import Systems
from matplotlib import pyplot as plt
import random

from my_rk4 import *
from lpv_int import *
from lpv_rk4 import *

import time

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

    #f = Function('f', [x, u], [nn_NL + nn_Lin])
    return nn_NL + nn_Lin

def NMPC(system, encoder, x_min, x_max, u_min, u_max, x0, u_ref, Q, R, dt, dlam, stages, x_reference_list, Nc=5, Nsim=30, max_iterations=1):
    # declared sym variables
    nx = encoder.nx
    x = MX.sym("x",nx,1)
    nu = I_enc.nu if I_enc.nu is not None else 1
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

    # determine getA and getB functions
    Jfx = Function("Jfx", [x, u], [jacobian(rhs_c,x)])
    Jfu = Function("Jfu", [x, u], [jacobian(rhs_c,u)])
    [A_sym, B_sym] = lpv_int(x,n_states,u,n_controls,Jfx,Jfu,dlam,stages)
    get_A = Function("get_A",[x,u],[A_sym])
    get_B = Function("get_B",[x,u],[B_sym])
    list_A = opti.parameter(Nc*n_states,n_states)
    list_B = opti.parameter(Nc*n_states,n_controls)

    # declare bounds of system
    opti.subject_to(opti.bounded(x_min,states,x_max))
    opti.subject_to(opti.bounded(u_min,controls,u_max))
    opti.subject_to(states[:,0] == x_initial)

    opts = {'print_time' : 0, 'ipopt': {'print_level': 0}}
    opti.solver("ipopt",opts)

    objective = 0 # add some soft bounds on the states and inputs
    for i in np.arange(Nc):
        opti.subject_to(states[:,i+1] == list_A[(n_states*i):(n_states*i+n_states),:]@states[:,i] \
            + list_B[(n_states*i):(n_states*i+n_states),:]@controls[:,i] + correction)
        objective = (objective + 
                        mtimes(mtimes((states[:,i]-reference).T,Q),(states[:,i]-reference)) +
                        mtimes(mtimes((controls[:,i]-u_ref).T,R),(controls[:,i]-u_ref)))
    opti.minimize(objective)

    # normalize reference list
    norm = encoder.norm
    reference_list_normalized = (x_reference_list - norm.y0[1])/norm.ystd[1]
    x0_norm = (x0 - norm.y0)/norm.ystd

    # logging list
    t = np.zeros(Nsim)
    t0 = 0
    u_log = np.zeros([n_controls,Nsim])
    x_log = np.zeros([n_states,Nsim+1])
    comp_t_log = np.zeros(Nsim)
    x_log[:,0] = x0
    start = time.time()
    lpv_counter = np.zeros(Nsim,int)

    # set initial values for x
    x = np.zeros([n_states,Nc+1])
    u = np.zeros([n_controls,Nc]) # change this to u0 instead of 0 maybe
    opti.set_initial(states, x)
    opti.set_initial(controls, u)
    opti.set_value(x_initial,x0_norm)

    for mpciter in np.arange(Nsim):
        start_time_iter = time.time()

        # solve for u and x
        opti.set_value(reference,[0,reference_list_normalized[mpciter]])

        # MPC loop
        while True:
            # determine A,B
            for i in np.arange(Nc):
                opti.set_value(list_A[(n_states*i):(n_states*i+n_states),:],get_A(x[:,i],u[:,i]))
                opti.set_value(list_B[(n_states*i):(n_states*i+n_states),:],get_B(x[:,i],u[:,i]))
            
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
        
        x = horzcat(x[:,1:(Nc+1)],x[:,-1])
        x[:,0] = x0_norm
        u = horzcat(u[:,1:Nc],u[:,-1])
        opti.set_value(x_initial, x0_norm)
        opti.set_initial(states, x)
        opti.set_initial(controls, u)

        # finished mpc time measurement
        end_time_iter = time.time()
        comp_t_log[mpciter] = end_time_iter - start_time_iter

    end = time.time()
    runtime = end - start

    return x_log, u_log, comp_t_log, t, runtime, lpv_counter, reference_list_normalized

if __name__ == "__main__":
    # MPC parameters
    dt = 0.1
    Nc = 10
    Nsim = 100
    dlam = 0.01
    stages = 1
    max_iterations = 10

    # Weight matrices for the cost function
    Q = np.matrix('1,0;0,100')
    R = 1
    
    # Box constraints
    x_min = -20
    x_max = 20
    u_min = -3
    u_max = 3

    # Initial and final values
    x0 = [0,0]
    x_ref = [0, 1.5]
    u_ref = 0

    #x_reference_list = np.sin(np.arange(Nsim+1)*2*np.pi/32)*1
    x_reference_list = np.array([])
    Nsim_remaining = Nsim
    while True:
        Nsim_steps = random.randint(10,15)
        Nsim_remaining = Nsim_remaining - Nsim_steps
        x_reference_list = np.hstack((x_reference_list, np.ones(Nsim_steps)*random.randint(-15,15)/10))

        if Nsim_remaining <= 0:
            break

    # Weight matrices for the cost function
    #Q = 100
    Q = np.matrix('1,0;0,100')
    R = 1

    #sys_Duff = DuffingOscillator()
    sys_unblanced = Systems.NoisyUnbalancedDisc(dt=0.1, sigma_n=[0.33, 0.032])
    I_enc = deepSI.load_system("systems/UnbalancedDisk_dt01_e100_SNR_100")
    
    #opti = NMPC(sys_Duff, I_enc)
    x_log, u_log, comp_t_log, t, runtime, lpv_counter, reference_list_normalized = NMPC(sys_unblanced, I_enc, x_min=x_min, x_max= x_max, u_min=u_min, u_max=u_max, x0=x0,\
         u_ref=u_ref, Q=Q, R=R, dt=dt, dlam=dlam, stages=stages, x_reference_list=x_reference_list, Nc=Nc, Nsim=Nsim, max_iterations=1)

    fig = plt.figure(figsize=[14.0, 3.0])

    plt.subplot(2,3,1)
    plt.plot(np.arange(Nsim+1)*dt, x_log[0,:], label='velocity')
    plt.plot(np.arange(Nsim+1)*dt, np.ones(x_log.shape[1])*x_ref[0], '--', label='reference')
    plt.ylabel("velocity [m/s]") # not sure about the unit
    plt.xlabel("time [s]")
    plt.grid()
    plt.legend()

    plt.subplot(2,3,2)
    plt.plot(np.arange(Nsim+1)*dt, x_log[1,:], label='displacement')
    #plt.plot(np.arange(Nsim+1)*dt, np.ones(x_log.shape[1])*x_ref[1], label='reference')
    plt.plot(np.arange(Nsim+1)*dt, np.hstack((np.zeros(1),x_reference_list[:Nsim])), '--', label='reference') # figure out what the correct hstack should be here
    #plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*u_ref, '--', label='max')
    #plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*u_ref, '--', label='min')
    plt.ylabel("displacement [m]") # not sure about the unit
    plt.xlabel("time [s]")
    plt.grid()
    plt.legend()

    plt.subplot(2,3,3)
    plt.plot(np.arange(Nsim)*dt, u_log[0,:], label='input')
    plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*u_ref, '--', label='reference')
    plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*u_max, 'r-.', label='max')
    plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*u_min, 'r-.', label='min')
    plt.ylabel("input") # not sure about the unit
    plt.xlabel("time [s]")
    plt.grid()
    plt.legend();

    plt.subplot(2,3,4)
    plt.step(np.arange(Nsim), lpv_counter, label='lpv counter')
    plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*max_iterations, '--', label='max iter')
    plt.ylabel("lpv counter") # not sure about the unit
    plt.xlabel("mpciter")
    plt.grid()
    plt.legend();

    plt.subplot(2,3,5)
    plt.step(np.arange(Nsim), comp_t_log, label='computation time')
    plt.plot(np.arange(Nsim), np.ones(Nsim)*dt, '--', label='dt')
    plt.ylabel("computation time") # not sure about the unit
    plt.xlabel("mpciter")
    plt.grid()
    plt.legend();

    # add epsilon plot

    plt.show()
