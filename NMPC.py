import deepSI
import numpy as np
from casadi import *
from torch import nn
from Systems import DuffingOscillator
from matplotlib import pyplot as plt

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

def CasADiFn(ss_enc, x, u):
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

def NMPC(x_min, x_max, u_min, u_max, x0, x_ref, u_ref, Q, R, dt, Nc, Nsim, dlam, stages, reference_list):
    sys_Duff = DuffingOscillator()
    I_enc = deepSI.load_system("systems/FullOutputSS_dt01_e300")

    # declared sym variables
    x = MX.sym("x",I_enc.nx,1)
    nu = I_enc.nu if I_enc.nu is not None else 1
    u = MX.sym("u",nu,1)

    # convert torch nn to casadi function
    rhs = CasADiFn(I_enc, x, u)
    f = Function('f', [x, u], [rhs])
    correction = f([0,0], 0)
    rhs_c = rhs - correction
    f_c = Function('f_c', [x, u], [rhs_c])

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

    objective = 0 #maybe add some soft bounds on the states and inputs
    for i in np.arange(Nc):
        #opti.subject_to(states[:,i+1] == lpv_rk4(states[:,i],controls[:,i],\
        #    list_A[(n_states*i):(n_states*i+n_states),:],list_B[(n_states*i):(n_states*i+n_states),:],dt, correction)) # should this be changed to discrete time model
        opti.subject_to(states[:,i+1] == list_A[(n_states*i):(n_states*i+n_states),:]@states[:,i] \
            + list_B[(n_states*i):(n_states*i+n_states),:]@controls[:,i] + correction)
        objective = (objective + 
                        #mtimes(mtimes((states[:,i]-x_ref).T,Q),(states[:,i]-x_ref)) +
                        #mtimes(mtimes((controls[:,i]-u_ref).T,R),(controls[:,i]-u_ref)))
                        mtimes(mtimes((states[:,i]-reference).T,Q),(states[:,i]-reference)) +
                        mtimes(mtimes((controls[:,i]-u_ref).T,R),(controls[:,i]-u_ref)))

    opti.minimize(objective)
    
    # determine reference
    #reference_list = np.sin(np.arange(Nsim+1)*2*np.pi/32)*1

    # logging list
    u_log = np.zeros([n_controls,Nsim])
    t = np.zeros(Nsim)
    t0 = 0
    x_log = np.zeros([n_states,Nsim+1])
    start = time.time()

    # iteration values
    lpv_counter = np.zeros(Nsim,int)

    # set initial values for x
    x = np.zeros([n_states,Nc+1]) # np.repeat(x0, NC+1)change this to x0 instead of 0
    u = np.zeros([n_controls,Nc]) # change this to u0 instead of 0 maybe
    opti.set_initial(states, x)
    opti.set_initial(controls, u)
    opti.set_value(x_initial,x0)
    #x[:,0] = np.ravel(x0)

    norm = I_enc.norm
    max_iterations = 10

    for mpciter in np.arange(Nsim):
        # determine A,B
        for i in np.arange(Nc):
            opti.set_value(list_A[(n_states*i):(n_states*i+n_states),:],get_A(x[:,i],u[:,i]))
            opti.set_value(list_B[(n_states*i):(n_states*i+n_states),:],get_B(x[:,i],u[:,i]))
        
        # solve for u and x
        opti.set_value(reference,[0,reference_list[mpciter]])
        sol = opti.solve();
        u_old = u
        u = np.reshape(sol.value(controls),[n_controls,Nc])
        #for i in np.arange(Nc):
        #    x[:,i+1] = np.ravel(my_rk4(x[:,i],u[:,i],f,dt),order='F') # change this to f_c of f?
        x = np.reshape(sol.value(states),[n_states,Nc+1])
        opti.set_initial(states, x)
        opti.set_initial(controls, u)

        lpv_counter[mpciter] += 1

        while (lpv_counter[mpciter] < max_iterations) and (np.linalg.norm(u-u_old) > 1e-5):
            # determine A,B
            for i in np.arange(Nc):
                opti.set_value(list_A[(n_states*i):(n_states*i+n_states),:],get_A(x[:,i],u[:,i]))
                opti.set_value(list_B[(n_states*i):(n_states*i+n_states),:],get_B(x[:,i],u[:,i]))
            
            # solve for u and x
            sol = opti.solve()
            u_old = u
            u = np.reshape(sol.value(controls),[n_controls,Nc])

            # simulate next step using rk4 over non correction casadi function
            #for i in np.arange(Nc):
            #    x[:,i+1] = np.ravel(my_rk4(x[:,i],u[:,i],f,dt),order='F')
            x = np.reshape(sol.value(states),[n_states,Nc+1]) # change this to nn maybe
            
            # set new x and u values into optimizer
            opti.set_initial(states, x)
            opti.set_initial(controls, u)

            lpv_counter[mpciter] += 1  

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
        x_log[:,mpciter] = x[:,0]
        u_log[:,mpciter] = u[:,0]
        
        # denormalize x and u and run system step
        #x_denormalized = norm.inverse_transform(deepSI.System_data(y=x0)) # make sure
        x_denormalized = norm.ystd*x0 + norm.y0
        #u_denormalized = norm.inverse_transform(deepSI.System_data(u=u[0,0]))
        u_denormalized = norm.ustd*u[0,0] + norm.u0
        #x0 = x[:,1] # change this to real system step

        # normalize output of system step
        x_denormalized = sys_Duff.f(x_denormalized, u_denormalized)
        x0 = norm.transform(deepSI.System_data(y=x_denormalized))
        x0 = (x_denormalized - norm.y0)/norm.ystd

        x_log[:,mpciter+1] = x0
        
        #!!! add x shift and u shift for hot start
        x = horzcat(x[:,1:(Nc+1)],x[:,-1])
        x[:,0] = x0
        u = horzcat(u[:,1:Nc],u[:,-1])
        opti.set_value(x_initial, x0)
        opti.set_initial(states, x)
        opti.set_initial(controls, u)

    end = time.time()
    runtime = end - start

    return x_log, u_log, t, runtime, lpv_counter

if __name__ == "__main__":
    # MPC parameters
    dt = 1/10
    Nc = 10
    Nsim = 30
    dlam = 0.01
    stages = 1
    
    # Box constraints
    x_min = -4
    x_max = 4
    u_min = -2
    u_max = 2

    # Initial and final values
    x0 = [0,0]
    x_ref = [0, 1.5]
    reference_list = np.sin(np.arange(Nsim+1)*2*np.pi/32)*1
    u_ref = 0

    # Weight matrices for the cost function
    #Q = 100
    Q = np.matrix('1,0;0,100')
    R = 0.001
    
    x_log, u_log, t, runtime, lpv_counter = NMPC(x_min, x_max, u_min, u_max, x0, x_ref, u_ref, Q, R, dt, Nc, Nsim, dlam, stages, reference_list)

    fig = plt.figure(figsize=[14.0, 3.0])

    plt.subplot(1,3,1)
    plt.plot(np.arange(Nsim+1)*dt, x_log[0,:], label='velocity')
    plt.plot(np.arange(Nsim+1)*dt, np.ones(x_log.shape[1])*x_ref[0], label='reference')
    plt.ylabel("velocity [m/s]") # not sure about the unit
    plt.xlabel("time [s]")
    plt.grid()
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(np.arange(Nsim+1)*dt, x_log[1,:], label='displacement')
    #plt.plot(np.arange(Nsim+1)*dt, np.ones(x_log.shape[1])*x_ref[1], label='reference')
    plt.plot(np.arange(Nsim+1)*dt, reference_list, label='reference') # figure out what the correct hstack should be here
    plt.ylabel("displacement [m]") # not sure about the unit
    plt.xlabel("time [s]")
    plt.grid()
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(np.arange(Nsim)*dt, u_log[0,:], label='input')
    plt.plot(np.arange(Nsim)*dt, np.ones(Nsim)*u_ref, label='reference')
    plt.ylabel("input") # not sure about the unit
    plt.xlabel("time [s]")
    plt.grid()
    plt.legend();

    plt.show()