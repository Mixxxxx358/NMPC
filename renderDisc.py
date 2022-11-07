import Systems
import deepSI
from matplotlib import pyplot as plt
from matplotlib import patches as ptc
import numpy as np
import torch
import time
from torch import nn
import qpsolvers as qp


class I_encoder(deepSI.fit_systems.SS_encoder):
    def __init__(self, nx = 2, na=2, nb=2, feedthrough=False) -> None:
        super().__init__(nx=nx, na=na, nb=nb, feedthrough=feedthrough)

    def init_nets(self, nu, ny): # a bit weird
        ny = ny if ny is not None else 1
        nu = nu if nu is not None else 1
        self.encoder = self.e_net(self.nb*nu+self.na*ny, self.nx, n_nodes_per_layer=self.e_n_nodes_per_layer, n_hidden_layers=self.e_n_hidden_layers, activation=self.e_activation)
        self.fn =      self.f_net(self.nx+nu,            self.nx, n_nodes_per_layer=self.f_n_nodes_per_layer, n_hidden_layers=self.f_n_hidden_layers, activation=self.f_activation)
        hn_in = self.nx + nu if self.feedthrough else self.nx
        self.hn =      nn.Identity(hn_in)#

if __name__ == "__main__":
    u = deepSI.deepSI.exp_design.multisine(100000, pmax=49999, n_crest_factor_optim=20)
    u = np.clip(u*4.0, -8.0, 8.0)
    
    system = Systems.SinCosUnbalancedDisc(dt=0.1)
    I_enc = deepSI.load_system("systems/ObserverUnbalancedDisk_dt01_nab_4_e200")

    data = system.apply_experiment(deepSI.System_data(u=u))

    fig1 = plt.figure(figsize=[6.0, 6.0])
    theta = np.linspace(0, 2*np.pi, 150)
    radius = 1.0
    a = radius*np.cos(theta)
    b = radius*np.sin(theta)

    for i in range(2,100):
        plt.cla()
        plt.grid()
        plt.axis([-1.2, 1.2, -1.2, 1.2])
        plt.plot(a, b)
        plt.scatter(data.y[i,1], -data.y[i,2])
        plt.scatter(data.y[i-1,1], -data.y[i-1,2])
        plt.scatter(data.y[i-2,1], -data.y[i-2,2])
        plt.pause(0.1)

    plt.show()