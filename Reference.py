import numpy as np
from matplotlib import pyplot as plt
import random
import deepSI

def stackReferences(references_tuple):
    return np.vstack(references_tuple)

def randomLevelReference(Nsim, nt_range, level_range):
    x_reference_list = np.array([])
    Nsim_remaining = Nsim
    while True:
        Nsim_steps = random.randint(nt_range[0],nt_range[1])
        Nsim_remaining = Nsim_remaining - Nsim_steps
        x_reference_list = np.hstack((x_reference_list, np.ones(Nsim_steps)*random.randint(level_range[0]*10,level_range[1]*10)/10))

        if Nsim_remaining <= 0:
            x_reference_list = x_reference_list[:Nsim]
            break
    return x_reference_list

if __name__ == "__main__":
    # determine state references
    #x1_reference_list = np.ones((Nsim))
    # x1_reference_list = np.hstack((np.zeros((50)),np.ones((50))))
    # x1_reference_list = np.array([])
    # Nsim_remaining = Nsim
    # while True:
    #     Nsim_steps = random.randint(10,15)
    #     Nsim_remaining = Nsim_remaining - Nsim_steps
    #     x1_reference_list = np.hstack((x1_reference_list, np.ones(Nsim_steps)*random.randint(-10,10)/10))

    #     if Nsim_remaining <= 0:
    #         x1_reference_list = x1_reference_list[:Nsim]
    #         break

    #x_reference_list = np.vstack((x0_reference_list, x1_reference_list))
    #x_reference_list = stackReferences((np.zeros((500)), randomLevelReference(500, [25,30], [-1.5, 1.5])))
    x_reference_list = stackReferences((np.zeros((500)), 0.5*deepSI.deepSI.exp_design.multisine(500, pmax=11, n_crest_factor_optim=20)))
    np.save("references/multiSineNsim500.npy", x_reference_list)
    print(x_reference_list.shape)
    plt.plot(x_reference_list[1,:])
    plt.show()
    #print(x_reference_list)
    #print(x_reference_list)