import deepSI
import numpy as np

class DuffingOscillator(deepSI.System_deriv):
    def __init__(self, dt=0.1):
        super(DuffingOscillator, self).__init__(nx=2, dt=dt)
        self.alpha = 1
        self.beta = 5
        self.gamma = 1
        self.delta = 0.02
        self.omega = 0.5

    def deriv(self,x,u):
        z1,z2 = x
        dz1 = self.gamma*u - self.delta*z1 - self.alpha*z2 - self.beta*(z2**3)
        dz2 = z1
        return [dz1,dz2]

    def h(self,x,u):
        return x[1]

class FullStateDuffingOscillator(deepSI.System_deriv):
    def __init__(self):
        super(FullStateDuffingOscillator, self).__init__(nx=2, dt=0.1)
        self.alpha = 1
        self.beta = 5
        self.gamma = 1
        self.delta = 0.02
        self.omega = 0.5

    def deriv(self,x,u):
        z1,z2 = x
        dz1 = self.gamma*u - self.delta*z1 - self.alpha*z2 - self.beta*(z2**3)
        dz2 = z1
        return [dz1,dz2]

    def h(self,x,u):
        return x

class UnbalancedDisc(deepSI.System_deriv):
    def __init__(self, dt=0.025):
        super(UnbalancedDisc, self).__init__(nx=2, dt=dt)
        self.g = 9.80155078791343
        self.J = 0.000244210523960356
        self.Km = 10.5081817407479
        self.I = 0.0410772235841364
        self.M = 0.0761844495320390
        self.tau = 0.397973147009910

    def deriv(self,x,u):
        z1,z2 = x
        dz1 = -self.M*self.g*self.I/self.J*np.sin(z2) - 1/self.tau*z1 + self.Km/self.tau*u
        dz2 = z1
        return [dz1,dz2]

    def h(self,x,u):
        return x