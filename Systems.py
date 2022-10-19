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

class NoisyUnbalancedDisc(deepSI.System_deriv):
    def __init__(self, dt=0.025, sigma_n=[1, 0.1]):
        super(NoisyUnbalancedDisc, self).__init__(nx=2, dt=dt)
        self.g = 9.80155078791343
        self.J = 0.000244210523960356
        self.Km = 10.5081817407479
        self.I = 0.0410772235841364
        self.M = 0.0761844495320390
        self.tau = 0.397973147009910
        self.sigma_n = sigma_n

    def deriv(self,x,u):
        z1,z2 = x
        dz1 = -self.M*self.g*self.I/self.J*np.sin(z2) - 1/self.tau*z1 + self.Km/self.tau*u
        dz2 = z1
        return [dz1,dz2]

    def h(self,x,u):
        return x[1] #+ np.hstack((np.random.normal(0, self.sigma_n[0], 1), np.random.normal(0, self.sigma_n[1], 1)))
        #return x[0], (x[1] + np.pi)%2.0*np.pi - np.pi

class DiscreteUnbalancedDisc(deepSI.System_ss):
    def __init__(self, dt=0.025):
        super(DiscreteUnbalancedDisc, self).__init__(nx=2, nu=1)
        self.g = 9.80155078791343
        self.J = 0.000244210523960356
        self.Km = 10.5081817407479
        self.I = 0.0410772235841364
        self.M = 0.0761844495320390
        self.tau = 0.397973147009910
        self.Ts = dt

    def f(self,x,u):
        z1,z2 = x
        dz1 = -self.Ts*self.M*self.g*self.I/self.J*np.sin(z2) - self.Ts/self.tau*z1 + self.Ts*self.Km/self.tau*u + z1
        dz2 = self.Ts*z1 + z2
        return [dz1,dz2]

    def h(self,x,u):
        return x #+ np.hstack((np.random.normal(0, 0.47, 1), np.random.normal(0, 0.044, 1)))