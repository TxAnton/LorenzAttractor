from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.integrate import ode

#def jak

def de(t,x):
	x1,x2,x3 = x
	s=10.0
	r=28.0
	b = 8.0/3.0
	y1 = s*(x2-x1)
	y2 = x1*(r-x3)-x2
	y3 = x1*x2-b*x3
	return [y1,y2,y3]


ge = ode(de).set_integrator("dopri5", atol = 1e-6, rtol = 1e-3, nsteps = 100, first_step = 0, max_step = np.inf, safety = 0.9, ifactor = 1, dfactor = 1, beta = 1, verbosity = 1).integrate(np.linspace(0,10,10000),True)

print(ge)

