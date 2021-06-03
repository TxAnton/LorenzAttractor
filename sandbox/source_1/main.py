import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import timeit
from scipy import interpolate

from sandbox.source_1.Methods.Adams_bashforth import adams_bashforth_4th



from sandbox.source_1.Methods.Adams_moulton import adams_moulton_4th
from sandbox.source_1.Methods.Adams_moulton import rk4_step
from sandbox.source_1.Methods.Euler import euler_test_cycle
from sandbox.source_1.Methods.Euler import euler_test_func
from sandbox.source_1.Methods.Euler_backward import backward_euler_fixed_point
from sandbox.source_1.Methods.RK4 import rk4
from sandbox.source_1.Methods.Runge_kutta_fehlberga import runge_kutta_fehlberga
from sandbox.source_1.Methods.ode_solver import Ode_solver


v0 = 400
alpha = 35 * np.pi / 180
x0 = 0
y0 = 0
vx0 = v0 * np.cos(alpha)
vy0 = v0 * np.sin(alpha)
Y0 = [x0, y0, vx0, vy0]
counter = 0
reverse_counter = 0
t0 = 0.0
tend = 25.8
deltaT = 0.1

gamma = 7800
c_f = 0.47
d = 0.1
rho = 1.29
g = 9.81
k = 3 * c_f * rho / 4 / gamma / d
t = np.arange(t0, tend, deltaT)


def f(Y, t):
    global counter
    v = np.sqrt(Y[2] * Y[2] + Y[3] * Y[3])
    dYdt = [Y[2], Y[3], (-1) * k * v * Y[2], (-1) * k * v * Y[3] - g]
    counter+=1
    return np.array(dYdt)



def f_reverse(t, Y):
    global reverse_counter
    v = np.sqrt(Y[2] * Y[2] + Y[3] * Y[3])
    dYdt = [Y[2], Y[3], (-1) * k * v * Y[2], (-1) * k * v * Y[3] - g]
    reverse_counter+=1
    return np.array(dYdt)


def draw(t, y, col, label):
    plt.plot(t, y, col, label=label)
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()

def draw_diff(t, y, col):
    print("MEAN: ", np.mean(y))
    print("MEDIAN: ", np.median(y))
    print("STD: ", np.std(y))
    print("MAX: ", np.max(np.abs(y)))
    plt.plot(t, y, "b", label="Difference")
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()

sol_back = solve_ivp(f_reverse, [0, 27], Y0, 'DOP853', np.arange(t0, 27, deltaT))
inter = interpolate.interp1d(sol_back.y[0, :], sol_back.y[1, :])

#sol = odeint(f, Y0, t)
#sol = adams_moulton_4th(f, Y0, t)
#sol = rk4(f, Y0, t)
#sol = backward_euler_fixed_point(f, t, Y0)
#sol = adams_bashforth_4th(f, Y0, t)
#sol = runge_kutta_fehlberga(f, t0, Y0, tend, deltaT, 0.000000000001, 0.000000000001)

# sol = np.zeros((len(t), len(Y0)))
# sol[0] = Y0
# i = 1
# integrator = Ode_solver(f, "RK4") ##rk4 or Backward euler##
# integrator.set_params(t=t0, Y0=Y0)
# while integrator.t < tend - deltaT:
#     sol[i] = integrator.integrate(integrator.t + deltaT)
#     i+=1

# sol = np.zeros((len(t), len(Y0)))
# sol[0] = Y0
# i = 1
# integrator = Ode_solver(f, "RK45")
# integrator.set_params(t=t0, Y0=Y0)
# while integrator.t < tend - deltaT:
#     sol[i] = integrator.integrate()
#     i+=1


sol = np.zeros((len(t), len(Y0))) ##AB or AM##
sol[0] = Y0
i = 4
integrator = Ode_solver(f, "AB",)
integrator.set_params(t=t0, h=0.1, Y0=Y0)
sol[1] = integrator.y_arr[1]
sol[2] = integrator.y_arr[2]
sol[3] = integrator.y_arr[3]
while integrator.t < tend - deltaT:
    sol[i] = integrator.integrate(integrator.t + deltaT)
    i+=1


draw(sol[:, 0], sol[:, 1], 'r', "Adams-Moulton")
