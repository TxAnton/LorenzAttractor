import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import timeit
from scipy import interpolate

from src.lorenzMethods import AttractorLorenz

if __name__ =="__main__":

    def f(t, Y):  # System in standart form
        s = 10
        r = 28
        b = 2.667
        d1 = s * (Y[1] - Y[0])
        d2 = r * Y[0] - Y[1] - Y[0] * Y[2]
        d3 = Y[0] * Y[1] - b * Y[2]
        return [d1,d2,d3]


    h = 0.02
    Y0 = np.array([1,2,3])
    # n_steps = 1000
    max_t = 100.0

    arg = {"s": 10, "r": 28, "b": 2.667, "step": 0.00001, "num_steps": 100000, "init_value": (0., 1., 1.05)}
    AL1 = AttractorLorenz(**arg)
    sol_back = solve_ivp(AL1, [0, max_t], Y0, 'DOP853', np.arange(0.0, max_t, h))
    y = sol_back.y

    fig = plt.figure()
    fig.set_facecolor("mintcream")

    ax = fig.gca(projection='3d')
    ax.plot(y[0], y[1], y[2], lw=0.5)

    ax.set_facecolor('mintcream')
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    ax.tick_params(axis='x', colors="orange")
    ax.tick_params(axis='y', colors="orange")
    ax.tick_params(axis='z', colors="orange")

    plt.show()

