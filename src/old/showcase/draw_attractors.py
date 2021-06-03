import matplotlib.pyplot as plt
import numpy as np


from src.lorenzMethods import AttractorLorenz

if __name__ == "__main__":

    show = True
    t = 50
    step = 0.001
    print("steps:",t//step)
    arg = {"s": 10.0, "r": 28.0, "b": 2.667, "step": step, "num_steps": int(t//step), "init_value": (0., 1., 1.05)}
    # arg  ={"s":10.0, "r":28.0, "b":20.0, "step":0.0000001, "num_steps":100000, "init_value":(0., 1., 1.05)}

    # Euler
    a_eu = AttractorLorenz(**arg)
    a_eu.EulerMethod()

    # Midpoint
    a_mp = AttractorLorenz(**arg)
    a_mp.midpointMethod()

    # RK4
    a_rk = AttractorLorenz(**arg)
    a_rk.RKMethod(arg["num_steps"])

    # Adam Bashforts
    a_ab = AttractorLorenz(**arg)
    a_ab.overclocking(3, False)

    # Adam Moulton
    a_am = AttractorLorenz(**arg)
    a_am.overclocking(3, True)

    a_dp = AttractorLorenz(**arg)
    a_dp.sp_ivp()

    a_eu.createPNG("draw_eu",do_show=show)
    a_mp.createPNG("draw_mp",do_show=show)
    a_rk.createPNG("draw_rk",do_show=show)
    a_ab.createPNG("draw_ab",do_show=show)
    a_am.createPNG("draw_am",do_show=show)
    a_dp.createPNG("draw_dop", do_show=show)



