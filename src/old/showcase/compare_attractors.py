from src.lorenzMethods import AttractorLorenz
import numpy as np


if __name__ == "__main__":
    show = False
    ms=[]
    t = 10
    step = 0.0005
    eth_step = 0.00001
    print("steps:", t // step)
    arg = {"s": 10.0, "r": 28.0, "b": 2.667, "step": step, "num_steps": int(t // step), "init_value": (0., 1., 1.05)}
    eth_arg = {"s": 10.0, "r": 28.0, "b": 2.667, "step": eth_step, "num_steps": int(t // eth_step),
               "init_value": (0., 1., 1.05)}
    # arg  ={"s":10.0, "r":28.0, "b":20.0, "step":0.0000001, "num_steps":100000, "init_value":(0., 1., 1.05)}

    a_ethalon = AttractorLorenz(**eth_arg)
    # a_ethalon.sp_ivp()
    a_ethalon.sp_ivp() # TODO SET ETHALON HERE
    # a_ethalon.createPNG(do_show=False)

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

    # Dorman Prince
    a_dp = AttractorLorenz(**arg)
    a_dp.sp_ivp()
    # ms.append(m)

    _,m=a_ethalon.compare(a_eu, do_show=show, do_save=True, adopt_time_scale=True)
    ms.append(m)
    _,m=a_ethalon.compare(a_mp, do_show=show, do_save=True, adopt_time_scale=True)
    ms.append(m)
    _,m=a_ethalon.compare(a_rk, do_show=show, do_save=True, adopt_time_scale=True)
    ms.append(m)
    _,m=a_ethalon.compare(a_ab, do_show=show, do_save=True, adopt_time_scale=True)
    ms.append(m)
    _,m=a_ethalon.compare(a_am, do_show=show, do_save=True, adopt_time_scale=True)
    ms.append(m)
    _,m=a_ethalon.compare(a_dp, do_show=show, do_save=True, adopt_time_scale=True)
    ms.append(m)

    print(ms)
    print(np.argsort(ms))
