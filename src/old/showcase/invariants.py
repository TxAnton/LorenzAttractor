from src.lorenzMethods import AttractorLorenz
import numpy as np

if __name__ == "__main__":
    show = False

    t = 5
    step = 0.0001
    print("steps:", t // step)
    arg = {"s": 10.0, "r": 28.0, "b": 2.667, "step": step, "num_steps": int(t // step), "init_value": (0., 1., 1.05)}

    # arg  ={"s":10.0, "r":28.0, "b":20.0, "step":0.0000001, "num_steps":100000, "init_value":(0., 1., 1.05)}

    ms=[[],[],[],[],[]]
    # Euler
    for n_inv in range (1,6):
        a = AttractorLorenz(**arg)
        a.set_invariant_params(n_inv)
        a.EulerMethod()
        _,_,m=a.get_invariant_err(n_inv)
        print(m)
        ms[n_inv-1].append(m)


    # Midpoint
    for n_inv in range (1,6):
        a = AttractorLorenz(**arg)
        a.set_invariant_params(n_inv)
        a.midpointMethod()
        _,_,m = a.get_invariant_err(n_inv)
        print(m)
        ms[n_inv - 1].append(m)

    # RK4
    for n_inv in range(1, 6):
        a = AttractorLorenz(**arg)
        a.set_invariant_params(n_inv)
        a.RKMethod(arg["num_steps"])
        _,_,m=a.get_invariant_err(n_inv)
        print(m)
        ms[n_inv - 1].append(m)

    # Adam Bashforts
    for n_inv in range(1, 6):
        a = AttractorLorenz(**arg)
        a.set_invariant_params(n_inv)
        a.overclocking(3, False)
        _,_,m=a.get_invariant_err(n_inv)
        print(m)
        ms[n_inv - 1].append(m)

    # Adam Moulton
    for n_inv in range(1, 6):
        a = AttractorLorenz(**arg)
        a.set_invariant_params(n_inv)
        a.overclocking(3, True)
        _,_,m=a.get_invariant_err(n_inv)
        print(m)
        ms[n_inv - 1].append(m)

    # Dorman Prince
    for n_inv in range(1, 6):
        a = AttractorLorenz(**arg)
        a.set_invariant_params(n_inv)
        a.sp_ivp()
        _,_,m=a.get_invariant_err(n_inv)
        print(m)
        ms[n_inv - 1].append(m)


    print(ms)
    print(np.sort(ms))
    print(np.argsort(ms))