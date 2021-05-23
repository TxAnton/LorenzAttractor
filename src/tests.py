import matplotlib.pyplot as plt
import numpy as np


from src.lorenzMethods import AttractorLorenz

if __name__ == "__main__":

    t = 50
    step = 0.0001
    print("steps:",t//step)
    arg  ={"s":10.0, "r":28.0, "b":20.0, "step":0.0000001, "num_steps":100000, "init_value":(0., 1., 1.05)}


    # Euler
    AL1 = AttractorLorenz(s=10,r=28,b=20,step = step, num_steps=int(t//step))
    AL1.set_invariant_params(5)
    AL1.RKMethod()
    AL1.show()
    I,r = AL1.get_invariant_err(5)

    plt.title("5  инвариант методом рунге куты")
    plt.plot(r[1],r[0])

    plt.show()

    # I,r = AL1.get_invariant_err(1)
    # plt.plot(I)
    # AL1.printDots(15)
    # AL1.createPNG("PNG/AL1")
    # AL1.clearDots()

    # Midpoint
    AL2 = AttractorLorenz(**arg)
    AL2.midpointMethod()
    # AL2.printDots(15)
    # AL2.createPNG("PNG/AL2")
    AL2.clearDots()

    # RK4
    AL3 = AttractorLorenz(**arg)
    AL3.RKMethod(arg["num_steps"])
    # AL3.printDots(15)
    # AL3.createPNG("PNG/AL3")
    # AL3.clearDots()

    # Adam Bashforts
    AL4 = AttractorLorenz(**arg)
    AL4.overclocking(3)
    # AL4.printDots(15)
    # AL4.printValFunc(15)
    # AL4.createPNG("PNG/AL4")
    # AL4.clearDots()
    # AL4.clearFunction()

    AL5 = AttractorLorenz(**arg)
    AL5.set_invariant_params(1)
    AL5.RKMethod(arg["num_steps"])
    # AL5.createPNG("AL5",True)
    # AL5.sp_ivp()
    err = AL5.compare(AL1)
    #  = AL1.compare(AL2)

    plt.title("расстояние меж РК4 и эйлера")
    plt.plot(err[1],err[0])
    plt.show()

    err = AL5.compare(AL4)
    plt.title("расстояние меж РК4 и Адама-Башфорта")
    plt.plot(err[1], err[0])
    plt.show()





    exit(2)
