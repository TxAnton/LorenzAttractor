from lorenzMethods import AttractorLorenz
from attractor import Attractor
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Euler
    AL1 = Attractor()
    AL1.iterator_method("EUL1")
    AL1.show("img/Euler.png", "Euler method", True)
    print("Вызовов f: ", AL1.get_counter())

    # Midpoint
    AL2 = Attractor()
    AL2.iterator_method("MIDP2")
    AL2.show("img/Midpoint", "Midpoint method", True)
    print("Вызовов f: ", AL2.get_counter())

    # RK4
    AL3 = Attractor()
    AL3.iterator_method("RK4")
    AL3.show("img/RK4.png", "RK4 method", True)
    print("Вызовов f: ", AL3.get_counter())

    # Adam Bashforts
    AL4 = Attractor()
    AL4.iterator_method("AB4")
    AL4.show("img/Adam_Bashforts_.png", "Adam Bashforts method", True)
    print("Вызовов f: ", AL4.get_counter())

    # Adam Moulton
    AL5 = Attractor()
    AL5.iterator_method("AM4")
    AL5.show("img/Adam_Moulton.png", "Adam Moulton method (+ RK4)", True)
    print("Вызовов f: ", AL5.get_counter())

    # Adam Moulton 5
    AL6 = Attractor()
    AL6.iterator_method("ABM5")
    AL6.show("img/Adam_Bashforts_Moulton.png", "Adam Moulton 5 method (Bashforts + RK4)", True)
    print("Вызовов f: ", AL6.get_counter())

    # # DOP853
    # AL7 = Attractor()
    # AL7.sp_ivp()
    # AL7.show("img/DOP853", "DOP853", True)
    # print("Вызовов f: ", AL7.get_counter())
    #
    # err = AL3.compare(AL2)
    # plt.plot(err[1], err[0])
    # plt.show()
    #
    # err = AL6.compare(AL1)
    # plt.plot(err[1], err[0])
    # plt.show()
