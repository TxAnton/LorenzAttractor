from attractor import Attractor

if __name__ == "__main__":
    # Euler
    AL1 = Attractor()
    AL1.call_method("EUL1")
    AL1.show("img/Euler", "Euler's method", False, True)
    print("Вызовов EUL1 f: ", AL1.get_counter())

    # Midpoint
    AL2 = Attractor()
    AL2.call_method("MIDP2")
    AL2.show("img/Midpoint", "Midpoint method", False, True)
    print("Вызовов MIDP2 f: ", AL2.get_counter())

    # RK4
    AL3 = Attractor()
    AL3.call_method("RK4")
    AL3.show("img/RK4", "RK4 method", False, True)
    print("Вызовов RK4 f: ", AL3.get_counter())

    # Adams Bashforts
    AL4 = Attractor()
    AL4.call_method("AB4")
    AL4.show("img/Adams_Bashforts_", "Adams Bashfort method", False, True)
    print("Вызовов AB4 f: ", AL4.get_counter())

    # Adams Moulton
    AL5 = Attractor()
    AL5.call_method("AM4")
    AL5.show("img/Adams_Moulton", "Adams Moulton method", False, True)
    print("Вызовов AM4 f: ", AL5.get_counter())

    # Adams Moulton 5
    AL6 = Attractor()
    AL6.call_method("ABM5")
    AL6.show("img/Adams_Bashforts_Moulton", "Adams-Bashfort-Moulton", False, True)
    print("Вызовов ABM5 f: ", AL6.get_counter())

