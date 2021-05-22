from src.lorenzMethods import AttractorLorenz

if __name__ =="__main__":
    # Euler
    AL1 = AttractorLorenz(10, 28, 2.667, 0.01, 10000, (0., 1., 1.05))
    AL1.EulerMethod()
    AL1.printDots(15)
    AL1.createPNG("PNG/AL1")
    AL1.clearDots()

    # Midpoint
    AL2 = AttractorLorenz(10, 28, 2.667, 0.01, 10000, (0., 1., 1.05))
    AL2.midpointMethod()
    AL2.printDots(15)
    AL2.createPNG("PNG/AL2")
    AL2.clearDots()

    # RK4
    AL3 = AttractorLorenz(10, 28, 2.667, 0.01, 10000, (0., 1., 1.05))
    AL3.RKMethod(10000)
    AL3.printDots(15)
    AL3.createPNG("PNG/AL3")
    AL3.clearDots()

    # Adam Bashforts
    AL4 = AttractorLorenz(10, 28, 2.667, 0.01, 10000, (0., 1., 1.05))
    AL4.overclocking(3)
    AL4.printDots(15)
    AL4.printValFunc(15)
    AL4.createPNG("PNG/AL4")
    AL4.clearDots()
    AL4.clearFunction()