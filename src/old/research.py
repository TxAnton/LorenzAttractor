import matplotlib.pyplot as plt
import numpy as np


from src.lorenzMethods import AttractorLorenz

if __name__ == "__main__":

    t = 50
    step = 0.0001
    print("steps:",t//step)
    arg  ={"s":10.0, "r":28.0, "b":20.0, "step":0.0000001, "num_steps":100000, "init_value":(0., 1., 1.05)}
