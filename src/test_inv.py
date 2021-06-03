import attractor
from attractor import Attractor
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread, Lock
import time
import json
import copy

def lse(dots):
    x = dots[1]
    y = dots[0]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m,c


# calcs = []
# res_h = []

# def calc(method,inv):
#     AL1 = Attractor(step=0.00001, num_steps=10000) #1000000
#     AL1.set_invariant_params(inv)
#     AL1.iterator_method(method)
#     calls = AL1.get_counter()
#     # print("calls_f: ", calls)
#     # Get inv func
#     I, err = AL1.get_invariant_err(inv, dt=0.000001)
#     # Cut thirds
#     l = int(I.shape[1] * (1.0 / 3.0))
#     I = I[:, l:-l]
#     err = err[:, l:-l]
#
#     # plt.plot(I[1], I[0])
#     # plt.savefig(f'img/{method}@{inv}.png')
#
#     M = np.mean(I[0])
#     D = np.std(I[0] - M)
#     K, C = lse(I)
#
#     txt = f'{method}@{inv}#{calls}'
#
#     fig = plt.figure()
#     fig.set_facecolor("mintcream")
#
#     ax = fig.gca()
#     ax.plot(I[1], I[0], lw=0.5)
#
#     ax.set_facecolor('mintcream')
#     ax.set_xlabel("X Axis")
#     ax.set_ylabel("Y Axis")
#     ax.set_title(f'{method}@{inv}')
#
#     ax.tick_params(axis='x', colors="orange")
#     ax.tick_params(axis='y', colors="orange")
#
#     ax.savefig(f'img/{txt}:{M}:{D}:{K}:{C}.png')
#
#     ax.show
#
#
#
#     return M, D, K, C
#
#
# def calc_th(calcs, TH, N_TH):
#     for i,ca in enumerate(calcs):
#         if i%N_TH == TH:
#             time.sleep(0.001 * np.pi * TH)
#             print(f'TH:{TH}:{ca}@{i}')
#             res = calc(ca[0],ca[1])
#             # res_h[i].extend(res)
#
#             try:
#                 with open(f'dump/dump_TH:{TH}:{ca}@{i}.json','w') as sf:
#                     json.dump(res,sf)
#             except:
#                 pass
#

methods = Attractor.methods

res = {}



for method in methods:
    for inv in range(1,6):
        # calcs.append((method,inv))
        # res_h.append([])
        #
        # continue
        print("=============================")
        print(f'{method}@{inv}')

        AL1 = Attractor(step=0.0001, num_steps=100) # TODO Шаг выставляется тут!
        AL1.set_invariant_params(inv)
        AL1.iterator_method(method)
        calls = AL1.get_counter()
        print("calls_f: ", calls)
        # Get inv func
        I, err = AL1.get_invariant_err(inv)
        # Cut thirds
        l = int(I.shape[1] * (1.0 / 3.0))
        I = I[:, l:-l]
        err = err[:, l:-l]
        txt = f'{method}@{inv}#{calls}'

        # fig.savefig
        M = np.mean(I[0])
        D = np.std(I[0] - M)
        K, C = lse(I)

        res[str((method,inv))]=[M,D,K,C]

        fig = plt.figure()
        fig.set_facecolor("mintcream")

        ax = fig.gca()
        ax.plot(I[1], I[0], lw=0.5)

        ax.set_facecolor('mintcream')
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_title(f'{method}@{inv}')

        ax.tick_params(axis='x', colors="orange")
        ax.tick_params(axis='y', colors="orange")

        fig.savefig(f'img/{txt}:{M}:{D}:{K}:{C}.png')

        # plt.plot(I[1], I[0])
        # plt.savefig(f'img/{txt}:{M}:{D}:{K}:{C}.png')  # (f'img/{method}@{inv}.png')

        print("M =", M)
        print("D =", D)
        print("K =", K)
        print("C =", C)
        print(res)
        print()
# N_TH = 8
#
# for i in range(N_TH):
#     Thread(target=calc_th,
#            args=(copy.copy(calcs),i, N_TH)).start()

# print(res_h)
with open("res.json", "w") as fp:
    json.dump(res,fp)
print(res)



