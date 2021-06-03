import attractor
from attractor import Attractor
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread, Lock
import time
import json
import copy



methods = Attractor.methods


# s = {"('EUL1', 1)": [-20.999978365291824, 4.223627160613191e-06, 0.00043892622374454884, -21.00000031158106], "('EUL1', 2)": [2.1024999901149104, 1.728660045470213e-09, -1.7960649654137428e-07, 2.102499999095226], "('EUL1', 3)": [1.784157995254503, 0.11131241594421473, -11.528596839747868, 2.360587260812054], "('EUL1', 4)": [0.4999999987468464, 2.525572443310224e-10, -2.624567090807205e-08, 0.5000000000591287], "('EUL1', 5)": [112.39999535215067, 8.940833433633235e-07, -9.291568008397273e-05, 112.39999999793001], "('MIDP2', 1)": [-21.000000000014737, 2.942411065936392e-12, -3.053178672128565e-10, -20.999999999999467], "('MIDP2', 2)": [2.1025000000000413, 4.3448787279132926e-14, -3.57825277998172e-12, 2.1025000000002216], "('MIDP2', 3)": [1.7841580960640644, 0.11131239155187896, -11.528594299780138, 2.360587234623356], "('MIDP2', 4)": [0.4999999999999841, 5.353062098428974e-15, -1.5024703357171836e-13, 0.49999999999999145], "('MIDP2', 5)": [112.39999999999938, 9.309487505846372e-13, 2.929653182326727e-11, 112.3999999999979], "('RK4', 1)": [-21.00000000000052, 2.0904742404287742e-13, -1.985678568836432e-11, -20.99999999999952], "('RK4', 2)": [2.1025000000000413, 4.3745525795669287e-14, -3.648586767389185e-12, 2.1025000000002247], "('RK4', 3)": [1.7841580960640504, 0.11131239155188279, -11.528594299780531, 2.3605872346233605], "('RK4', 4)": [0.4999999999999841, 5.330395405198325e-15, -1.4645586621769874e-13, 0.49999999999999123], "('RK4', 5)": [112.39999999999873, 8.977882754189427e-13, 1.557057704601589e-11, 112.39999999999804], "('AB4', 1)": [-21.00000000000052, 2.0755681294601098e-13, -1.9615755174141162e-11, -20.999999999999535], "('AB4', 2)": [2.1025000000000413, 4.3745525795669287e-14, -3.648586767389185e-12, 2.1025000000002247], "('AB4', 3)": [1.7841580960640504, 0.11131239155188279, -11.528594299780531, 2.3605872346233605], "('AB4', 4)": [0.4999999999999841, 5.330389697477995e-15, -1.4645586621769874e-13, 0.49999999999999123], "('AB4', 5)": [112.39999999999873, 9.003054943148962e-13, 1.524973775919638e-11, 112.39999999999806], "('AM4', 1)": [-21.00000000000052, 2.0785304321719623e-13, -1.9696165520400303e-11, -20.99999999999953], "('AM4', 2)": [2.1025000000000413, 4.3745525795669287e-14, -3.648586767389185e-12, 2.1025000000002247], "('AM4', 3)": [1.7841580960640504, 0.11131239155188279, -11.528594299780531, 2.3605872346233605], "('AM4', 4)": [0.4999999999999841, 5.330363372510478e-15, -1.4645586621769874e-13, 0.49999999999999123], "('AM4', 5)": [112.39999999999873, 8.977889599893469e-13, 1.557057704601589e-11, 112.39999999999804], "('ABM5', 1)": [-21.000000000000515, 2.079475707704951e-13, -1.9696005286303425e-11, -20.999999999999527], "('ABM5', 2)": [2.102500000000041, 4.374652295564811e-14, -3.638535474106792e-12, 2.1025000000002243], "('ABM5', 3)": [1.7841580960640504, 0.11131239155188279, -11.528594299780531, 2.3605872346233605], "('ABM5', 4)": [0.4999999999999841, 5.330363372510478e-15, -1.4645586621769874e-13, 0.49999999999999123], "('ABM5', 5)": [112.39999999999871, 8.977156080976748e-13, 1.491478302812907e-11, 112.39999999999806]}
# s = {"('EUL1', 1)": [-20.99783657614906, 0.00042245545295634595, 0.043893572811398114, -21.000031035321765], "('EUL1', 2)": [2.102499011564858, 1.7291086486007592e-07, -1.796175224234876e-05, 2.1024999095626606], "('EUL1', 3)": [1.7842010344684969, 0.11132652468535849, -11.527754813290036, 2.360531136358932], "('EUL1', 4)": [0.4999998746936982, 2.5260427108444258e-08, -2.6245361487303663e-06, 0.5000000059073829], "('EUL1', 5)": [112.39953524964533, 8.942789188972226e-05, -0.009291761093627064, 112.39999979124124], "('MIDP2', 1)": [-21.000000144024476, 2.7582655501499636e-08, -2.865899847223064e-06, -21.00000000074382], "('MIDP2', 2)": [2.1025000000105445, 2.5135227930011065e-12, 2.611396380461127e-10, 2.1024999999974887], "('MIDP2', 3)": [1.7842111146321304, 0.11132408490795923, -11.527500810038298, 2.3605285176299944], "('MIDP2', 4)": [0.4999999999995874, 6.883282874676326e-14, -7.1470022767797e-12, 0.49999999999994477], "('MIDP2', 5)": [112.40000000633883, 1.2019327446683545e-09, 1.2488263221438016e-07, 112.40000000009535], "('RK4', 1)": [-20.999999999999844, 2.095937235346319e-14, -1.1983986486325192e-12, -20.999999999999783], "('RK4', 2)": [2.1025000000000063, 3.203913448601721e-15, 3.234987758947289e-13, 2.10249999999999], "('RK4', 3)": [1.7842111144880735, 0.11132408495044677, -11.527500814448825, 2.360528517706443], "('RK4', 4)": [0.4999999999999991, 9.278263154673656e-16, 3.9303615547204284e-14, 0.4999999999999971], "('RK4', 5)": [112.40000000000036, 7.553844084754701e-14, 5.690755110691073e-12, 112.40000000000006], "('AB4', 1)": [-20.99999999999985, 2.139836030559994e-14, -1.3532643581836433e-12, -20.99999999999979], "('AB4', 2)": [2.1025000000000063, 3.203913448601721e-15, 3.234987758947289e-13, 2.10249999999999], "('AB4', 3)": [1.784211114488074, 0.11132408495044673, -11.527500814448825, 2.360528517706443], "('AB4', 4)": [0.49999999999999917, 9.284691615499599e-16, 3.936469960757356e-14, 0.4999999999999972], "('AB4', 5)": [112.40000000000036, 7.561954219928461e-14, 5.690755110691073e-12, 112.40000000000006], "('AM4', 1)": [-20.999999999999833, 1.9945089646836385e-14, -9.954358593162615e-13, -20.99999999999979], "('AM4', 2)": [2.1025000000000063, 3.203913448601721e-15, 3.234987758947289e-13, 2.10249999999999], "('AM4', 3)": [1.7842111144880735, 0.11132408495044677, -11.527500814448825, 2.360528517706443], "('AM4', 4)": [0.49999999999999906, 9.135991939453786e-16, 3.2995486560324294e-14, 0.49999999999999734], "('AM4', 5)": [112.40000000000036, 7.553844084754701e-14, 5.690755110691073e-12, 112.40000000000006], "('ABM5', 1)": [-20.999999999999833, 1.9927389279614424e-14, -9.954358593162615e-13, -20.99999999999979], "('ABM5', 2)": [2.1025000000000063, 3.1143287594276894e-15, 3.172329373852388e-13, 2.10249999999999], "('ABM5', 3)": [1.7842111144880735, 0.11132408495044677, -11.527500814448825, 2.360528517706443], "('ABM5', 4)": [0.4999999999999991, 9.278408270201495e-16, 3.9303615547204284e-14, 0.4999999999999971], "('ABM5', 5)": [112.40000000000036, 7.553844084754701e-14, 5.690755110691073e-12, 112.40000000000006]}
s = {"('EUL1', 1)": [-20.997868872983947, 0.0004223544301549088, 0.4305034982384656, -20.99999986530023], "('EUL1', 2)": [2.102498960077725, 2.0580382724941075e-07, -0.00020977466861785276, 2.102499998462336], "('EUL1', 3)": [2.0947086803286314, 0.001995796977689265, -2.0325247805613778, 2.1047696779924108], "('EUL1', 4)": [0.499999881780706, 2.3552308423043548e-08, -2.4006724851644253e-05, 0.5000000006139941], "('EUL1', 5)": [112.39953904660467, 9.13340065296067e-05, -0.09309623972171643, 112.3999998729913], "('MIDP2', 1)": [-21.000001438353873, 2.847564214586938e-07, -0.0002902506145820754, -21.000000001613337], "('MIDP2', 2)": [2.1025000000472396, 1.1531744083951329e-11, 1.1747923002897235e-08, 2.1024999999890883], "('MIDP2', 3)": [2.094715168279857, 0.001994422273802311, -2.0311230158171063, 2.104769227208152], "('MIDP2', 4)": [0.4999999999953215, 9.152736645039293e-13, -9.329631890566863e-10, 0.4999999999999397], "('MIDP2', 5)": [112.40000006375942, 1.2615511776411471e-08, 1.285891480136009e-05, 112.40000000010782], "('RK4', 1)": [-21.00000000000029, 5.92426706872332e-14, -5.974820780548574e-11, -21.000000000000004], "('RK4', 2)": [2.1025000000000014, 3.275794329645676e-16, 3.734339100859723e-14, 2.1025000000000014], "('RK4', 3)": [2.094715167655526, 0.001994422414793105, -2.0311231595190358, 2.1047692272951455], "('RK4', 4)": [0.4999999999999999, 1.1199048642455407e-16, -1.391593982808325e-13, 0.5000000000000006], "('RK4', 5)": [112.4, 1.6252387855053684e-14, 1.5054393136641803e-12, 112.40000000000002], "('AB4', 1)": [-21.00000000001323, 2.7622277208553537e-12, -2.8133623736512504e-09, -20.99999999999931], "('AB4', 2)": [2.102499999999993, 1.8904386449847918e-15, -1.9721893328091685e-12, 2.1025000000000036], "('AB4', 3)": [2.0947151676555262, 0.001994422414793017, -2.0311231595189567, 2.1047692272951455], "('AB4', 4)": [0.4999999999999999, 1.1199048642455407e-16, -1.391593982808325e-13, 0.5000000000000006], "('AB4', 5)": [112.40000000000002, 2.1738266923330637e-14, 6.378957754356253e-12, 112.39999999999999], "('AM4', 1)": [-20.99999999999899, 2.0807610921561709e-13, 2.1270567736058017e-10, -21.00000000000005], "('AM4', 2)": [2.1025000000000014, 3.3656995430479756e-16, 3.734339100859723e-14, 2.1025000000000014], "('AM4', 3)": [2.094715167655526, 0.001994422414793105, -2.0311231595190358, 2.1047692272951455], "('AM4', 4)": [0.4999999999999999, 1.1199048642455407e-16, -1.391593982808325e-13, 0.5000000000000006], "('AM4', 5)": [112.4, 1.6252387855053684e-14, 1.5054393136641803e-12, 112.40000000000002], "('ABM5', 1)": [-21.000000000000014, 2.7407884168448345e-15, 1.0203739941590393e-12, -21.000000000000032], "('ABM5', 2)": [2.1025000000000014, 3.275794329645676e-16, 3.734339100859723e-14, 2.1025000000000014], "('ABM5', 3)": [2.094715167655526, 0.001994422414793105, -2.0311231595190358, 2.1047692272951455], "('ABM5', 4)": [0.4999999999999999, 1.1199048642455407e-16, -1.391593982808325e-13, 0.5000000000000006], "('ABM5', 5)": [112.4, 1.6252387855053684e-14, 1.5054393136641803e-12, 112.40000000000002]}

d = {(i.strip('()').split(',')[0].strip("'"),int(i.strip('()').split(',')[1].strip(" "))):s[i] for i in s}

print(methods)
print([i for i in range(1,6)])
print('M D K C')

mn = {m:ix for ix,m in enumerate(methods)}


# 0:M
# 1:D
# 2:K
# 3:C
par = 0
arr = np.empty([5,len(methods)])

for tp in d:
    i = tp[1]-1 # inv
    j = mn[tp[0]] # method
    print(i,j)
    arr[i,j] = d[tp][par]

print(arr)
