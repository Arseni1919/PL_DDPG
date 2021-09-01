import numpy as np

from alg_constrants_amd_packages import *
for i in range(10):
    b = np.array([0, 0])
    a = np.random.normal(0, ACT_NOISE, 2)
    c = b + a
    print(c)
    d = np.clip(a, -0.1, 0.1)
    print(d)
    print('---')