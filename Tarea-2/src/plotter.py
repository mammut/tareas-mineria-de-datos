#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import matplotlib.pyplot as plt
import subprocess
from scipy.interpolate import interp1d
import numpy as np

RUN = "./run.sh"
INPUT = "supermarket.arff"
c = map(lambda x: str(x/100.0), range(85, 96))
alg_minsup = {
    "Apriori": ['0.1', '0.15'],
    "FP-growth": map(lambda x: str(x/100.0), range(16, 26))
}

alg_type = {"Apriori":"1", "FP-growth":"2"}

for alg, minsups in alg_minsup.items():
    for minsup in minsups:
        for conf in c:
            filename = '-'.join([alg,minsup,conf])+"-out.txt"
            cmd = [RUN, INPUT, filename, conf, minsup, alg_type[alg]];
            subprocess.call(cmd)
exit()

x = range(1,11)
x_interp   = np.linspace(1, len(x), 30)
y  = map(lambda i: i**2, x)
y_interp = np.interp(x_interp, x, y)

fig, ax1 = plt.subplots()
ax1.plot(x, y, 'bo', x_interp, y_interp, 'b-')

ax1.set_xlabel('t_min')
ax1.set_ylabel('Creditos/costo', color="r")

for tl in ax1.get_yticklabels():
    tl.set_color('r')

plt.show()
