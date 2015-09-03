#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import subprocess
from scipy.interpolate import interp1d
import numpy as np

INPUT = "supermarket.arff"
RUN = "./run.sh"
FILTER = "./filter.sh"
filtered = "supermarket-filtered.arff"
filter_indexes = map(lambda x: str(x) ,
                        [
                            13, # bread and cake
                            86  # vegetables
                        ])
if filter_indexes:
    cmd = [FILTER, INPUT, filtered,  ",".join(filter_indexes)]
    subprocess.call(cmd)
else:
    filtered = INPUT

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
            cmd = [RUN, filtered, filename, conf, minsup, alg_type[alg]];
            subprocess.call(cmd)
exit()


x = range(1,11)
y  = map(lambda i: i**2, x)
plotter(x, y)

def plotter(x, y, filename = 'foo.png', interp_points = 30, title = 'Alg wtih Min Sup = minsup', x_title = 'Confidence', y_title = '# Rules'):
    x_interp   = np.linspace(min(x), max(x), interp_points)
    y_interp = np.interp(x_interp, x, y)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(x_title, fontweight = 'bold')
    ax1.set_ylabel(y_title, fontweight = 'bold')
    fig.suptitle(title, fontsize = 14, fontweight = 'bold')

    ax1.plot(x, y, 'bo', x_interp, y_interp, 'b-')

    plt.savefig(filename, bbox_inches='tight')
    # plt.show()



