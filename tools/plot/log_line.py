import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

import itertools
from itertools import product
from scipy.interpolate import make_interp_spline

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


file_path_dict = {
    "rte": "/workspace/output/fedglue/rte/fedchild_n/20220728032430_bert.eval.log",
    "qnli": "/workspace/output/fedglue/qnli/fedchild_n/20220727161306_bert.eval.log",
    "cola": "/workspace/output/fedglue/cola/fedchild_n/20220727161054_bert.eval.log",
    "mrpc": "/workspace/output/fedglue/mrpc/fedchild_n/20220728032211_bert.eval.log",
    # "qqp":

}

task = sys.argv[1]

file_path = file_path_dict[task]

with open(file_path, "rb") as file:
    data = pickle.load(file)

loss_log, acc_log = [], []
for round_metric in data["logs"]:
    for rd, metric in round_metric.items():
        loss_log.append(round(float(metric["loss"]), 3))
        acc_log.append(round(float(metric["acc"][0:4]), 3))

fig = plt.figure(figsize=(10, 6))
my_x_ticks = np.arange(1, len(acc_log) + 1, 1)

plt.plot(my_x_ticks, loss_log, label="loss", markersize=8)
plt.plot(my_x_ticks, acc_log, label="acc", markersize=8)
plt.title(f"Task  {task.upper()}")
plt.legend()
plt.show()
