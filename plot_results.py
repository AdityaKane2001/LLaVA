import numpy as np
from matplotlib import pyplot as plt
import math

x = np.arange(25)


dataset = "pope" # "pope"

if dataset == "gqa":

    # GQA
    y = [
        61.94,
        59.28,
        59.41,
        59.06,
        59.45,
        59.21,
        59.42,
        59.52,
        59.57,
        59.61,
        59.31,
        59.58,
        59.43,
        59.47,
        59.51,
        59.45,
        58.97,
        58.75,
        58.76,
        58.25,
        58.10,
        57.63,
        57.71,
        55.83,
        53.11
    ]
    xticks = np.arange(0, 110, 10)
    
    plt.xticks(ticks=np.arange(0, 2.4 * 11, 2.4), labels = xticks)
    plt.yticks(np.linspace(52, 62, 11))
    plt.axhline(y=y[0], color='r', linestyle='-')
    plt.plot(x, y, "-*")
    plt.title("GQA accuracy (%)")
    plt.savefig("results-gqa.png")

elif dataset == "pope":
    y = [
        85.87,
        85.61,
        85.64,
        85.49,
        85.35,
        85.24,
        85.46,
        85.75,
        85.55,
        85.38,
        85.48,
        86.04,
        86.08,
        86.56,
        86.28,
        86.85,
        86.56,
        86.50,
        86.32,
        86.29,
        86.19,
        86.36,
        85.21,
        82.68,
        76.08
    ]
    xticks = np.arange(0, 110, 10)
    
    plt.xticks(ticks=np.arange(0, 2.4 * 11, 2.4), labels = xticks)
    plt.yticks(np.linspace(75, 87, 13))
    plt.axhline(y=y[0], color='r', linestyle='-')
    plt.axhline(y=86.85, color='r', linestyle='--')
    plt.plot(x, y, "-*")
    plt.title("PoPE accuracy (%)")
    plt.savefig("results-pope.png")

elif dataset == "time":
    y= [
        0.1470,
        0.1220,
        0.1177,
        0.1172,
        0.1154,
        0.1156,
        0.1121,
        0.1102,
        0.1106,
        0.1076,
        0.1067,
        0.1081,
        0.0977,
        0.0960,
        0.0961,
        0.0941,
        0.0939,
        0.0887,
        0.0862,
        0.0865,
        0.0832,
        0.0816,
        0.0811,
        0.0806,
        0.0808
     ]
    xticks = np.arange(0, 110, 10)
    
    plt.xticks(ticks=np.arange(0, 2.4 * 11, 2.4), labels = xticks)
    plt.yticks(np.linspace(0.15, 0.07, 9))
    plt.axhline(y=y[0], color='r', linestyle='-')
    plt.plot(x, y, "-*")
    plt.title("Average generation time (sec)")
    plt.savefig("results-time.png")