import numpy as np
import matplotlib.pyplot as plt

def plotCUSUMGraph(high, low, model="all"):

    """
    Def plotCUSUMGraph
    Decides what CUSUM parameter to plot based on multiple traces
    """

    fig, (ax1, ax2) = plt.subplots(2)

    if model == "all":
      ax1.plot(high, label="Upper")
      ax2.plot(low, label="Lower")

    elif model == "median":
      ax1.plot(np.median(high, axis=1), label="Upper")
      ax2.plot(np.median(low, axis=1), label="Lower")

    elif model == "mean":
      ax1.plot(np.mean(high, axis=1), label="Upper")
      ax2.plot(np.mean(low, axis=1), label="Lower")

    else:
      raise ValueError("Unknown model type requested.")

    fig.suptitle("CUSUM Change Detection")

    if model != "all":
      ax1.legend()
      ax2.legend()

    plt.show()
