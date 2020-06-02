import numpy as np
import matplotlib.pyplot as plt

def plotCUSUMGraph(high, low, model="all"):

    """
    Def plotCUSUMGraph
    Decides what CUSUM parameter to plot based on multiple traces
    """

    if model == "all":
      plt.plot(high, label="Upper")
      plt.plot(low, label="Lower")

    elif model == "median":
      plt.plot(np.median(high, axis=1), label="Upper")
      plt.plot(np.median(low, axis=1), label="Lower")

    elif model == "mean":
      plt.plot(np.mean(high, axis=1), label="Upper")
      plt.plot(np.mean(low, axis=1), label="Lower")

    else:
      raise ValueError("Unknown model type requested.")

    plt.show()
