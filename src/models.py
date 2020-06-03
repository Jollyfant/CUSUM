import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Model():

    """
    Class MedianModel
    Base class for CUSUM analysis models
    """

    # Read MEMS coordinates
    coordinates = np.loadtxt("./data/coordinates.txt")

    def __init__(self, filepath, decimate=0):

        """
        Def Model.__init__
        Initialises a model by reading the CSV data to a data frame
        """

        self.frame = pd.read_csv(filepath, sep="\t", header=0)

    @property
    def values(self):
        return self.frame.values

    def decimate(self, factor):
        self.frame = self.frame.iloc[1::factor, :]

    def normalise(self):
        self.frame = self.frame - self.frame.mean()

    def simple(self, model="median", window_length=100):

        """
        Def SimpleModel.estimate
        Prediction for a simple model
        """

        # Pandas rolling is fast
        windows = self.frame.rolling(window_length)

        # For the median model we return the median and variances
        if model == "median":
            return windows.median(), windows.var()
        elif model == "mean":
            return windows.mean(), windows.var()
        else:
            raise ValueError("Unknown model type requested.")


    def polynomial(self, model="linear", window_length=100):

        """
        Def PolynomialModel.estimate
        Predicts
        """

        # Pandas rolling is fast
        windows = self.frame.rolling(window_length)

        # Not very efficient fitting polynomials twice..
        if model == "linear":
            return windows.apply(self.linearFit), windows.apply(self.linearRes)
        elif model == "quadratic":
            return windows.apply(self.quadraticFit), windows.apply(self.quadraticRes)
        else:
            raise ValueError("Unknown model type requested.")


    # Polynomial for simple trend
    def linearRes(self, window):
        return np.polyfit(np.arange(window.size), window, 1, full=True)[1] / window.size

    def linearFit(self, window):
        return np.polyfit(np.arange(window.size), window, 1)[0]

    # Same but quadratic (2nd order)
    def quadraticRes(self, window):
        return np.polyfit(np.arange(window.size), window, 2, full=True)[1] / window.size

    def quadraticFit(self, window):
        return np.polyfit(np.arange(window.size), window, 2)[0]
