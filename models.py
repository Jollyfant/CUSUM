import pandas as pd
import numpy as np

class Model():

  """
  Class MedianModel
  Base class for CUSUM analysis models
  """

  # Reference the pandas data frame
  def __init__(self, filepath, decimate=0):
      """
      Def Model.__init__
      Initialises a model by reading the CSV data to a data frame
      """

      self.frame = pd.read_csv(filepath, sep="\t", header=0)

      # No decimation: filter instead?
      self.frame = self.frame.iloc[1::10, :]

  @property
  def values(self):
      return self.frame.values


class PolynomialModel(Model):

  """
  Class LinearTrendModel
  Extends Model with MedianModel for CUSUM analysis
  """

  # Inherit from model and set internal data
  def __init__(self, filepath):

      """
      Def PolynomialModel.__init__
      Initialises a simple single parameter model (e.g. mean, median)
      """

      Model.__init__(self, filepath)
    
  def estimate(self, model="linear", window_length=100):

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


class SimpleModel(Model):

  """
  Class MedianModel
  Extends Model with MedianModel for CUSUM analysis
  """

  # Inherit from model and set internal data
  def __init__(self, filepath):

      """
      Def SimpleModel.__init__
      Initialises a simple single parameter model (e.g. mean, median)
      """

      Model.__init__(self, filepath)
    
  def estimate(self, model="median", window_length=100):

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
