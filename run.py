"""

    NEWTON-g Data Mining Tool

    Authors:
        - Mathijs Koymans 2020
        - Flavio Cannavo 2020

    Supports:
        - Plotting features for gravimeter timeseries (rolling)
            - Quantiles (68, 95, 99.7)
            - Mean
            - Median
            - Stacking
            - All
        - CUSUM Analysis for change detection
        - EOF Analysis for spatio-temporal correlations

"""

from src.geometry import Grid
from src.model import Model
from src.forward import singleAscendingSource
from src.plot import plotCUSUMGraph, plotEOFEigenvalues, plotEOFEigenvectors, plotModel
from src.cusum import CUSUM
from src.eof import EOF

if __name__ == "__main__":

  """
  Def __main__
  NEWTON-g algorithms for data mining multiple gravimeter traces 
  """

  #-----------------------#
  # FORWARD MODELING DEMO #
  #-----------------------#

  # MEMS stations locations on flat surface: todo fix with elevation
  x, y = Model.coordinates.T

  # Simple grid spacing and npoints (x, y): no z
  grid = Grid(x, y, 0)

  # Number of timesteps (length) and source model
  model = Model(grid.solveSteps(1000, singleAscendingSource))

  # Show
  plotModel(model, mode="mean", window_length=100)

  #--------------#
  # INV ANALYSIS #
  #--------------#

  # Data path: csv gravimeter data
  filepath = "./data/gravity.csv"

  #------------------------#
  # DEMO PLOTTING ANALYSIS #
  #------------------------#

  # Create a simple data model using a Pandas data frame
  model = Model(filepath)
  model.update(Model(filepath))

  # Decimation the model
  deci = model.decimate(10)

  # Show the data in the model
  plotModel(model, mode="mean", window_length=100)

  #-----------------------------------#
  # DEMO CUSUM ANALYSIS: SIMPLE MODEL #
  #-----------------------------------#

  # We can reuse the SimpleModel
  # Estimate mean for the simple model used in the CUSUM algo
  means, variances = model.simple(model="mean", window_length=500)

  # Get the upper and lower limits from the data frames
  high, low = CUSUM(model, means, variances, k=1)

  # Show the graph taking an e.g. (mean, all, median) model of all high / low thresholds
  plotCUSUMGraph(high, low, mode="mean")

  #---------------------------------------#
  # DEMO CUSUM ANALYSIS: POLYNOMIAL MODEL #
  #---------------------------------------#

  # We can reuse the SimpleModel
  # Estimate mean for the simple model used in the CUSUM algo
  # means, variances = model.polynomial(model="linear", window_length=500)

  # Get the upper and lower limits from the data frames
  # high, low = CUSUM(model.frame, means, variances, k=1)

  # Show the graph taking an e.g. (mean, all, median) model of all high / low thresholds
  # plotCUSUMGraph(high, low, model="mean")

  #-------------------#
  # DEMO EOF ANALYSIS #
  #-------------------#

  # Normalising the simple data model for EOF analysis
  model.normalise()

  # Get the rolling means and variances (simple model)
  w, v = EOF(model, normalise=True)

  # Show weight of eigenvalues
  plotEOFEigenvalues(w)

  # Plot EOF PCA with cutoff eigenvalues < 5%
  plotEOFEigenvectors(w, v, model, weight_cutoff=0.05)
