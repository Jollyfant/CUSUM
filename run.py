from models import SimpleModel, PolynomialModel
from plot import plotCUSUMGraph
from cusum import CUSUM

if __name__ == "__main__":

  """
  Def __main__
  CUSUM algorithm working on multiple traces from gravimeters
  """

  filepath = "./data/gravity.csv"

  # Create a simple parameter model using a Pandas data frame
  #model = PolynomialModel(filepath)
  model = SimpleModel(filepath)

  # Get the rolling means and variances (simple model)
  #means, variances = model.estimate(model="quadratic", window_length=100)
  means, variances = model.estimate(model="mean", window_length=500)
  
  # Get the upper and lower limits from the data frames
  high, low = CUSUM(model.frame, means, variances, k=1)

  # Show the graph taking an e.g. (mean, all, median) model of all high / low thresholds
  plotCUSUMGraph(high, low, model="mean")
