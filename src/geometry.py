import numpy as np

class Grid():

  """
  class Grid
  Creates a (nx, ny) grid on which gravity from a source will be evaluated
  """

  def __init__(self, x, y, z):

    # Create the grid spacings
    self.x = x
    self.y = y
    self.z = z

  def solveSteps(self, length, model):

    """
    def Grid.solveSteps
    Solves the forward problem over a number of time steps
    """

    slices = list()

    # Apply the model from over a unit time interval
    for step in np.linspace(0, 1, length):
      slices.append(model(self, step))

    # Stack the data in the right shape for timeseries
    # Return model consistent with the reach of the code
    return np.vstack(slices)

  def solve(self, source):

    """
    def Grid.solve
    Fast vectorized solver of gravity on a grid
    """

    # Physical parameters
    G = 6.67E-11
    anoise = 10

    # Mogi point source volume change
    mogi = ((source.dm / source.rho) * (1 - source.v) / np.pi)

    # Subtract the source position from the virtual gravimeter positions
    dx = self.x - source.x
    dy = self.y - source.y
    dz = self.z - source.z

    # Vectorized solve for distance to source
    r2 = (dx ** 2 + dy ** 2 + dz ** 2) ** (3/2)

    # Vertical displacement because of volume change
    mz = mogi * (dz / r2)

    # Bouguer slab approximation
    bg = 111.9 * mz

    # And the free air gradient (in microgal)
    fag = 308.6 * mz

    # Add some noise
    noise = anoise * np.random.rand(self.x.size)

    # Without deformation
    return noise + (1E8 * G * (((source.dm) * dz / r2))) 

    # With mogi deformation
    # return noise - fag + (1E8 * G * (((source.dm) * dz / r2) + bg)) 
