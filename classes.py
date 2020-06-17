import numpy as np 
import pickle as pl

class molDyn:
  def __init__(self, dt, masses, v0):
    self.dt = dt
    self.time = 0
    self.v = v0
    self.pos = np.zeros((masses.shape[0], 3))

    self.masses = masses
    self.potentialEnergy = 0

    self.ccDistances = np.zeros((self.pos.shape[0] - 1, 3))
    self.cccAngles = np.zeros((self.ccDistances.shape[0] - 1 ))

    self.K_a = 573.8
    self.K_b = 222.
    self.K_gamma = 53.58
    self.K_delta = 76.28
    self.K_theta = 44.
    self.K_phi = 2.836

  def getVectorMagnitude(self, v):
    return np.sqrt(
        v[0] * v[0] +
        v[1] * v[1] +
        v[2] * v[2])

  def getVectorDotProduct(self, v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] + v2[2]

  def getCcDistances(self):
    for i in range(self.pos.shape[0] - 1):
      self.ccDistances[i] = self.pos[i + 1] - self.pos[i]

  def getCccAngles(self):
    for i in range(self.ccDistances.shape[0] - 1):
      self.cccAngles[i] = np.arccos(
        (self.getVectorDotProduct(self.ccDistances[i + 1], self.ccDistances[i])) / (
        self.getVectorMagnitude(self.ccDistances[i + 1] * self.getVectorMagnitude(self.ccDistances[i]))))

  def calcPotential(self):
    self.potentialEnergy = 0
    for i in range(self.ccDistances.shape[0]):
      self.potentialEnergy += 0.5 * self.K_b * np.square(self.getVectorMagnitude(self.ccDistances[i]))
    for i in range(self.cccAngles.shape[0]):
      self.potentialEnergy += 0.5 * self.K_theta * np.square(self.cccAngles[i])
  
  def calcForce(self):
    self.F = np.ones_like(self.masses)
    self.F = np.sin(self.time*np.array([2, 4]))

  def update(self):
    self.calcForce()
    self.pos += self.v * self.dt

    self.getCcDistances()
    self.getCccAngles()
    self.calcPotential()

    print(self.pos)
    print(self.ccDistances)
    print(self.cccAngles)
    print(self.potentialEnergy)
    print()

dt = 1
sim = molDyn(
  dt,
  np.array([1, 2, 3, 4]),
  np.matrix([
      [0.5, 0, 0],
      [2, 0, 0],
      [1, -0.5, 0],
      [0, 1, 0.5]
    ]))

for i in range(10):
  sim.update()