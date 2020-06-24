import numpy as np 
import jax.numpy as jnp
import pickle as pl

class molDyn:
  def __init__(self, dt, atoms, pos0, v0, bonds):
    self.dt = dt
    self.time = 0
    self.v = v0
    self.pos = pos0
    self.F = np.zeros((atoms.shape[0], 3))

    self.atoms = atoms
    self.bonds = bonds
    self.potentialEnergy = 0
    self.kineticEnergy = 0

    self.ccPairs = []
    self.chPairs = []
    self.cccTriples = []
    self.hchTriples = []
    self.cchTriples = []
    
    self.calcPairs()
    self.calcTriples()

    self.ccDistances = np.zeros((len(self.ccPairs), 3))
    self.chDistances = np.zeros((len(self.chPairs), 3))
    self.cccBondAngles = np.zeros((len(self.cccTriples), 1))
    self.hchBondAngles = np.zeros((len(self.hchTriples), 1))
    self.cchBondAngles = np.zeros((len(self.cchTriples), 1))

    self.calcNumCcTorsionalAngles()

    self.ccTorsionalAngles = np.zeros((self.numCcTorsionalAngles, 1))

    self.K_b = 573.8
    self.K_a = 222.
    self.K_theta = 53.58
    self.K_delta = 76.28
    self.K_gamma = 44.
    self.K_phi = 2.836

    self.massC = 12.0107
    self.massH = 1.00784

  def calcPairs(self):
    for i in range(len(self.bonds)):
      for j in range(len(self.bonds[i])):
        if i < self.bonds[i][j]:
          if self.atoms[i] and self.atoms[self.bonds[i][j]]:
            self.ccPairs.append([i, self.bonds[i][j]])
          if self.atoms[i] ^ self.atoms[self.bonds[i][j]]:
            self.chPairs.append([i, self.bonds[i][j]])

  def calcTriples(self):
    for i in range(len(self.bonds)):
      if self.atoms[i]:
        for j in range(len(self.bonds[i])):
          for k in range(j + 1, len(self.bonds[i])):
            if self.atoms[self.bonds[i][j]] and self.atoms[self.bonds[i][k]]:
              self.cccTriples.append([i, self.bonds[i][j], self.bonds[i][k]])
            elif not(self.atoms[self.bonds[i][j]] or self.atoms[self.bonds[i][k]]):
              self.hchTriples.append([i, self.bonds[i][j], self.bonds[i][k]])
            elif self.atoms[self.bonds[i][j]] ^ self.atoms[self.bonds[i][k]]:
              self.cchTriples.append([i, self.bonds[i][j], self.bonds[i][k]])

  def calcNumCcTorsionalAngles(self):
    self.numCcTorsionalAngles = 0
    for i in range(len(self.ccPairs)):
      self.numCcTorsionalAngles += (len(self.bonds[self.ccPairs[i][0]]) - 1) * (len(self.bonds[self.ccPairs[i][1]]) - 1)
    return self.numCcTorsionalAngles

#  def getVectorMagnitude(self, v):
#    return np.sqrt(
#        v[0] * v[0] +
#        v[1] * v[1] +
#        v[2] * v[2])

#  def getVectorDotProduct(self, v1, v2):
#    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

#  def getVectorCrossProduct(self, v1, v2):
#    return np.matrix([
#        [v1[1] * v2[2] - v1[2] * v2[1]],
#        [v1[2] * v2[0] - v1[0] * v2[2]],
#        [v1[0] * v2[1] - v1[1] * v2[0]]])

  def getVectorAngle(self, v1, v2):
    return np.arccos(np.dot(v1, v2.transpose()) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

  def getVectorTorsionalAngle(self, v1, v2, v3):
    return self.getVectorAngle(np.cross(v1, v2), np.cross(v1, v3))

  def calcCcDistances(self):
    for i in range(len(self.ccPairs)):
      self.ccDistances[i] = self.pos[self.ccPairs[i][0]] - self.pos[self.ccPairs[i][1]]

  def calcChDistances(self):
    for i in range(len(self.chPairs)):
      self.chDistances[i] = self.pos[self.chPairs[i][0]] - self.pos[self.chPairs[i][1]]

  def calcCccAngles(self):
    for i in range(len(self.cccTriples)):
      self.cccBondAngles[i] = self.getVectorAngle(
        self.pos[self.cccTriples[i][1]] - self.pos[self.cccTriples[i][0]],
        self.pos[self.cccTriples[i][2]] - self.pos[self.cccTriples[i][0]])

  def calcHchAngles(self):
    for i in range(len(self.hchTriples)):
      self.hchBondAngles[i] = self.getVectorAngle(
        self.pos[self.hchTriples[i][1]] - self.pos[self.hchTriples[i][0]],
        self.pos[self.hchTriples[i][2]] - self.pos[self.hchTriples[i][0]])

  def calcCchAngles(self):
    for i in range(len(self.cchTriples)):
      self.cchBondAngles[i] = self.getVectorAngle(
        self.pos[self.cchTriples[i][1]] - self.pos[self.cchTriples[i][0]],
        self.pos[self.cchTriples[i][2]] - self.pos[self.cchTriples[i][0]])

  def calcCcTorsionalAngles(self):
    x = 0
    for i in range(len(self.ccPairs)):
      for j in range(len(self.bonds[self.ccPairs[i][0]])):
        if not self.bonds[self.ccPairs[i][0]][j] == self.ccPairs[i][1]:
          for k in range(len(self.bonds[self.ccPairs[i][1]])):
            if not self.bonds[self.ccPairs[i][1]][k] == self.ccPairs[i][0]:
              self.ccTorsionalAngles[x] = self.getVectorTorsionalAngle(
                self.pos[self.ccPairs[i][0]] - self.pos[self.ccPairs[i][1]],
                self.pos[self.bonds[self.ccPairs[i][0]][j]] - self.pos[self.ccPairs[i][0]],
                self.pos[self.bonds[self.ccPairs[i][1]][k]] - self.pos[self.ccPairs[i][1]])
              x += 1

  def calcPotential(self):
    self.calcCcDistances()
    self.calcChDistances()
    self.calcCccAngles()
    self.calcHchAngles()
    self.calcCchAngles()
    self.calcCcTorsionalAngles()

    self.potentialEnergy = 0

    for i in range(len(self.ccDistances)):
      self.potentialEnergy += 0.5 * self.K_b * np.square(np.linalg.norm(self.ccDistances[i]))

    for i in range(len(self.chDistances)):
      self.potentialEnergy += 0.5 * self.K_a * np.square(np.linalg.norm(self.chDistances[i]))

    for i in range(len(self.cccBondAngles)):
      self.potentialEnergy += 0.5 * self.K_theta * np.square(self.cccBondAngles[i][0])

    for i in range(len(self.hchBondAngles)):
      self.potentialEnergy += 0.5 * self.K_delta * np.square(self.hchBondAngles[i][0])

    for i in range(len(self.cchBondAngles)):
      self.potentialEnergy += 0.5 * self.K_gamma * np.square(self.cchBondAngles[i][0])

    for i in range(len(self.ccTorsionalAngles)):
      self.potentialEnergy += 0.5 * self.K_phi * (1 + np.cos(3 * self.ccTorsionalAngles[i][0]))
    
    return self.potentialEnergy

  def calcKinetic(self):
    self.kineticEnergy = 0

    for i in range(self.v.shape[0]):
      self.kineticEnergy += 0.5 * np.linalg.norm(self.v[i])
  
  def calcForceManual(self):
    originalPE = self.potentialEnergy

    if not originalPE == 0:
      for i in range(len(self.F)):
        h = 1e-9
        for j in range(3):
          self.pos[i, j] += h
          self.F[i, j] = (originalPE - self.calcPotential()) / h
          self.pos[i, j] -= h

  def update(self):
    self.calcForceManual()

    self.pos += self.v * self.dt
    self.v += self.F * self.dt

    self.calcPotential()
    self.calcKinetic()

    print(self.pos)
    print(self.v)
    print(self.F)
    print(-self.potentialEnergy)
    print(self.kineticEnergy)
    print(self.kineticEnergy - self.potentialEnergy)
    print()

dt = 1e-3
sim = molDyn(
  dt,
  # True = C, False = H
  np.array([True, True, False, False, False, False, False, False]),
  np.matrix([
      [0., 0., 0.],
      [2., 0., 0.],
      [0., -1., 0.],
      [-1., 1., 2.],
      [-1., 1., -1.],
      [2., 1., 0.],
      [3., 1., 1.],
      [3., 0., -1.]
    ]),
  np.matrix([
      [0., 0., 0.],
      [2., 0., 0.],
      [0., -1., 0.],
      [-1., 1., 2.],
      [-1., 1., -1.],
      [2., 1., 0.],
      [3., 1., 1.],
      [3., 0., -1.]
    ]),
  [
      [1, 2, 3, 4],
      [0, 5, 6, 7],
      [0],
      [0],
      [0],
      [1],
      [1],
      [1]
    ])

for i in range(1000):
  sim.update()