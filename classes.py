from collections import namedtuple
from enum import Enum
import jax as jax
import jax.numpy as np
import numpy as onp
import pickle as pl
import time as time

# masses in amu
class atoms(Enum):
  C = 12.0107
  H = 1.00784

ethane = {
    "C0" : {
      "Type" : atoms.C,
      "Neighbors" : ["C1", "H0", "H1", "H2"],
      "Position" : np.array([0.00, 0.00, 0.00])
    },
    "C1" : {
      "Type" : atoms.C,
      "Neighbors" : ["C0", "H3", "H4", "H5"],
      "Position" : np.array([1.90, 0.00, 0.00])
    },
    "H0" : {
      "Type" : atoms.H,
      "Neighbors" : ["C0"],
      "Position" : np.array([-0.62, 0.00, -1.80])
    },
    "H1" : {
      "Type" : atoms.H,
      "Neighbors" : ["C0"],
      "Position" : np.array([-0.62, 1.56, 0.90])
    },
    "H2" : {
      "Type" : atoms.H,
      "Neighbors" : ["C0"],
      "Position" : np.array([-0.62, -1.56, 0.90])
    },
    "H3" : {
      "Type" : atoms.H,
      "Neighbors" : ["C0"],
      "Position" : np.array([2.52, 0.00, 1.80])
    },
    "H4" : {
      "Type" : atoms.H,
      "Neighbors" : ["C0"],
      "Position" : np.array([2.52, 1.56, -0.90])
    },
    "H5" : {
      "Type" : atoms.H,
      "Neighbors" : ["C0"],
      "Position" : np.array([2.52, -1.56, -0.90])
    }
  }

class mol:
  def __init__ (self, atoms, dt):
    self.atoms = atoms
    self.dt = dt

    self.initAtomArrays()
    self.initPairs()
    self.initTriples()
    self.initQuads()
    self.initMatrices()

    # kcal/Å^2
    # kcal/rad^2

    self.K_cc = 573.8
    self.K_ch = 222.
    self.K_ccc = 53.58
    self.K_hch = 76.28
    self.K_cch = 44.
    self.K_ccTorsional = 2.836

    # Avogadro's number

    self.N = 6.0221409e+23

    # ångström to meter
    # kcal to joules
    # amu to kg

    self.A2m = 1e-10
    self.kcal2J = 4184
    self.amu2kg = 1.660539e-27
    
    # distance-energy conversion constants
    # angle-energy conversion constants

    self.distEnergyK_cc = self.K_cc * self.kcal2J / self.N
    self.distEnergyK_ch = self.K_ch * self.kcal2J / self.N
    self.angleEnergyK_ccc = self.K_ccc * self.kcal2J / self.N
    self.angleEnergyK_hch = self.K_hch * self.kcal2J / self.N
    self.angleEnergyK_cch = self.K_cch * self.kcal2J / self.N
    self.angleEnergyK_ccTorsional = self.K_ccTorsional * self.kcal2J / self.N

    self.initJax()

  # reads bond relations and converts to readable format

  def initAtomArrays(self):
    self.atomArray = []
    self.atomMap = {}
    for i, (k, v) in enumerate(self.atoms.items()):
      self.atomArray.append((k, v))
      self.atomMap[k] = i

  # creates array of CC, CH pairs

  def initPairs(self):
    self.pairs = []
    for i, (_, v) in enumerate(self.atoms.items()):
      for atom in v["Neighbors"]:
        j = self.atomMap[atom]
        if j < i:
          continue

        self.pairs.append(
          (i, j, v["Type"], self.atoms[atom]["Type"])
        )

    self.ccPairs = np.array([np.array([t[0], t[1]]) for t in self.pairs if t[2] == atoms.C and t[3] == atoms.C])
    self.chPairs = np.array([np.array([t[0], t[1]]) for t in self.pairs if (t[2] == atoms.H and t[3] == atoms.C) or (t[2] == atoms.C and t[3] == atoms.H)])

  # creates array of CCC, HCH, CCH triples

  def initTriples(self):
    self.triples = []
    for i, (k, v) in enumerate(self.atoms.items()):
      if v["Type"] != atoms.C:
        continue

      neighbors = v["Neighbors"]
      for j in range(len(neighbors)):
        for k in range(j + 1, len(neighbors)):
          self.triples.append(
            np.array([self.atomMap[neighbors[j]], i, self.atomMap[neighbors[k]]])
          )

    self.cccTriples = np.array([t for t in self.triples if self.atomArray[t[0]][1]["Type"] == atoms.C and self.atomArray[t[2]][1]["Type"] == atoms.C])
    self.hchTriples = np.array([t for t in self.triples if self.atomArray[t[0]][1]["Type"] == atoms.H and self.atomArray[t[2]][1]["Type"] == atoms.H])
    self.cchTriples = np.array([t for t in self.triples if (self.atomArray[t[0]][1]["Type"] == atoms.H and self.atomArray[t[2]][1]["Type"] == atoms.C) \
      or (self.atomArray[t[0]][1]["Type"] == atoms.C and self.atomArray[t[2]][1]["Type"] == atoms.H)])

  # creates array of _CC_ quads

  def initQuads(self):
    self.quads = []
    for pair in self.ccPairs:
      for left in self.atomArray[pair[0]][1]["Neighbors"]:
        if self.atomMap[left] == pair[1]:
          continue

        for right in self.atomArray[pair[1]][1]["Neighbors"]:
          if self.atomMap[right] == pair[0]:
            continue

          self.quads.append(
            (self.atomMap[left], pair[0], pair[1], self.atomMap[right])
          )
    
    self.quads = np.array(self.quads)

  # creates variable matrices

  def initMatrices(self):
    # angstroms
    self.posMatrix = np.array([(atom[1]["Position"]) for atom in self.atomArray])
    self.posMatrix = np.array([[ 0., 0., 0. ],
      [ 1.6982005, 0.17761631, 0.02418225],
      [-0.54968464, -0.0835496, -1.7867701 ],
      [-0.13722242, 1.148577, 0.8286429 ],
      [-0.7728126, -1.4011788, 0.8939908 ],
      [ 2.011011, 0.49305442, 1.8159293 ],
      [ 2.218594, 1.8368096, -0.8754015 ],
      [ 1.6481868, -0.7784976, -0.80968946]])
    # angstroms/second
    self.velMatrix = np.zeros((len(self.atomArray), 3))
    # angstroms/second^2
    self.accelMatrix = np.zeros((len(self.atomArray), 3))
    self.prevAccelMatrix = np.zeros((len(self.atomArray), 3))
    # newtons
    self.forceMatrix = np.zeros((len(self.atomArray), 3))
    # atomc masses
    self.massMatrix = np.array([atom[1]["Type"].value for atom in self.atomArray])
    # joules
    self.potential = 0
    # joules
    self.kinetic = 0
    # seconds
    self.t = 0

  def vmap(self, f, in_axes):
    if vmap_funcs:
      return jax.vmap(f, in_axes)

    return f

  def jit(self, f):
    if jit_funcs:
      return jax.jit(f)
    else:
      return f

  def initJax(self):
    self.calcPotential_j = self.jit(self.getCalcPotential(
      (len(self.ccPairs) != 0),
      (len(self.chPairs) != 0),
      (len(self.cccTriples) != 0),
      (len(self.hchTriples) != 0),
      (len(self.cchTriples) != 0),
      (len(self.quads) != 0)))
    # self.calcPotential_j = self.jit(self.calcPotential)
    self.gradient_j = self.jit(jax.grad(self.getCalcPotential))
    self.kineticAtom_v = self.jit(self.vmap(self.kineticAtom_, in_axes = (0, 0)))
    self.distance_v = self.jit(self.vmap(self.distance_, in_axes = (0, )))
    self.angle_v = self.jit(self.vmap(self.angle_, in_axes = (0, )))
    self.cosTorsionalAngle_v = self.jit(self.vmap(self.cosTorsionalAngle_, in_axes = (0, )))
    self.accelAtom_v = self.jit(self.vmap(self.accelAtom_, in_axes = (0, 0)))
    self.posAtom_v = self.jit(self.vmap(self.posAtom_, in_axes = (0, 0)))
    self.velAtom_v = self.jit(self.vmap(self.velAtom_, in_axes = (0, 0)))
    self.updatePosition_v = self.jit(self.vmap(self.updatePosition, in_axes = (0, 0, 0, 0, 0)))

  # calculates length AB given positions

  def distance_(self, P):
    r = P[:,0] - P[:,1]
    r_mag = np.sqrt(np.sum(np.square(r)))
    return r_mag

  # calculates cosine of angle ABC given positions

  def cosAngle_(self, P):
    r1 = P[:,0] - P[:,1]
    r2 = P[:,2] - P[:,1]
    dot = np.sum(np.multiply(r1, r2))
    r1_mag = np.sqrt(np.sum(np.square(r1)))
    r2_mag = np.sqrt(np.sum(np.square(r2)))
    return dot / (r1_mag * r2_mag)

  # calculates angle ABC given positions

  def angle_(self, P):
    return np.arccos(self.cosAngle_(P))

  # calculates torsional angle ABCD given positions

  def cosTorsionalAngle_(self, P):
    r1 = P[0,:] - P[1,:]
    r2 = P[1,:] - P[2,:]
    r3 = P[3,:] - P[2,:]
    cp_12 = np.cross(r1, r2)
    cp_32 = np.cross(r3, r2)
    cp = np.array([cp_12, np.zeros(cp_12.shape), cp_32])
    cosTorsionalAngle = self.cosAngle_(cp)
    return cosTorsionalAngle

  # calculates potential energy of molecule given positions

  def getCalcPotential(self, has_ccPairs, has_chPairs, has_cccTriples, has_hchTriples, has_cchTriples, has_quads):
    this = self

    def calcPotential_(pos):
      potential = 0

      if has_ccPairs:
        potential += 0.5 * np.sum(np.square(this.distance_v(pos[this.ccPairs]))) * this.distEnergyK_cc

      if has_chPairs:
        potential += 0.5 * np.sum(np.square(this.distance_v(pos[this.chPairs]))) * this.distEnergyK_ch

      if has_cccTriples:
        potential += 0.5 * np.sum(np.square(this.angle_v(pos[this.cccTriples]))) * this.angleEnergyK_ccc

      if has_hchTriples:
        potential += 0.5 * np.sum(np.square(this.angle_v(pos[this.hchTriples]))) * this.angleEnergyK_hch

      if has_cchTriples:
        potential += 0.5 * np.sum(np.square(this.angle_v(pos[this.cchTriples]))) * this.angleEnergyK_cch

      if has_quads:
        cosAngle = this.cosTorsionalAngle_v(pos[this.quads])
        potential += 0.5 \
          * np.sum(1 + 4 * np.power(cosAngle, 3) - 3 * cosAngle ) \
          * this.angleEnergyK_ccTorsional

      return potential

    return calcPotential_

  # def calcPotential2(self, P, ccPairs, chPairs, cccTriples, hchTriples, cchTriples, quads):
  #   return self.calcPotential_2(P, ccPairs, chPairs, cccTriples, hchTriples, cchTriples, quads, np)

  # def calcPotential_(self, P):
  #   potential = 0

  #   if len(self.ccPairs) != 0:
  #     potential += 0.5 * np.sum(np.square(self.distance_v(P[self.ccPairs]))) * self.K_cc * self.kcal2J / self.N

  #   if len(self.chPairs) != 0:
  #     potential += 0.5 * np.sum(np.square(self.distance_v(P[self.chPairs]))) * self.K_ch * self.kcal2J / self.N

  #   if len(self.cccTriples) != 0:
  #     potential += 0.5 * np.sum(np.square(self.angle_v(P[self.cccTriples]))) * self.K_ccc * self.kcal2J / self.N

  #   if len(self.hchTriples) != 0:
  #     potential += 0.5 * np.sum(np.square(self.angle_v(P[self.hchTriples]))) * self.K_hch * self.kcal2J / self.N

  #   if len(self.cchTriples) != 0:
  #     potential += 0.5 * np.sum(np.square(self.angle_v(P[self.cchTriples]))) * self.K_cch * self.kcal2J / self.N

  #   if len(self.quads) != 0:
  #     cosAngle = self.cosTorsionalAngle_v(P[self.quads])
  #     potential += 0.5 \
  #       * np.sum(1 + 4 * np.power(cosAngle, 3) - 3 * cosAngle ) \
  #       * self.K_ccTorsional * self.kcal2J \
  #       / self.N

  #   return potential

  # calculates kinetic energy of single atom given mass and velocity

  def kineticAtom_(self, M, V):
    v_mag = np.sqrt(np.sum(np.square(V)))
    return 0.5 * M * v_mag ** 2 * self.amu2kg * self.A2m ** 2

  # calculates kinetic energy of molecule given masses and velocities

  def calcKinetic(self, M, V):
    return np.sum(self.kineticAtom_v(M, V))

  # calculates force matrix

  def calcForce(self, P):
    self.forceMatrix = -1 * self.gradient_j(P) / self.A2m

    return self.forceMatrix

  # calculates acceleration matrix of single atom given mass and force

  def accelAtom_(self, M, F):
    return F / M / self.amu2kg / self.A2m

  # calculates acceleration matrix gives masses and forces

  def calcAccel(self, M, F):
    self.accelMatrix = self.accelAtom_v(M, F)

    return self.accelMatrix

  # calculates position matrix of single atom given position and velocity

  def posAtom_(self, P, V, A):
    return P + V * self.dt + A * self.dt ** 2 / 2

  # calculates position matrix given positions and velocities

  def calcPos(self, P, V):
    self.posMatrix = self.posAtom_v(P, V)

    return self.posMatrix

  # calculates velocity matrix of single atom given velocity and acceleration

  def velAtom_(self, V, A):
    return V + A * self.dt

  # calculates velocity matrix given velocities and accelerations

  def calcVel(self, V, A):
    self.velMatrix = self.velAtom_v(V, A)

    return self.velMatrix

  def updatePosition(self, P, V, A, pA, dt):
    # dA = (A - pA) / dt
    P = P + V * dt + A * (dt * dt / 2) # + dA * (dt * dt * dt / 3)
    V = V + A * dt # + dA * (dt * dt / 2)
    return (P, V)

  def updatePos(self):
    (self.posMatrix, self.velMatrix) = self.updatePosition(
      self.posMatrix,
      self.velMatrix,
      self.accelMatrix,
      self.prevAccelMatrix,
      self.dt)

    self.prevAccelMatrix = self.accelMatrix

  def update(self):
    self.calcForce(self.posMatrix)
    self.calcAccel(self.massMatrix, self.forceMatrix)
    self.updatePos()
    # self.calcVel(self.velMatrix, self.accelMatrix)
    # self.calcPos(self.posMatrix, self.velMatrix)

    self.t += self.dt
  def print(self):
    self.potential = self.calcPotential_j(self.posMatrix)
    kineticE = self.calcKinetic(self.massMatrix, self.velMatrix)
    print("t:")
    print(self.t)
    print("potential:")
    print(self.potential)
    print("kinetic:")
    print(kineticE)
    print("total:")
    print(self.potential + kineticE)
    print()
    print("posMatrix:")
    print(self.posMatrix)
    print("velMatrix:")
    print(self.velMatrix)
    print("accelMatrix")
    print(self.accelMatrix)
    print()

print("program start")
jit_funcs = False
vmap_funcs = True
dt = 1e-17
sim = mol(ethane, dt)
sim.print()
sim.update()
start_time = time.perf_counter()
for i in range(1000):
  sim.update()
  if i % 100 == 0:
    sim.print()

print("--- %s seconds ---" % (time.perf_counter() - start_time))