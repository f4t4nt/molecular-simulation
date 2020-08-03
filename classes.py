from collections import namedtuple
from enum import Enum
import jax as jax
import jax.numpy as np
import numpy as onp
import pickle as pl

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

    self.ccPairs = np.array([(t[0], t[1]) for t in self.pairs if t[2] == atoms.C and t[3] == atoms.C])
    self.chPairs = np.array([(t[0], t[1]) for t in self.pairs if (t[2] == atoms.H and t[3] == atoms.C) or (t[2] == atoms.C and t[3] == atoms.H)])

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
            (self.atomMap[neighbors[j]], i, self.atomMap[neighbors[k]])
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
    # angstroms/second
    self.velMatrix = np.zeros((len(self.atomArray), 3))
    # angstroms/second^2
    self.accelMatrix = np.zeros((len(self.atomArray), 3))
    # newtons
    self.forceMatrix = np.zeros((len(self.atomArray), 3))
    # atomc masses
    self.massMatrix = np.array([atom[1]["Type"].value for atom in self.atomArray])
    # joules
    self.potential = 0
    # joules
    self.kinetic = 0

  # calculates length AB given positions

  def distance_(self, P, nplib):
    r = P[0,:] - P[1,:]
    r_mag = nplib.sqrt(nplib.sum(nplib.square(r)))
    return r_mag

  def distance_np(self, P):
    return self.distance_(P, np)

  # calculates angle ABC given positions

  def angle_(self, P, nplib):
    r1 = P[0,:] - P[1,:]
    r2 = P[2,:] - P[1,:]
    dot = nplib.sum(nplib.multiply(r1, r2))
    r1_mag = nplib.sqrt(nplib.sum(nplib.square(r1)))
    r2_mag = nplib.sqrt(nplib.sum(nplib.square(r2)))
    angle = nplib.arccos(dot / (r1_mag * r2_mag))
    return angle

  def angle_np(self, P):
    return self.angle_(P, np)

  # calculates torsional angle ABCD given positions

  def torsionalAngle_(self, P, nplib):
    r1 = P[0,:] - P[1,:]
    r2 = P[1,:] - P[2,:]
    r3 = P[2,:] - P[3,:]
    cp_12 = np.cross(r1, r2)
    cp_32 = np.cross(r3, r2)
    cp = np.array([cp_12, np.zeros(3), cp_32])
    torsionalAngle = self.angle_(cp, nplib)
    return torsionalAngle

  def torsionalAngle_np(self, P):
    return self.torsionalAngle_(P, np)

  # calculates potential energy of molecule given positions

  def calcPotential(self, P):
    self.potential = 0

    distance = jax.vmap(self.distance_np, in_axes = (0, ))

    if len(self.ccPairs) != 0:
      self.potential += 0.5 * np.sum(np.square(distance(P[self.ccPairs]))) * self.K_cc * self.kcal2J / self.N

    if len(self.chPairs) != 0:
      self.potential += 0.5 * np.sum(np.square(distance(P[self.chPairs]))) * self.K_ch * self.kcal2J / self.N

    angle = jax.vmap(self.angle_np, in_axes = (0, ))

    if len(self.cccTriples) != 0:
      self.potential += 0.5 * np.sum(np.square(angle(P[self.cccTriples]))) * self.K_ccc * self.kcal2J / self.N

    if len(self.hchTriples) != 0:
      self.potential += 0.5 * np.sum(np.square(angle(P[self.hchTriples]))) * self.K_hch * self.kcal2J / self.N

    if len(self.cchTriples) != 0:
      self.potential += 0.5 * np.sum(np.square(angle(P[self.cchTriples]))) * self.K_cch * self.kcal2J / self.N

    torsionalAngle = jax.vmap(self.torsionalAngle_np, in_axes = (0, ))

    if len(self.quads) != 0:
      self.potential += 0.5 * np.sum(1 + np.cos(3 * torsionalAngle(P[self.quads]))) * self.K_ccTorsional * self.kcal2J / self.N

    return self.potential

  # calculates kinetic energy of single atom given mass and velocity

  def kineticAtom_(self, M, V, nplib):
    v_mag = nplib.sqrt(nplib.sum(nplib.square(V)))
    return 0.5 * M * v_mag ** 2 * self.amu2kg * self.A2m ** 2 

  def kineticAtom_np(self, M, V):
    return self.kineticAtom_(M, V, np)

  # calculates kinetic energy of molecule given masses and velocities

  def calcKinetic(self, M, V):
    kineticAtom = jax.vmap(self.kineticAtom_np, in_axes = (0, 0))

    self.kinetic = np.sum(kineticAtom(M, V))

    return self.kinetic

  # calculates force matrix

  def calcForce(self, P):
    gradient_v = jax.grad(self.calcPotential)

    self.forceMatrix = -1 * gradient_v(P) / self.A2m

    return self.forceMatrix

  # calculates acceleration matrix of single atom given mass and force

  def accelAtom_(self, M, F, nplib):
    return F / M / self.amu2kg / self.A2m 

  def accelAtom_np(self, M, F):
    return self.accelAtom_(M, F, np)

  # calculates acceleration matrix gives masses and forces

  def calcAccel(self, M, F):
    accelAtom = jax.vmap(self.accelAtom_np, in_axes = (0, 0))

    self.accelMatrix = accelAtom(M, F)

    return self.accelMatrix

  def update(self):
    self.calcPotential(self.posMatrix)
    self.calcKinetic(self.massMatrix, self.velMatrix)
    self.calcForce(self.posMatrix)
    self.calcAccel(self.massMatrix, self.forceMatrix)

    self.posMatrix += self.dt * self.velMatrix
    self.velMatrix += self.dt * self.accelMatrix

    print("potential:")
    print(self.potential)
    print("kinetic:")
    print(self.kinetic)
    print("total:")
    print(self.potential + self.kinetic)
    print()
    print("posMatrix:")
    print(self.posMatrix)
    print("velMatrix:")
    print(self.velMatrix)
    print("accelMatrix")
    print(self.accelMatrix)

dt = 1e-12
sim = mol(ethane, dt)
sim.update()

# class molV2:
#   def __init__ (self, atoms):
#     self.atoms = atoms

#     self.initAtomArrays()
#     self.initPairs()
#     self.initTriples()
#     self.initQuads()
#     self.initMatrices()

#     # kcal/Å^2
#     # kcal/rad^2

#     self.K_cc = 573.8
#     self.K_ch = 222.
#     self.K_ccc = 53.58
#     self.K_hch = 76.28
#     self.K_cch = 44.
#     self.K_ccTorsional = 2.836

#     # atomic masses (amu)

#     self.massC = 12.0107
#     self.massH = 1.00784

#     # ångström to meter
#     # kcal to joules
#     # amu to kg

#     self.A2m = 10e-10
#     self.kcal2J = 4184
#     self.amu2kg = 1.660539e-27

#   def initAtomArrays(self):
#     self.atomArray = []
#     self.atomMap = {}
#     for i, (k, v) in enumerate(self.atoms.items()):
#       self.atomArray.append((k, v))
#       self.atomMap[k] = i

#   def initPairs(self):
#     self.pairs = []
#     for i, (_, v) in enumerate(self.atoms.items()):
#       for atom in v["Neighbors"]:
#         j = self.atomMap[atom]
#         if j < i:
#           continue

#         self.pairs.append(
#           (i, j, v["Type"], self.atoms[atom]["Type"])
#         )

#     self.ccPairs = [(t[0], t[1]) for t in self.pairs if t[2] == atoms.C and t[3] == atoms.C]
#     self.chPairs = [(t[0], t[1]) for t in self.pairs if (t[2] == atoms.H and t[3] == atoms.C) or (t[2] == atoms.C and t[3] == atoms.H)]

#   def initTriples(self):
#     self.triples = []
#     for i, (k, v) in enumerate(self.atoms.items()):
#       if v["Type"] != atoms.C:
#         continue

#       neighbors = v["Neighbors"]
#       for j in range(len(neighbors)):
#         for k in range(j + 1, len(neighbors)):
#           self.triples.append(
#             (self.atomMap[neighbors[j]], i, self.atomMap[neighbors[k]])
#           )

#     self.cccTriples = [t for t in self.triples if self.atomArray[t[0]][1]["Type"] == atoms.C and self.atomArray[t[2]][1]["Type"] == atoms.C]
#     self.hchTriples = [t for t in self.triples if self.atomArray[t[0]][1]["Type"] == atoms.H and self.atomArray[t[2]][1]["Type"] == atoms.H]
#     self.cchTriples = [t for t in self.triples if (self.atomArray[t[0]][1]["Type"] == atoms.H and self.atomArray[t[2]][1]["Type"] == atoms.C) or (self.atomArray[t[0]][1]["Type"] == atoms.C and self.atomArray[t[2]][1]["Type"] == atoms.H)]

#   # only calculates _cc_ quads

#   def initQuads(self):
#     self.quads = []
#     for pair in self.ccPairs:
#       for left in self.atomArray[pair[0]][1]["Neighbors"]:
#         if self.atomMap[left] == pair[1]:
#           continue

#         for right in self.atomArray[pair[1]][1]["Neighbors"]:
#           if self.atomMap[right] == pair[0]:
#             continue

#           self.quads.append(
#             (self.atomMap[left], pair[0], pair[1], self.atomMap[right])
#           )

#   def initMatrices(self):
#     self.posList = [atom[1]["Position"] for atom in self.atomArray]

#   # need to add empty matrix checks

#   def calcPotential(self):
#     self.potential = 0

#     if len(self.ccPairs) != 0:
#       self.ccBond0 = np.array([self.posList[i[0]] for i in self.ccPairs])
#       print(self.ccBond0.shape)
#       self.ccBond1 = np.array([self.posList[i[1]] for i in self.ccPairs])
#       self.ccBond0_1 = self.ccBond0 - self.ccBond1
#       self.ccBondLengthSquare = np.sum(np.square(self.ccBond0_1))

#     if len(self.chPairs) != 0:
#       self.chBond0 = [self.posList[i[0]] for i in self.chPairs]
#       self.chBond1 = [self.posList[i[1]] for i in self.chPairs]
#       self.chBond0_1 = np.subtract(self.chBond0, self.chBond1)
#       self.chBondLengthSquare = np.sum(np.square(self.chBond0_1), axis = 1)

#     if len(self.cccTriples) != 0:
#       self.cccBondAngle0 = [self.posList[i[0]] for i in self.cccTriples]
#       self.cccBondAngle1 = [self.posList[i[1]] for i in self.cccTriples]
#       self.cccBondAngle2 = [self.posList[i[2]] for i in self.cccTriples]
#       self.cccBondAngle0_1 = np.subtract(self.cccBondAngle0, self.cccBondAngle1)
#       self.cccBondAngle2_1 = np.subtract(self.cccBondAngle2, self.cccBondAngle1)
#       self.cccBondAngleDotProd = np.sum(np.multiply(self.cccBondAngle0_1, self.cccBondAngle2_1), axis = 1)
#       self.cccBondAngle0_1Mag = np.sum(np.square(self.cccBondAngle0_1), axis = 1)
#       self.cccBondAngle2_1Mag = np.sum(np.square(self.cccBondAngle2_1), axis = 1)
#       self.cccBondAngleMagProd = np.sum(np.multiply(self.cccBondAngle0_1Mag, self.cccBondAngle2_1Mag), axis = 1)
#       self.cccBondAngle = np.arccos(self.cccBondAngleDotProd / self.cccBondAngleMagProd)

#     if len(self.hchTriples) != 0:
#       self.hchBondAngle0 = [self.posList[i[0]] for i in self.hchTriples]
#       self.hchBondAngle1 = [self.posList[i[1]] for i in self.hchTriples]
#       self.hchBondAngle2 = [self.posList[i[2]] for i in self.hchTriples]
#       self.hchBondAngle0_1 = np.subtract(self.hchBondAngle0, self.hchBondAngle1)
#       self.hchBondAngle2_1 = np.subtract(self.hchBondAngle2, self.hchBondAngle1)
#       self.hchBondAngleDotProd = np.sum(np.multiply(self.hchBondAngle0_1, self.hchBondAngle2_1), axis = 1)
#       self.hchBondAngle0_1Mag = np.sum(np.square(self.hchBondAngle0_1), axis = 1)
#       self.hchBondAngle2_1Mag = np.sum(np.square(self.hchBondAngle2_1), axis = 1)
#       self.hchBondAngleMagProd = np.sum(np.multiply(self.hchBondAngle0_1Mag, self.hchBondAngle2_1Mag), axis = 1)
#       self.hchBondAngle = np.arccos(self.hchBondAngleDotProd / self.hchBondAngleMagProd)

#     if len(self.cchTriples) != 0:
#       self.cchBondAngle0 = [self.posList[i[0]] for i in self.cchTriples]
#       self.cchBondAngle1 = [self.posList[i[1]] for i in self.cchTriples]
#       self.cchBondAngle2 = [self.posList[i[2]] for i in self.cchTriples]
#       self.cchBondAngle0_1 = np.subtract(self.cchBondAngle0, self.cchBondAngle1)
#       self.cchBondAngle2_1 = np.subtract(self.cchBondAngle2, self.cchBondAngle1)
#       self.cchBondAngleDotProd = np.sum(np.multiply(self.cchBondAngle0_1, self.cchBondAngle2_1), axis = 1)
#       self.cchBondAngle0_1Mag = np.sum(np.square(self.cchBondAngle0_1), axis = 1)
#       self.cchBondAngle2_1Mag = np.sum(np.square(self.cchBondAngle2_1), axis = 1)
#       self.cchBondAngleMagProd = np.sum(np.multiply(self.cchBondAngle0_1Mag, self.cchBondAngle2_1Mag), axis = 1)
#       self.cchBondAngle = np.arccos(self.cchBondAngleDotProd / self.cchBondAngleMagProd)

#     print()

#   def update(self):
#     print(self.calcPotential())
      
# dt = 1e-12
# sim = molV2(ethane)

# sim.calcPotential()

# class molDyn:
#   def __init__(self, dt, atoms, pos0, v0, bonds):
#     self.dt = dt
#     self.time = 0
#     self.v = v0
#  #    self.pos = pos0
#      self.F = np.zeros((atoms.shape[0], 3))

# #     self.atoms = atoms
# #     self.bonds = bonds
#     self.potentialEnergy = 0
#     self.kineticEnergy = 0

#     self.ccPairs = []
#     self.chPairs = []
#     self.cccTriples = []
#     self.hchTriples = []
#     self.cchTriples = []
    
#     self.calcPairs()
#     self.calcTriples()

#     self.ccDistances = np.zeros((len(self.ccPairs), 3))
#     self.chDistances = np.zeros((len(self.chPairs), 3))
#     self.cccBondAngles = np.zeros((len(self.cccTriples), 1))
#     self.hchBondAngles = np.zeros((len(self.hchTriples), 1))
#     self.cchBondAngles = np.zeros((len(self.cchTriples), 1))

#     self.calcNumCcTorsionalAngles()

#     self.ccTorsionalAngles = np.zeros((self.numCcTorsionalAngles, 1))

#     # kcal/Å^2
#     # kcal/rad^2

#     self.K_cc = 573.8
#     self.K_ch = 222.
#     self.K_ccc = 53.58
#     self.K_hch = 76.28
#     self.K_cch = 44.
#     self.K_ccTorsional = 2.836

#     # atomic masses (amu)

#     self.massC = 12.0107
#     self.massH = 1.00784

#     # ångström to meter
#     # kcal to joules
#     # amu to kg

#     self.A2m = 10e-10
#     self.kcal2J = 4184
#     self.amu2kg = 1.660539e-27

#   def calcPairs(self):
#     for i in range(len(self.bonds)):
#       for j in range(len(self.bonds[i])):
#         if i < self.bonds[i][j]:
#           if self.atoms[i] and self.atoms[self.bonds[i][j]]:
#             self.ccPairs.append([i, self.bonds[i][j]])
#           if self.atoms[i] ^ self.atoms[self.bonds[i][j]]:
#             self.chPairs.append([i, self.bonds[i][j]])

#   def calcTriples(self):
#     for i in range(len(self.bonds)):
#       if self.atoms[i]:
#         for j in range(len(self.bonds[i])):
#           for k in range(j + 1, len(self.bonds[i])):
#             if self.atoms[self.bonds[i][j]] and self.atoms[self.bonds[i][k]]:
#               self.cccTriples.append([i, self.bonds[i][j], self.bonds[i][k]])
#             elif not(self.atoms[self.bonds[i][j]] or self.atoms[self.bonds[i][k]]):
#               self.hchTriples.append([i, self.bonds[i][j], self.bonds[i][k]])
#             elif self.atoms[self.bonds[i][j]] ^ self.atoms[self.bonds[i][k]]:
#               self.cchTriples.append([i, self.bonds[i][j], self.bonds[i][k]])

#   def calcNumCcTorsionalAngles(self):
#     self.numCcTorsionalAngles = 0
#     for i in range(len(self.ccPairs)):
#       self.numCcTorsionalAngles += (len(self.bonds[self.ccPairs[i][0]]) - 1) * (len(self.bonds[self.ccPairs[i][1]]) - 1)
#     return self.numCcTorsionalAngles

# #  def getVectorMagnitude(self, v):
# #    return np.sqrt(
# #        v[0] * v[0] +
# #        v[1] * v[1] +
# #        v[2] * v[2])

# #  def getVectorDotProduct(self, v1, v2):
# #    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

# #  def getVectorCrossProduct(self, v1, v2):
# #    return np.array([
# #        [v1[1] * v2[2] - v1[2] * v2[1]],
# #        [v1[2] * v2[0] - v1[0] * v2[2]],
# #        [v1[0] * v2[1] - v1[1] * v2[0]]])

#   # rad

#   def getVectorAngle(self, v1, v2):
#     return np.arccos(np.dot(v1, v2.transpose()) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

#   # rad

#   def getVectorTorsionalAngle(self, v1, v2, v3):
#     return self.getVectorAngle(np.cross(v1, v2), np.cross(v1, v3))

#   # Å

#   def calcCcDistances(self):
#     for i in range(len(self.ccPairs)):
#       self.ccDistances[i] = self.pos[self.ccPairs[i][0]] - self.pos[self.ccPairs[i][1]]

#   # Å

#   def calcChDistances(self):
#     for i in range(len(self.chPairs)):
#       self.chDistances[i] = self.pos[self.chPairs[i][0]] - self.pos[self.chPairs[i][1]]

#   # rad

#   def calcCccAngles(self):
#     for i in range(len(self.cccTriples)):
#       self.cccBondAngles[i] = self.getVectorAngle(
#         self.pos[self.cccTriples[i][1]] - self.pos[self.cccTriples[i][0]],
#         self.pos[self.cccTriples[i][2]] - self.pos[self.cccTriples[i][0]])

#   # rad

#   def calcHchAngles(self):
#     for i in range(len(self.hchTriples)):
#       self.hchBondAngles[i] = self.getVectorAngle(
#         self.pos[self.hchTriples[i][1]] - self.pos[self.hchTriples[i][0]],
#         self.pos[self.hchTriples[i][2]] - self.pos[self.hchTriples[i][0]])

#   # rad

#   def calcCchAngles(self):
#     for i in range(len(self.cchTriples)):
#       self.cchBondAngles[i] = self.getVectorAngle(
#         self.pos[self.cchTriples[i][1]] - self.pos[self.cchTriples[i][0]],
#         self.pos[self.cchTriples[i][2]] - self.pos[self.cchTriples[i][0]])

#   # rad

#   def calcCcTorsionalAngles(self):
#     x = 0
#     for i in range(len(self.ccPairs)):
#       for j in range(len(self.bonds[self.ccPairs[i][0]])):
#         if not self.bonds[self.ccPairs[i][0]][j] == self.ccPairs[i][1]:
#           for k in range(len(self.bonds[self.ccPairs[i][1]])):
#             if not self.bonds[self.ccPairs[i][1]][k] == self.ccPairs[i][0]:
#               self.ccTorsionalAngles[x] = self.getVectorTorsionalAngle(
#                 self.pos[self.ccPairs[i][0]] - self.pos[self.ccPairs[i][1]],
#                 self.pos[self.bonds[self.ccPairs[i][0]][j]] - self.pos[self.ccPairs[i][0]],
#                 self.pos[self.bonds[self.ccPairs[i][1]][k]] - self.pos[self.ccPairs[i][1]])
#               x += 1

#   # J

#   def calcPotential(self):
#     self.calcCcDistances()
#     self.calcChDistances()
#     self.calcCccAngles()
#     self.calcHchAngles()
#     self.calcCchAngles()
#     self.calcCcTorsionalAngles()

#     self.potentialEnergy = 0

#     for i in range(len(self.ccDistances)):
#       self.potentialEnergy += 0.5 * self.K_cc * np.square(np.linalg.norm(self.ccDistances[i])) * self.A2m ** 2 * self.kcal2J

#     for i in range(len(self.chDistances)):
#       self.potentialEnergy += 0.5 * self.K_ch * np.square(np.linalg.norm(self.chDistances[i])) * self.A2m ** 2 * self.kcal2J

#     for i in range(len(self.cccBondAngles)):
#       self.potentialEnergy += 0.5 * self.K_ccc * np.square(self.cccBondAngles[i][0]) * self.kcal2J

#     for i in range(len(self.hchBondAngles)):
#       self.potentialEnergy += 0.5 * self.K_hch * np.square(self.hchBondAngles[i][0]) * self.kcal2J

#     for i in range(len(self.cchBondAngles)):
#       self.potentialEnergy += 0.5 * self.K_cch * np.square(self.cchBondAngles[i][0]) * self.kcal2J

#     for i in range(len(self.ccTorsionalAngles)):
#       self.potentialEnergy += 0.5 * self.K_ccTorsional * (1 + np.cos(3 * self.ccTorsionalAngles[i][0])) * self.kcal2J
    
#     return self.potentialEnergy

#   # J

#   def calcKinetic(self):
#     self.kineticEnergy = 0

#     for i in range(self.v.shape[0]):
#       self.kineticEnergy += 0.5 * np.linalg.norm(self.v[i]) * self.A2m ** 2
  
#   # N

#   def calcForceManual(self):
#     originalPE = self.potentialEnergy

#     if not originalPE == 0:
#       for i in range(len(self.F)):
#         h = 1e-12
#         for j in range(3):
#           self.pos[i, j] += h
#           self.F[i, j] = (originalPE - self.calcPotential()) / h
#           self.pos[i, j] -= h

#   def update(self):
#     self.calcForceManual()

#     self.pos += self.v * self.dt
#     self.v += self.F * self.dt

#     self.calcPotential()
#     self.calcKinetic()

#     print(self.pos)
#     print(self.v)
#     print(self.F)
#     print(-self.potentialEnergy)
#     print(self.kineticEnergy)
#     print(self.kineticEnergy - self.potentialEnergy)
#     print()

# sim = molDyn(
#   dt,
#   np.array([atoms.C, atoms.C, atoms.H, atoms.H, atoms.H, atoms.H, atoms.H, atoms.H]),
#   np.array([
#       [0., 0., 0.],
#       [2., 0., 0.],
#       [0., -1., 0.],
#       [-1., 1., 2.],
#       [-1., 1., -1.],
#       [2., 1., 0.],
#       [3., 1., 1.],
#       [3., 0., -1.]
#     ]),
#   np.array([
#       [0., 0., 0.],
#       [2., 0., 0.],
#       [0., -1., 0.],
#       [-1., 1., 2.],
#       [-1., 1., -1.],
#       [2., 1., 0.],
#       [3., 1., 1.],
#       [3., 0., -1.]
#     ]),
#   [
#       [1, 2, 3, 4],
#       [0, 5, 6, 7],
#       [0],
#       [0],
#       [0],
#       [1],
#       [1],
#       [1]
#     ])

# for i in range(1000):
#   sim.update()