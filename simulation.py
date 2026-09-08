import random as rand

import jax
import jax.numpy as np

from constants import (
  A2m,
  X_cc,
  X_cc_aromatic,
  X_ccc,
  X_ccc_aromatic,
  X_cch,
  X_cch_aromatic,
  X_ch,
  X_hch,
  amu2kg,
  angleEnergyK_ccc,
  angleEnergyK_ccc_aromatic,
  angleEnergyK_cch,
  angleEnergyK_cch_aromatic,
  angleEnergyK_ccTorsional,
  angleEnergyK_ccTorsional_aromatic,
  angleEnergyK_hch,
  dist_unit,
  distEnergyK_cc,
  distEnergyK_cc_aromatic,
  distEnergyK_ch,
  jit_funcs,
  mass_unit,
  time_unit,
  vmap_funcs,
)
from molecules import atoms


class mol:
  def __init__ (self, atoms, dt, randomize):
    self.atoms = atoms
    self.dt = dt
    self.randomize = randomize

    self.initAtomArrays()
    self.initPairs()
    self.initTriples()
    self.initQuads()
    self.initRandMatrix()
    self.initMatrices()
    self.initJax()

  def initAtomArrays(self):
    self.atomArray = []
    self.atomMap = {}
    for i, (k, v) in enumerate(self.atoms.items()):
      self.atomArray.append((k, v))
      self.atomMap[k] = i

  def initPairs(self):
    self.pairs = []
    for i, (_, v) in enumerate(self.atoms.items()):
      for atom in v["Neighbors"]:
        j = self.atomMap[atom]
        if j < i:
          continue

        self.pairs.append(
          (i, j, v["Type"], self.atoms[atom]["Type"],
            v.get("Aromatic", False) and self.atoms[atom].get("Aromatic", False))
        )

    self.ccPairs = np.array([np.array([t[0], t[1]]) for t in self.pairs if t[2] == atoms.C and t[3] == atoms.C and not t[4]])
    self.ccAromaticPairs = np.array([np.array([t[0], t[1]]) for t in self.pairs if t[2] == atoms.C and t[3] == atoms.C and t[4]])
    self.chPairs = np.array([np.array([t[0], t[1]]) for t in self.pairs if (t[2] == atoms.H and t[3] == atoms.C) or (t[2] == atoms.C and t[3] == atoms.H)])

    if len(self.ccPairs) == 0:
      self.ccPairs = np.full((0, 2), 0)

    if len(self.ccAromaticPairs) == 0:
      self.ccAromaticPairs = np.full((0, 2), 0)

    if len(self.chPairs) == 0:
      self.chPairs = np.full((0, 2), 0)

    self.allCcPairs = np.concatenate((self.ccPairs, self.ccAromaticPairs), axis = 0)
    self.atomPairs = np.concatenate((self.ccPairs, self.ccAromaticPairs, self.chPairs), axis = 0)
    self.pairEnergyConstants = np.concatenate((
      np.full((1, len(self.ccPairs)), distEnergyK_cc),
      np.full((1, len(self.ccAromaticPairs)), distEnergyK_cc_aromatic),
      np.full((1, len(self.chPairs)), distEnergyK_ch)),
      axis = 1)

  def initTriples(self):
    self.triples = []
    for i, (k, v) in enumerate(self.atoms.items()):
      if v["Type"] != atoms.C:
        continue

      isAromatic = v.get("Aromatic", False)
      neighbors = v["Neighbors"]
      for j in range(len(neighbors)):
        for m in range(j + 1, len(neighbors)):
          self.triples.append(
            (self.atomMap[neighbors[j]], i, self.atomMap[neighbors[m]], isAromatic)
          )

    self.cccTriples = np.array([[t[0], t[1], t[2]] for t in self.triples if self.atomArray[t[0]][1]["Type"] == atoms.C and self.atomArray[t[2]][1]["Type"] == atoms.C and not t[3]])
    self.cccAromaticTriples = np.array([[t[0], t[1], t[2]] for t in self.triples if self.atomArray[t[0]][1]["Type"] == atoms.C and self.atomArray[t[2]][1]["Type"] == atoms.C and t[3]])
    self.hchTriples = np.array([[t[0], t[1], t[2]] for t in self.triples if self.atomArray[t[0]][1]["Type"] == atoms.H and self.atomArray[t[2]][1]["Type"] == atoms.H])
    self.cchTriples = np.array([[t[0], t[1], t[2]] for t in self.triples if ((self.atomArray[t[0]][1]["Type"] == atoms.H and self.atomArray[t[2]][1]["Type"] == atoms.C) \
      or (self.atomArray[t[0]][1]["Type"] == atoms.C and self.atomArray[t[2]][1]["Type"] == atoms.H)) and not t[3]])
    self.cchAromaticTriples = np.array([[t[0], t[1], t[2]] for t in self.triples if ((self.atomArray[t[0]][1]["Type"] == atoms.H and self.atomArray[t[2]][1]["Type"] == atoms.C) \
      or (self.atomArray[t[0]][1]["Type"] == atoms.C and self.atomArray[t[2]][1]["Type"] == atoms.H)) and t[3]])

    if len(self.cccTriples) == 0:
      self.cccTriples = np.full((0, 3), 0)

    if len(self.cccAromaticTriples) == 0:
      self.cccAromaticTriples = np.full((0, 3), 0)

    if len(self.hchTriples) == 0:
      self.hchTriples = np.full((0, 3), 0)

    if len(self.cchTriples) == 0:
      self.cchTriples = np.full((0, 3), 0)

    if len(self.cchAromaticTriples) == 0:
      self.cchAromaticTriples = np.full((0, 3), 0)

    self.atomTriples = np.concatenate((self.cccTriples, self.cccAromaticTriples, self.hchTriples, self.cchTriples, self.cchAromaticTriples), axis = 0)
    self.triplesAngleEneryConstants = np.concatenate((
      np.full((1, len(self.cccTriples)), angleEnergyK_ccc),
      np.full((1, len(self.cccAromaticTriples)), angleEnergyK_ccc_aromatic),
      np.full((1, len(self.hchTriples)), angleEnergyK_hch),
      np.full((1, len(self.cchTriples)), angleEnergyK_cch),
      np.full((1, len(self.cchAromaticTriples)), angleEnergyK_cch_aromatic)),
      axis = 1)

  def buildQuads(self, pairs):
    quads = []
    for pair in pairs:
      for left in self.atomArray[pair[0]][1]["Neighbors"]:
        if self.atomMap[left] == pair[1]:
          continue

        for right in self.atomArray[pair[1]][1]["Neighbors"]:
          if self.atomMap[right] == pair[0]:
            continue

          quads.append(
            (self.atomMap[left], pair[0], pair[1], self.atomMap[right])
          )

    if len(quads) == 0:
      return np.full((0, 4), 0)

    return np.array(quads)

  def initQuads(self):
    self.quads = self.buildQuads(self.ccPairs)
    self.quadsAromatic = self.buildQuads(self.ccAromaticPairs)

  def initRandMatrix(self):
    firstLine = True

    for _ in range(len(self.atomArray)):
      theta = rand.uniform(0, 2 * np.pi)
      z = rand.uniform(-1, 1)
      randVector = np.array([[np.sqrt(1 - z ** 2) * np.cos(theta),
            np.sqrt(1 - z ** 2) * np.sin(theta),
            z]]) * rand.uniform(0, self.randomize)

      if not firstLine:
        self.randMatrix = np.concatenate((self.randMatrix, randVector), axis = 0)
      else:
        self.randMatrix = np.array(randVector)
        firstLine = False

  def initMatrices(self):
    # angstroms
    self.posMatrix = np.array([([pos * A2m for pos in atom[1]["Position"]]) for atom in self.atomArray]) + self.randMatrix
    # angstroms/second
    self.velMatrix = np.zeros((len(self.atomArray), 3))
    # angstroms/second^2
    self.accelMatrix = np.zeros((len(self.atomArray), 3))
    self.prevAccelMatrix = np.zeros((len(self.atomArray), 3))
    # newtons
    self.forceMatrix = np.zeros((len(self.atomArray), 3))
    # atomic masses
    self.massMatrix = np.array([atom[1]["Type"].value * amu2kg for atom in self.atomArray])
    # joules
    self.potential = 0
    # seconds
    self.t = 0
    # tick index
    self.currTick = 0

    self.M_cc = np.zeros((len(self.ccPairs), 1)) + X_cc
    self.M_cc_aromatic = np.zeros((len(self.ccAromaticPairs), 1)) + X_cc_aromatic
    self.M_ch = np.zeros((len(self.chPairs), 1)) + X_ch
    self.M_ccc = np.zeros((len(self.cccTriples), 1)) + X_ccc
    self.M_ccc_aromatic = np.zeros((len(self.cccAromaticTriples), 1)) + X_ccc_aromatic
    self.M_hch = np.zeros((len(self.hchTriples), 1)) + X_hch
    self.M_cch = np.zeros((len(self.cchTriples), 1)) + X_cch
    self.M_cch_aromatic = np.zeros((len(self.cchAromaticTriples), 1)) + X_cch_aromatic

    self.M_pairs = np.concatenate((self.M_cc, self.M_cc_aromatic, self.M_ch), axis = 0).squeeze()
    self.M_triples = np.concatenate((self.M_ccc, self.M_ccc_aromatic, self.M_hch, self.M_cch, self.M_cch_aromatic), axis = 0).squeeze()

  def vmap(self, f, in_axes):
    return jax.vmap(f, in_axes)

  def jit(self, f):
    if jit_funcs:
      return jax.jit(f)
    else:
      return f

  def initJax(self):
    self.distance_v = self.vmap(self.distance_(True), in_axes = (0, ))
    self.angle_v = self.vmap(self.angle_(True), in_axes = (0, ))
    self.cosTorsionalAngle_v = self.vmap(self.cosTorsionalAngle_(True), in_axes = (0, ))
    self.accelAtom_v = self.jit(self.vmap(self.accelAtom_, in_axes = (0, 0)))

    self.update_j = self.jit(self.update(vmap_funcs))  # unused, equivalent to calling update_loop_j(1, ...) in a Python loop
    self.record_j = self.jit(self.record(vmap_funcs))
    self.update_loop_j = self.jit(self.update_loop(vmap_funcs))

  def distance_(self, use_v):
    def v(P):
      p0 = P[0]
      p1 = P[1]

      r = p0 - p1
      r_mag = np.sqrt(np.sum(np.square(r)))
      return r_mag

    def n(P):
      p0 = P[...,[0],[0,1,2]]
      p1 = P[...,[1],[0,1,2]]

      r = p0 - p1
      r_mag = np.sqrt(np.sum(np.square(r), axis = 1))
      return r_mag

    return v if use_v else n

  def cosAngle_(self, P):
    p0 = P[0]
    p1 = P[1]
    p2 = P[2]

    r1 = p0 - p1
    r2 = p2 - p1
    dot = np.sum(np.multiply(r1, r2))
    r1_mag = np.sqrt(np.sum(np.square(r1)))
    r2_mag = np.sqrt(np.sum(np.square(r2)))
    return dot / (r1_mag * r2_mag)

  def cosAngle(self, P):
    p0 = P[...,[0],[0,1,2]]
    p1 = P[...,[1],[0,1,2]]
    p2 = P[...,[2],[0,1,2]]

    r1 = p0 - p1
    r2 = p2 - p1
    dot = np.sum(np.multiply(r1, r2), axis = 1)
    r1_mag = np.sqrt(np.sum(np.square(r1), axis = 1))
    r2_mag = np.sqrt(np.sum(np.square(r2), axis = 1))
    return dot / (r1_mag * r2_mag)

  def angle_(self, use_v):
    cosAngle = self.cosAngle_ if use_v else self.cosAngle
    def angle(P):
      return np.arccos(cosAngle(P))

    return angle

  def torsionVecs_(self, P):
      p0 = P[0]
      p1 = P[1]
      p2 = P[2]
      p3 = P[3]

      r1 = p0 - p1
      r2 = p1 - p2
      r3 = p3 - p2
      cp_12 = np.cross(r1, r2)
      cp_32 = np.cross(r3, r2)
      return np.dstack((cp_12, np.zeros(cp_12.shape), cp_32)) \
        .squeeze() \
        .transpose([1, 0])

  def torsionVecs(self, P):
      p0 = P[...,[0],[0,1,2]]
      p1 = P[...,[1],[0,1,2]]
      p2 = P[...,[2],[0,1,2]]
      p3 = P[...,[3],[0,1,2]]

      r1 = p0 - p1
      r2 = p1 - p2
      r3 = p3 - p2
      cp_12 = np.cross(r1, r2)
      cp_32 = np.cross(r3, r2)
      return np.dstack((cp_12, np.zeros(cp_12.shape), cp_32)) \
        .squeeze() \
        .transpose([0, 2, 1])

  def cosTorsionalAngle_(self, use_v):
    cosAngle = self.cosAngle_ if use_v else self.cosAngle
    torsionVecs = self.torsionVecs_ if use_v else self.torsionVecs

    def internal(P):
      return cosAngle(torsionVecs(P))

    return internal

  def getCalcPotential(self, use_v):
    atomPairs = self.atomPairs
    pairEnergyConstants = self.pairEnergyConstants
    atomTriples = self.atomTriples
    triplesAngleEneryConstants = self.triplesAngleEneryConstants
    quads = self.quads
    quadsAromatic = self.quadsAromatic
    M_pairs = self.M_pairs
    M_triples = self.M_triples

    if use_v:
      angle = self.angle_v
      cosTorsionalAngle = self.cosTorsionalAngle_v
      distance = self.distance_v
    else:
      angle = self.angle_(False)
      cosTorsionalAngle = self.cosTorsionalAngle_(False)
      distance = self.distance_(False)

    def calcPotential_(pos):
      potential_0 = 0.5 * np.sum(
        np.multiply(
          np.square(distance(pos[atomPairs]) - M_pairs),
          pairEnergyConstants))

      potential_0 += 0.5 * np.sum(
        np.multiply(
          np.square(angle(pos[atomTriples]) - M_triples),
          triplesAngleEneryConstants))

      cosAngle = cosTorsionalAngle(pos[quads])
      potential_0 += 0.5 \
        * np.sum(1 + 4 * cosAngle ** 3 - 3 * cosAngle) \
        * angleEnergyK_ccTorsional

      # aromatic ring quads: 2-fold potential with planar (phi=0/180) minima,
      # via cos(2*phi) = 2*cosAngle^2 - 1, instead of the 3-fold alkane form
      # above (which has no planar minimum and puckers the ring).
      cosAngleAromatic = cosTorsionalAngle(pos[quadsAromatic])
      potential_0 += 0.5 \
        * np.sum(1 - (2 * cosAngleAromatic ** 2 - 1)) \
        * angleEnergyK_ccTorsional_aromatic

      return potential_0
    return calcPotential_

  def calcKinetic_(self):
    massMatrix = self.massMatrix

    def internal(V):
      sq = np.square(V).transpose()
      sq_sum = np.sum(sq, axis = 0)
      mv2 = np.sum(np.multiply(massMatrix, sq_sum), axis = 0)
      return mv2 * 0.5 * (A2m ** 2)

    return internal

  def calcEnergy_(self, use_v):
    calcPotential = self.getCalcPotential(use_v)
    calcKinetic = self.calcKinetic_()
    def internal(P, V):
      return (calcPotential(P), calcKinetic(V))

    return internal

  def calcForce(self, use_v):
    gradient = jax.grad(self.getCalcPotential(use_v))
    def internal(P):
      return -1 * gradient(P) / A2m

    return internal

  def accelAtom_(self, M, F):
    return F / M

  def updatePosition(self, P, V, A, pA, dt):
    # using dA improves speed/accuracy of simulation (3rd degree taylor series)
    dA = A - pA
    P = P + V * dt + A * (dt * dt / 2) + dA * (dt * dt / 3)
    V = V + A * dt + dA * (dt / 2)
    return (P, V)

  def update_loop(self, use_v):
    update_func = self.update(use_v)

    def loop_func(_, tupl):
      return update_func(tupl[0], tupl[1], tupl[2])

    def internal(loops, tupl):
      return jax.lax.fori_loop(0, loops, loop_func, tupl)

    return internal

  def update(self, use_v):
    calcForce = self.calcForce(use_v)
    massMatrix = self.massMatrix
    updatePosition = self.updatePosition
    accelAtom = self.accelAtom_v
    dt = self.dt

    def internal(accel, vel, pos):
      prevAccel = accel
      forceMatrix = calcForce(pos)
      accel = accelAtom(massMatrix, forceMatrix)
      (pos, vel) = updatePosition(pos, vel, accel, prevAccel, dt)
      return (accel, vel, pos)

    return internal

  def record(self, use_v):
    calcEnergy = self.calcEnergy_(use_v)
    distance = self.distance_v if use_v else self.distance_(False)
    ccPairs = self.allCcPairs
    chPairs = self.chPairs
    atoms = len(self.atomArray)
    idx = np.array([range(atoms)]).transpose()

    def internal(t, pos, vel):
      tM = np.full((atoms, 1), t)
      arr = np.concatenate((tM, idx, pos), axis = 1)
      (potential, kinetic) = calcEnergy(pos, vel)
      return (
        arr,
        np.array([[
          t * 1e12,
          potential * mass_unit * (dist_unit / time_unit) ** 2 * 1e21,
          kinetic * mass_unit * (dist_unit / time_unit) ** 2 * 1e21,
          np.average(distance(pos[ccPairs])),
          np.average(distance(pos[chPairs]))]]))

    return internal
