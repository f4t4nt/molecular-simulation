import argparse as ap
from collections import namedtuple
import csv
from enum import Enum
import jax as jax
import jax.numpy as np
import math as math
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
import pickle as pl
import random as rand
import time as time

jax.config.update('jax_enable_x64', True)

# masses in amu
class atoms(Enum):
  C = 12.0107
  H = 1.00784

# coordinates retrieved from https://cccbdb.nist.gov Experimental >> Geometry >> Experimental Geometries

molecules = {}

ethane = {
    "C1" : {
      "Type" : atoms.C,
      "Neighbors" : ["C2","H3","H4","H5"],
      "Position" : np.array([0, 0, 0.7680])
    },
    "C2" : {
      "Type" : atoms.C,
      "Neighbors" : ["C1","H6","H7","H8"],
      "Position" : np.array([0, 0, -0.7680])
    },
    "H3" : {
      "Type" : atoms.H,
      "Neighbors" : ["C1"],
      "Position" : np.array([-1.0192, 0, 1.1573])
    },
    "H4" : {
      "Type" : atoms.H,
      "Neighbors" : ["C1"],
      "Position" : np.array([0.5096, 0.8826, 1.1573])
    },
    "H5" : {
      "Type" : atoms.H,
      "Neighbors" : ["C1"],
      "Position" : np.array([0.5096, -0.8826, 1.1573])
    },
    "H6" : {
      "Type" : atoms.H,
      "Neighbors" : ["C2"],
      "Position" : np.array([1.0192, 0, -1.1573])
    },
    "H7" : {
      "Type" : atoms.H,
      "Neighbors" : ["C2"],
      "Position" : np.array([-0.5096, -0.8826, -1.1573])
    },
    "H8" : {
      "Type" : atoms.H,
      "Neighbors" : ["C2"],
      "Position" : np.array([-0.5096, 0.8826, -1.1573])
    }
  }

molecules["ethane"] = ethane

propane = {
  "C1" : {
    "Type" : atoms.C,
    "Neighbors" : ["C2","C3","H4","H5"],
    "Position" : np.array([0, 0.5863, 0])
  },
  "C2" : {
    "Type" : atoms.C,
    "Neighbors" : ["C1", "H6", "H8", "H9"],
    "Position" : np.array([-1.2681, -0.2626, 0])
  },
  "C3" : {
    "Type" : atoms.C,
    "Neighbors" : ["C1", "H7", "H10", "H11"],
    "Position" : np.array([1.2681, -0.2626, 0])
  },
  "H4" : {
    "Type" : atoms.H,
    "Neighbors" : ["C1"],
    "Position" : np.array([0, 1.2449, 0.876])
  },
  "H5" : {
    "Type" : atoms.H,
    "Neighbors" : ["C1"],
    "Position" : np.array([-0.0003, 1.2453, -0.876])
  },
  "H6" : {
    "Type" : atoms.H,
    "Neighbors" : ["C2"],
    "Position" : np.array([-2.1576, 0.3742, 0])
  },
  "H7" : {
    "Type" : atoms.H,
    "Neighbors" : ["C3"],
    "Position" : np.array([2.1576, 0.3743, 0])
  },
  "H8" : {
    "Type" : atoms.H,
    "Neighbors" : ["C2"],
    "Position" : np.array([-1.3271, -0.9014, 0.88])
  },
  "H9" : {
    "Type" : atoms.H,
    "Neighbors" : ["C2"],
    "Position" : np.array([-1.3271, -0.9014, -0.88])
  },
  "H10" : {
    "Type" : atoms.H,
    "Neighbors" : ["C3"],
    "Position" : np.array([1.3271, -0.9014, 0.88])
  },
  "H11" : {
    "Type" : atoms.H,
    "Neighbors" : ["C3"],
    "Position" : np.array([1.3272, -0.9014, -0.88])
  }
}

molecules["propane"] = propane

isobutane = {
  "C1" : {
    "Type" : atoms.C,
    "Neighbors" : ["H2", "C3", "C4", "C5"],
    "Position" : np.array([0, 0, 0.365])
  },
  "H2" : {
    "Type" : atoms.H,
    "Neighbors" : ["C1"],
    "Position" : np.array([0, 0, 1.473])
  },
  "C3" : {
    "Type" : atoms.C,
    "Neighbors" : ["C1", "H6", "H9", "H10"],
    "Position" : np.array([0, 1.4528, 0.0987])
  },
  "C4" : {
    "Type" : atoms.C,
    "Neighbors" : ["C1", "H7", "H11", "H12"],
    "Position" : np.array([1.2582, -0.7264, -0.0987])
  },
  "C5" : {
    "Type" : atoms.C,
    "Neighbors" : ["C1", "H8", "H13", "H14"],
    "Position" : np.array([-1.2582, -0.7264, -0.0987])
  },
  "H6" : {
    "Type" : atoms.H,
    "Neighbors" : ["C3"],
    "Position" : np.array([0, 1.4867, -1.1931])
  },
  "H7" : {
    "Type" : atoms.H,
    "Neighbors" : ["C4"],
    "Position" : np.array([1.2875, -0.7433, -1.1931])
  },
  "H8" : {
    "Type" : atoms.H,
    "Neighbors" : ["C5"],
    "Position" : np.array([-1.2875, -0.7433, -1.1931])
  },
  "H9" : {
    "Type" : atoms.H,
    "Neighbors" : ["C3"],
    "Position" : np.array([0.8941, 1.9575, 0.2821])
  },
  "H10" : {
    "Type" : atoms.H,
    "Neighbors" : ["C3"],
    "Position" : np.array([-0.8941, 1.9575, 0.2821])
  },
  "H11" : {
    "Type" : atoms.H,
    "Neighbors" : ["C4"],
    "Position" : np.array([1.2482, -1.752, 0.2821])
  },
  "H12" : {
    "Type" : atoms.H,
    "Neighbors" : ["C4"],
    "Position" : np.array([2.1422, -0.2045, 0.2821])
  },
  "H13" : {
    "Type" : atoms.H,
    "Neighbors" : ["C5"],
    "Position" : np.array([-2.1422, -0.2045, 0.2821])
  },
  "H14" : {
    "Type" : atoms.H,
    "Neighbors" : ["C5"],
    "Position" : np.array([-1.2482, -1.753, 0.2821])
  }
}

molecules["isobutane"] = isobutane

benzene = {
  "C1" : {
    "Type" : atoms.C,
    "Neighbors" : ["C2", "C6", "H7"],
    "Position" : np.array([0, 1.397, 0])
  },
  "C2" : {
    "Type" : atoms.C,
    "Neighbors" : ["C1", "C3", "H8"],
    "Position" : np.array([1.2098, 0.6985, 0])
  },
  "C3" : {
    "Type" : atoms.C,
    "Neighbors" : ["C2", "C4", "H9"],
    "Position" : np.array([1.2098, -0.6985, 0])
  },
  "C4" : {
    "Type" : atoms.C,
    "Neighbors" : ["C3", "C5", "H10"],
    "Position" : np.array([0, -1.397, 0])
  },
  "C5" : {
    "Type" : atoms.C,
    "Neighbors" : ["C4", "C6", "H11"],
    "Position" : np.array([-1.2098, -0.6985, 0])
  },
  "C6" : {
    "Type" : atoms.C,
    "Neighbors" : ["C5", "C1", "H12"],
    "Position" : np.array([-1.2098, 0.6985, 0])
  },
  "H7" : {
    "Type" : atoms.H,
    "Neighbors" : ["C1"],
    "Position" : np.array([0, 2.481, 0])
  },
  "H8" : {
    "Type" : atoms.H,
    "Neighbors" : ["C2"],
    "Position" : np.array([2.1486, 1.2405, 0])
  },
  "H9" : {
    "Type" : atoms.H,
    "Neighbors" : ["C3"],
    "Position" : np.array([2.1486, -1.2405, 0])
  },
  "H10" : {
    "Type" : atoms.H,
    "Neighbors" : ["C4"],
    "Position" : np.array([0, -2.481, 0])
  },
  "H11" : {
    "Type" : atoms.H,
    "Neighbors" : ["C5"],
    "Position" : np.array([-2.1486, -1.2405, 0])
  },
  "H12" : {
    "Type" : atoms.H,
    "Neighbors" : ["C6"],
    "Position" : np.array([-2.1486, 1.2405, 0])
  }
}

molecules["benzene"] = benzene

# scaled units to prevent overflow

time_unit = 1e-12
dist_unit = 1e-10
mass_unit = 1e-20

#############
# CONSTANTS #
#############

# kcal/Å^2
# kcal/rad^2

K_cc = 573.8
K_ch = 222.
K_ccc = 53.58
K_hch = 76.28
K_cch = 44.
K_ccTorsional = 2.836

# Avogadro's number

N = 6.0221409e+23

# ångström to meter
# kcal to joules
# amu to kg

A2m = 1e-10 / dist_unit
amu2kg = 1.660539e-27 / mass_unit
kcal2J = 4186.4
kcal2MU = kcal2J * (1 / mass_unit) * (time_unit / dist_unit) ** 2

# Å
# rad
X_cc = 1.455 * A2m
X_ch = 1.099 * A2m
X_ccc = 1.937
X_hch = 1.911
X_cch = 1.911

# distance-energy conversion constants
# angle-energy conversion constants

distEnergyK_cc = K_cc * kcal2MU / (N * A2m ** 2)
distEnergyK_ch = K_ch * kcal2MU / (N * A2m ** 2)
angleEnergyK_ccc = K_ccc * kcal2MU / N
angleEnergyK_hch = K_hch * kcal2MU / N
angleEnergyK_cch = K_cch * kcal2MU / N
angleEnergyK_ccTorsional = K_ccTorsional * kcal2MU / N

# jit/vmap switch

jit_funcs = True
vmap_funcs = True

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

  ########################################################
  # reads bond relations and converts to readable format #
  ########################################################

  def initAtomArrays(self):
    self.atomArray = []
    self.atomMap = {}
    for i, (k, v) in enumerate(self.atoms.items()):
      self.atomArray.append((k, v))
      self.atomMap[k] = i

  #################################
  # creates array of CC, CH pairs #
  #################################

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

    if len(self.ccPairs) == 0:
      self.ccPairs = np.full((0, 2), 0)

    if len(self.chPairs) == 0:
      self.chPairs = np.full((0, 2), 0)

    self.atomPairs = np.concatenate((self.ccPairs, self.chPairs), axis = 0)
    self.pairEnergyConstants = np.concatenate((
      np.full((1, len(self.ccPairs)), distEnergyK_cc),
      np.full((1, len(self.chPairs)), distEnergyK_ch)),
      axis = 1)
  
  ##########################################
  # creates array of CCC, HCH, CCH triples #
  ##########################################

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

    if len(self.cccTriples) == 0:
      self.cccTriples = np.full((0, 3), 0)

    if len(self.hchTriples) == 0:
      self.hchTriples = np.full((0, 3), 0)

    if len(self.cchTriples) == 0:
      self.cchTriples = np.full((0, 3), 0)

    self.atomTriples = np.concatenate((self.cccTriples, self.hchTriples, self.cchTriples), axis = 0)
    self.triplesAngleEneryConstants = np.concatenate((
      np.full((1, len(self.cccTriples)), angleEnergyK_ccc),
      np.full((1, len(self.hchTriples)), angleEnergyK_hch),
      np.full((1, len(self.cchTriples)), angleEnergyK_cch)),
      axis = 1)

  ###############################
  # creates array of _CC_ quads #
  ###############################

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

  #####################################################################
  # creates random displacement matrix to offset predefined positions #
  #####################################################################

  def initRandMatrix(self):
    firstLine = True

    for i in range(len(self.atomArray)):
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

  #############################
  # creates variable matrices #
  #############################

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
    # atomc masses
    self.massMatrix = np.array([atom[1]["Type"].value * amu2kg for atom in self.atomArray])
    # joules
    self.potential = 0
    # seconds
    self.t = 0
    # tick index
    self.currTick = 0
    
    self.M_cc = np.zeros((len(self.ccPairs), 1)) + X_cc
    self.M_ch = np.zeros((len(self.chPairs), 1)) + X_ch
    self.M_ccc = np.zeros((len(self.cccTriples), 1)) + X_ccc
    self.M_hch = np.zeros((len(self.hchTriples), 1)) + X_hch
    self.M_cch = np.zeros((len(self.cchTriples), 1)) + X_cch

    self.M_pairs = np.concatenate((self.M_cc, self.M_ch), axis = 0).squeeze()
    self.M_triples = np.concatenate((self.M_ccc, self.M_hch, self.M_cch), axis = 0).squeeze()

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

    self.update_j = self.jit(self.update(vmap_funcs))
    self.record_j = self.jit(self.record(vmap_funcs))
    self.update_loop_j = self.jit(self.update_loop(vmap_funcs))

  ########################################
  # calculates length AB given positions #
  ########################################

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

  ##################################################
  # calculates cosine of angle ABC given positions #
  ##################################################

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

  ########################################
  # calculates angle ABC given positions #
  ########################################

  def angle_(self, use_v):
    cosAngle = self.cosAngle_ if use_v else self.cosAngle
    def angle(P):
      return np.arccos(cosAngle(P))

    return angle

  ###################################################
  # calculates torsional angle ABCD given positions #
  ###################################################

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

  ###########################################################
  # calculates potential energy of molecule given positions #
  ##########################################################

  def getCalcPotential(self, use_v):
    atomPairs = self.atomPairs
    pairEnergyConstants = self.pairEnergyConstants
    atomTriples = self.atomTriples
    triplesAngleEneryConstants = self.triplesAngleEneryConstants
    quads = self.quads
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

  ###########################
  # calculates force matrix #
  ###########################

  def calcForce(self, use_v):
    gradient = jax.grad(self.getCalcPotential(use_v))
    def internal(P):
      return -1 * gradient(P) / A2m

    return internal

  def accelAtom_(self, M, F):
    return F / M # / self.amu2kg

  #############################################################################
  # calculates velocity matrix of single atom given velocity and acceleration #
  #############################################################################

  def updatePosition(self, P, V, A, pA, dt):
    # using dA improves speed/accuracy of simulation (3rd degree taylor series)
    dA = A - pA
    P = P + V * dt + A * (dt * dt / 2) + dA * (dt * dt / 3)
    V = V + A * dt + dA * (dt / 2)
    return (P, V)

  def update_loop(self, use_v):
    update_func = self.update(use_v)

    def loop_func(i, tupl):
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
    ccPairs = self.ccPairs
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

def get_df_posHistoryArr(positionHistoryArr, df):
  df2 = pd.DataFrame(data = positionHistoryArr, columns=["time", "atomId", "posX", "posY", "posZ"]) \
    .astype({"atomId": "int16"})

  if df is None:
    return df2

  return df.append(df2)

def get_df_tickHistoryArr(tickHistoryArray, df):
  tickHistDf = pd.DataFrame(
    data = tickHistoryArray,
    columns = ["time", "potentialE", "kineticE", "CC_Bonds", "CH_Bonds"])

  if df is None:
    return tickHistDf
  
  return df.append(tickHistDf)


def Main(
  input_mol,
  input_dt,
  input_randomize_const,
  input_ticks,
  input_scale,
  input_stablization_const):

  molecule = molecules[input_mol]
  dt = input_dt / time_unit
  randomize = input_randomize_const

  print("--- 0 seconds ---")

  sim = mol(molecule, dt, randomize)

  start_time = time.perf_counter()

  pos = sim.posMatrix
  vel = sim.velMatrix
  accel = sim.accelMatrix

  # atoms are stationary until (kinetic > potential * X_stable)

  stabilized = False
  X_stable = input_stablization_const

  totalTicks = input_ticks
  scale = input_scale

  rows = int(math.ceil((input_ticks + 1) / input_scale))
  natoms = len(sim.atomArray)

  posHistoryDf = None
  tickHistoryDf = None

  i = 0
  while not stabilized:
    (accel, vel, pos) = sim.update_loop_j(scale, (accel, vel, pos))
    i += 1

    if i % 1 == 0:
      res = sim.record_j(sim.t, pos, vel)

      if res[1][0][1] * X_stable > res[1][0][2]:
        pos = sim.posMatrix
      else:
        stabilized = True

  ranges = getRecordingRanges(totalTicks, 10_000_000, scale)
  pctDenominator = int(totalTicks / (100 * scale))
  lastPct = 0

  for rngIdx in range(0, len(ranges), 2):
    rows = ranges[rngIdx].stop - ranges[rngIdx].start
    currTick = 0
    positionHistoryArr = None
    tickHistoryArray = None
    positionHistoryArrAccum = None
    tickHistoryArrayAccum = None
    accum = 0
    arrBatch = 1000

    for i in ranges[rngIdx]:
      (accel, vel, pos) = sim.update_loop_j(scale, (accel, vel, pos))

      sim.t += sim.dt * time_unit * scale

      if int(i / scale) / pctDenominator > lastPct:
        print("--- %s%% %s seconds ---" % (str(int(100 * (sim.currTick / (totalTicks / scale)))), time.perf_counter() - start_time))
        lastPct = math.ceil(int(i / scale) / pctDenominator)

      res = sim.record_j(sim.t, pos, vel)

      # time (s), position (Å)

      if int(i / scale) % arrBatch == 0:
        if not positionHistoryArr is None:
          if not positionHistoryArrAccum is None:
            positionHistoryArrAccum = np.append(positionHistoryArrAccum, positionHistoryArr, axis = 0)
            tickHistoryArrayAccum = np.append(tickHistoryArrayAccum, tickHistoryArray, axis = 0)
          else:
            positionHistoryArrAccum = positionHistoryArr
            tickHistoryArrayAccum = tickHistoryArray
          accum += arrBatch

        arrRows = min(rows, arrBatch)
        positionHistoryArr = np.empty([arrRows * natoms,5])
        tickHistoryArray = np.empty([arrRows,5])
        rows -= arrBatch

      startIdx = (currTick - accum)

      positionHistoryArr = jax.ops.index_update(
        positionHistoryArr,
        jax.ops.index[(startIdx * natoms):(startIdx * natoms + natoms)],
        res[0])

      # time (ps), energy (zJ), bond lengths (Å)

      tickHistoryArray = jax.ops.index_update(
        tickHistoryArray,
        jax.ops.index[(startIdx): (startIdx + 1)],
        res[1])

      sim.currTick += 1
      currTick += 1

    if not positionHistoryArrAccum is None:
      positionHistoryArrAccum = np.append(positionHistoryArrAccum, positionHistoryArr, axis = 0)
      tickHistoryArrayAccum = np.append(tickHistoryArrayAccum, tickHistoryArray, axis = 0)
    else:
      positionHistoryArrAccum = positionHistoryArr
      tickHistoryArrayAccum = tickHistoryArray

    posHistoryDf = get_df_posHistoryArr(positionHistoryArrAccum, posHistoryDf)
    tickHistoryDf = get_df_tickHistoryArr(tickHistoryArrayAccum, tickHistoryDf)

    #Skip Recordings
    if rngIdx + 1 < len(ranges):
      curRng = ranges[rngIdx + 1]
      firstPercentTick = int(curRng.start / scale) / pctDenominator
      lastPercentTick = int(curRng.stop / scale) / pctDenominator

      step = int(int((curRng.stop - curRng.start) / (lastPercentTick - firstPercentTick)))

      for i in range(curRng.start, curRng.stop, step):
        (accel, vel, pos) = sim.update_loop_j(step, (accel, vel, pos))

        sim.t += sim.dt * time_unit * step

        if int(i / scale) / pctDenominator > lastPct:
          print("--- %s%% %s seconds ---" % (str(int(100 * (sim.currTick / (totalTicks / scale)))), time.perf_counter() - start_time))
          lastPct = math.ceil(int(i / scale) / pctDenominator)
        sim.currTick += step / scale

  print("--- %s seconds ---" % (time.perf_counter() - start_time))

  with open(input_mol + '.csv', mode='w') as molInfo:
    molWriter = csv.writer(molInfo, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    molWriter.writerow([len(molecule)])
    molWriter.writerow([scale])

    atomMap = {}

    for i, (k, v) in enumerate(molecule.items()):
      molWriter.writerow([v['Type']])
      atomMap[k] = i

    for i, (k, v) in enumerate(molecule.items()):
      bondCount = len(v['Neighbors'])
      molWriter.writerow([bondCount])
      for i in v['Neighbors']:
        molWriter.writerow([atomMap[i]])

  with open(input_mol + '_positionHistory.csv', mode='w') as posHistory:
    posHistoryDf.to_csv(posHistory)

  ####################
  # prints csv files #
  ####################

  with open(input_mol + '_energyHistory.csv', mode='w') as energyHistory:
    tickHistoryDf[["time", "potentialE", "kineticE"]].to_csv(energyHistory)

  with open(input_mol + '_bondLengthHistory.csv', mode='w') as bondLengthHistory:
    tickHistoryDf[["time", "CC_Bonds", "CH_Bonds"]].to_csv(bondLengthHistory)

  if len(ranges) == 1:
    ############################
    # prints full energy plot #
    ############################

    draw_energy(tickHistoryDf, input_mol, input_ticks, dt, time_unit, 0, 4, input_mol + '_energyPlot.png')

    #################################
    # prints Q1 of full energy plot #
    #################################

    draw_energy(tickHistoryDf, input_mol, input_ticks, dt, time_unit, 0, 1, input_mol + '_energyPlotQ1.png', " (Q1)")

    #################################
    # prints Q4 of full energy plot #
    #################################

    draw_energy(tickHistoryDf, input_mol, input_ticks, dt, time_unit, 3, 4, input_mol + '_energyPlotQ4.png', " (Q4)")

    ###########################
    # prints bond length plot #
    ###########################

    draw_bond(tickHistoryDf, input_mol, input_ticks, dt, time_unit, 0, 4, input_mol + '_bondLengthPlot.png')

    ################################
    # prints bond length histogram #
    ################################

    draw_bond_histogram(tickHistoryDf, input_mol, input_ticks, dt, time_unit, 0.001, input_mol + '_bondLengthHist.png')

def draw_energy(energyHistory, input_mol, input_ticks, dt, time_unit, q_start, q_end, out_file = None, title_suffix = ""):
    dataLen = energyHistory["time"].count()
    font = {'family' : 'DejaVu Sans',
        'size' : min(max(12, dataLen / 300 * (q_end - q_start) / 4), 24)}

    rng = range(int(q_start * dataLen / 4), int(q_end * dataLen / 4))

    plt.figure(figsize = (min(max(10, dataLen / 100 * (q_end - q_start) / 4), 600), 5))
    plt.rc('font', **font)
    plt.scatter(energyHistory["time"][rng], energyHistory["potentialE"][rng], label = 'Potential', s = 2.5)
    plt.scatter(energyHistory["time"][rng], energyHistory["kineticE"][rng], label = 'Kinetic', s = 2.5)
    plt.scatter(energyHistory["time"][rng], energyHistory["potentialE"][rng] + energyHistory["kineticE"][rng], label = 'Total', s = 2.5)

    plt.title(input_mol.capitalize() + " Energy for " + str(input_ticks) + " Ticks" + title_suffix + ", dt = " + str(dt * time_unit) + "s")
    plt.xlabel('Time (ps)')
    plt.ylabel('Energy (zJ)')
    plt.ylim(bottom=0)
    plt.legend(prop = {'size' : min(max(12, dataLen / 400 * (q_end - q_start) / 4), 24)}, markerscale = 5)
    plt.tight_layout()

    if out_file is None:
        plt.figure()
    else:
        plt.savefig(out_file)

def draw_bond(bondHistory, input_mol, input_ticks, dt, time_unit, q_start, q_end, out_file = None, title_suffix = ""):
    dataLen = bondHistory["time"].count()
    font = {'family' : 'DejaVu Sans',
        'size' : min(max(12, dataLen / 750 * (q_end - q_start) / 4), 24)}

    rng = range(int(q_start * dataLen / 4), int(q_end * dataLen / 4))

    plt.figure(figsize = (min(max(10, dataLen / 400 * (q_end - q_start) / 4), 600), 5))
    plt.rc('font', **font)
    plt.plot([bondHistory["time"][int(q_start * dataLen / 4)], bondHistory["time"][int(q_end * dataLen / 4) - 1]], [1.455, 1.455], color = 'blue', linestyle = ':')
    plt.plot([bondHistory["time"][int(q_start * dataLen / 4)], bondHistory["time"][int(q_end * dataLen / 4) - 1]], [1.099, 1.099], color = 'orange', linestyle = ':')

    plt.scatter(bondHistory["time"][rng], bondHistory["CC_Bonds"][rng], label = 'Average CC Bond Length', s = 2.5)
    plt.scatter(bondHistory["time"][rng], bondHistory["CH_Bonds"][rng], label = 'Average CH Bond Length', s = 2.5)

    plt.title("Average " + input_mol.capitalize() + " Bond Lengths for " + str(input_ticks) + " Ticks" + title_suffix + ", dt = " + str(dt * time_unit) + "s")
    plt.xlabel('Time (ps)')
    plt.ylabel('Bond Length (Å)')
    plt.ylim(bottom=0)
    plt.legend(prop = {'size' : min(max(12, dataLen / 1000 * (q_end - q_start) / 4), 24)}, markerscale = 5)
    plt.tight_layout()

    if out_file is None:
        plt.figure()
    else:
        plt.savefig(out_file)

def draw_bond_histogram(bondHistory, input_mol, input_ticks, dt, time_unit, binWidth, out_file = None):
    font = {'family' : 'DejaVu Sans',
        'size' : 12}

    plt.figure(figsize = (10, 5))
    plt.rc('font', **font)

    cc_bins = np.arange(min(bondHistory["CC_Bonds"]), max(bondHistory["CC_Bonds"]) + binWidth, binWidth)
    cc_hist = plt.hist(bondHistory["CC_Bonds"], bins = cc_bins, label = 'Average CC Bond Length')

    ch_bins = np.arange(min(bondHistory["CH_Bonds"]), max(bondHistory["CH_Bonds"]) + binWidth, binWidth)
    ch_hist = plt.hist(bondHistory["CH_Bonds"], bins = ch_bins, label = 'Average CH Bond Length')

    maxY = max(np.max(cc_hist[0]), np.max(ch_hist[0]))

    plt.plot([1.455, 1.455], [0, maxY * 1.25], color = 'blue', linestyle = ':')
    plt.plot([1.099, 1.099], [0, maxY * 1.25], color = 'orange', linestyle = ':')

    plt.title("Histogram of Average " + input_mol.capitalize() + " Bond Lengths Over Time for " + str(input_ticks) + " Ticks, dt = " + str(dt * time_unit) + "s")
    plt.xlabel('Bond Length (Å)')
    plt.ylabel('Counts')
    plt.legend(prop = {'size' : 12})
    plt.tight_layout()

    if out_file is None:
        plt.figure()
    else:
        plt.savefig(out_file)

def getRecordingRanges(totalTicks, recordingTicks, scale):
  totalTicks = int((totalTicks + scale - 1) / scale) * scale
  recordingTicks = int((recordingTicks + scale - 1) / scale) * scale

  if totalTicks < recordingTicks * 10:
    return [range(0, totalTicks + 1, scale)]

  stop1 = recordingTicks
  stop2 = int((int(totalTicks * .25 - recordingTicks / 2)) / scale) * scale
  stop3 = stop2 + recordingTicks
  stop4 = int((int(totalTicks * .5 - recordingTicks / 2)) / scale) * scale
  stop5 = stop4 + recordingTicks
  stop6 = int((int(totalTicks * .75 - recordingTicks / 2)) / scale) * scale
  stop7 = stop6 + recordingTicks
  stop8 = totalTicks - 2 * recordingTicks
  stop9 = stop8 + recordingTicks

  return [range(0,stop1, scale),
    range(stop1, stop2, scale),
    range(stop2, stop3, scale),
    range(stop3, stop4, scale),
    range(stop4, stop5, scale),
    range(stop5, stop6, scale),
    range(stop6, stop7, scale),
    range(stop7, stop9, scale),
    range(stop8, stop9, scale)]

parser = ap.ArgumentParser(description="Simulate one of following molecules: ethane, propane, isobutane, benzene")
parser.add_argument('molecule', help = "molecule name")
parser.add_argument('--dt', type=float, dest='dt', default = 1e-18, help = "size of timestep (default: 1e-18")
parser.add_argument('--randomize_const', type=float, dest='randomize_const', default = 0.05, help = "amount of randomization in initial positions, 0 for no randomization (default: 0.05)")
parser.add_argument('--iterations', type=int, dest = 'iterations', default= 10_000, help = 'number of iterations (default: 10,000)')
parser.add_argument('--scale', type=int, dest = 'scale', default=100, help = 'create output per scale iteration (default: 100)')
parser.add_argument('--stablization_const', type=float, dest = 'stablization_const', default=0, help = 'molecule will be frozen until kinetic_E>potential_E*stablization_const (default: 0)')

if __name__ == "__main__":
  args = parser.parse_args()
  if args:
    Main(
      args.molecule,
      args.dt,
      args.randomize_const,
      args.iterations,
      args.scale,
      args.stablization_const)