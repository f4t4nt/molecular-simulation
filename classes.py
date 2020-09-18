from collections import namedtuple
import csv
import math as math
from enum import Enum
import jax as jax
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import pickle as pl
import time as time
import pandas as pd

jax.config.update('jax_enable_x64', True)

# masses in amu
class atoms(Enum):
  C = 12.0107
  H = 1.00784

# coordinates retrieved from https://cccbdb.nist.gov Experimental >> Geometry >> Experimental Geometries

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

ethane_modified = {
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
      "Position" : np.array([0, 1e-3, 1.859])
    },
    "H4" : {
      "Type" : atoms.H,
      "Neighbors" : ["C1"],
      "Position" : np.array([0.866e-3, -0.5e-3, 1.859])
    },
    "H5" : {
      "Type" : atoms.H,
      "Neighbors" : ["C1"],
      "Position" : np.array([-0.866e-3, -0.5e-3, 1.859])
    },
    "H6" : {
      "Type" : atoms.H,
      "Neighbors" : ["C2"],
      "Position" : np.array([0, 1e-3, -1.859])
    },
    "H7" : {
      "Type" : atoms.H,
      "Neighbors" : ["C2"],
      "Position" : np.array([0.866e-3, -0.5e-3, -1.859])
    },
    "H8" : {
      "Type" : atoms.H,
      "Neighbors" : ["C2"],
      "Position" : np.array([-0.866e-3, -0.5e-3, -1.859])
    }
  }

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
    "Position" : np.array([1.3272, -0.9014, 0.88])
  }
}

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

isobutane_modified = {
  "C1" : {
    "Type" : atoms.C,
    "Neighbors" : ["H2", "C3", "C4", "C5"],
    "Position" : np.array([1, 0, 0.365])
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
    "Position" : np.array([0, -2.418, 0])
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

################################################################################################################################################################################
# 1.455Å according to https://dpl6hyzg28thp.cloudfront.net/media/Lifson_and_Warshel_-_1968_-_Consistent_Force_Field_for_Calculations_of_Conform.pdf but 1.54Å in other sources #
################################################################################################################################################################################

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

# timestep

dt = 1e-18 / time_unit

class mol:
  def __init__ (self, atoms, dt):
    self.atoms = atoms
    self.dt = dt

    self.initAtomArrays()
    self.initPairs()
    self.initTriples()
    self.initQuads()
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

  #############################
  # creates variable matrices #
  #############################

  def initMatrices(self):
    # angstroms
    self.posMatrix = np.array([([pos * A2m for pos in atom[1]["Position"]]) for atom in self.atomArray])
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
      r_mag = np.sqrt(np.sum(np.square(r)))
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

  def print(self):
    print("--- %s%% %s seconds ---" % (str(int(100 * (self.currTick / (totalTicks / scale)))), time.perf_counter() - start_time))

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
          potential * mass_unit * (time_unit / dist_unit) ** 2 * 1e-15,
          kinetic * mass_unit * (time_unit / dist_unit) ** 2 * 1e-15,
          np.average(distance(pos[ccPairs])),
          np.average(distance(pos[chPairs]))]]))

    return internal

molecule = isobutane 

print("--- 0 seconds ---")

sim = mol(molecule, dt)

start_time = time.perf_counter()

pos = sim.posMatrix
vel = sim.velMatrix
accel = sim.accelMatrix

# atoms are stationary until (kinetic > potential * X_stable)

stabilized = False
X_stable = 0

totalTicks = 1_000_000
scale = 100

rows = int(math.ceil((totalTicks + 1) / scale))
natoms = len(sim.atomArray)
positionHistoryArr = np.empty([rows * natoms,5])
tickHistoryArray = np.empty([rows,5])

for i in range(totalTicks + 1):
  (accel, vel, pos) = sim.update_j(
    accel,
    vel,
    pos)

  sim.t += sim.dt * time_unit

  if i % int(totalTicks / 10) == 0:
    sim.print()

  if i % scale == 0:
    res = sim.record_j(sim.t, pos, vel)

    # time (s), position (Å)

    positionHistoryArr = jax.ops.index_update(
      positionHistoryArr,
      jax.ops.index[(sim.currTick * natoms):(sim.currTick * natoms + natoms)],
      res[0])

    # time (ps), energy (fJ), bond lengths (Å)

    tickHistoryArray = jax.ops.index_update(
      tickHistoryArray,
      jax.ops.index[sim.currTick: sim.currTick + 1],
      res[1])

    sim.currTick += 1

  if not stabilized:
    res = sim.record_j(sim.t, pos, vel)

    if res[1][0][1] * X_stable > res[1][0][2]:
      pos = sim.posMatrix
    else:
      stabilized = True

print("--- %s seconds ---" % (time.perf_counter() - start_time))

with open('moleculeInformation.csv', mode='w') as molInfo:
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

with open('positionHistory.csv', mode='w') as posHistory:
  df = pd.DataFrame(data = positionHistoryArr, columns=["time", "atomId", "posX", "posY", "posZ"]) \
    .astype({"atomId": "int16"})

  df.to_csv(posHistory)

tickHistDf = pd.DataFrame(
  data = tickHistoryArray,
  columns = ["time", "potentialE", "kineticE", "CC_Bonds", "CH_Bonds"])

with open('energyHistory.csv', mode='w') as energyHistory:
  energyWriter = csv.writer(energyHistory, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  tickHistDf[["time", "potentialE", "kineticE"]].to_csv(energyHistory)

with open('bondLengthHistory.csv', mode='w') as bondLengthHistory:
  bondLengthWriter = csv.writer(bondLengthHistory, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  tickHistDf[["time", "CC_Bonds", "CH_Bonds"]].to_csv(bondLengthHistory)

plt.figure(figsize = (tickHistDf["time"].count() / 50, 5))
plt.scatter(tickHistDf["time"], tickHistDf["potentialE"], label = 'Potential')
plt.scatter(tickHistDf["time"], tickHistDf["kineticE"], label = 'Kinetic')
plt.scatter(tickHistDf["time"], tickHistDf["potentialE"] + tickHistDf["kineticE"], label = 'Total')

plt.title("Modified Ethane Energy Over Time for " + str(totalTicks) + " Ticks, dt = " + str(dt * time_unit) + "s")
plt.xlabel('Time (ps)')
plt.ylabel('Energy (fJ)')
plt.legend()
plt.savefig('energyPlot.png')

plt.figure(figsize = (tickHistDf["time"].count() / 50, 5))
plt.plot([tickHistDf["time"][0], totalTicks * dt * 1e12], [1.455, 1.455], color = 'blue', linestyle = ':')
plt.plot([tickHistDf["time"][0], totalTicks * dt * 1e12], [1.099, 1.099], color = 'orange', linestyle = ':')

plt.scatter(tickHistDf["time"], tickHistDf["CC_Bonds"], label = 'Average CC Bond Length')
plt.scatter(tickHistDf["time"], tickHistDf["CH_Bonds"], label = 'Average CH Bond Length')

plt.title("Average Modified Ethane Bond Lengths Over Time for " + str(totalTicks) + " Ticks, dt = " + str(dt * time_unit) + "s")
plt.xlabel('Time (ps)')
plt.ylabel('Bond Length (Å)')
plt.legend()
plt.savefig('bondLengthPlot.png')