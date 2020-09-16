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

# coordinates retrieved from https://cccbdb.nist.gov

ethane = {
    "C0" : {
      "Type" : atoms.C,
      "Neighbors" : ["C1","H0","H1","H2"],
      "Position" : np.array([0.00, 0.00, 0.7680])
    },
    "C1" : {
      "Type" : atoms.C,
      "Neighbors" : ["C0","H3","H4","H5"],
      "Position" : np.array([0.00, 0.00, -0.7680])
    },
    "H0" : {
      "Type" : atoms.H,
      "Neighbors" : ["C0"],
      "Position" : np.array([-1.0192, 0.00, 1.1573])
    },
    "H1" : {
      "Type" : atoms.H,
      "Neighbors" : ["C0"],
      "Position" : np.array([0.5096, 0.8826, 1.1573])
    },
    "H2" : {
      "Type" : atoms.H,
      "Neighbors" : ["C0"],
      "Position" : np.array([0.5096, -0.8826, 1.1573])
    },
    "H3" : {
      "Type" : atoms.H,
      "Neighbors" : ["C1"],
      "Position" : np.array([1.0192, 0.00, -1.1573])
    },
    "H4" : {
      "Type" : atoms.H,
      "Neighbors" : ["C1"],
      "Position" : np.array([-0.5096, -0.8826, -1.1573])
    },
    "H5" : {
      "Type" : atoms.H,
      "Neighbors" : ["C1"],
      "Position" : np.array([-0.5096, 0.8826, -1.1573])
    }
  }



# total # of update calls

time_unit = 1e-12
dist_unit = 1e-10
mass_unit = 1e-20
molecule = ethane

# are we using jit/vmap?
jit_funcs = True
vmap_funcs = True

# timestep, how often are we recording?

dt = 1e-18 / time_unit

# are we recording position/energy/bondLengths?

class mol:
  def __init__ (self, atoms, dt):
    self.atoms = atoms
    self.dt = dt

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

    self.A2m = 1e-10 / dist_unit
    self.amu2kg = 1.660539e-27 / mass_unit
    self.kcal2J = 4186.4
    self.kcal2MU = self.kcal2J * (1 / mass_unit) * (time_unit / dist_unit) ** 2

    # Å
    # rad
    self.X_cc = 1.455 * self.A2m
    self.X_ch = 1.099 * self.A2m
    self.X_ccc = 1.937
    self.X_hch = 1.911
    self.X_cch = 1.911

    # distance-energy conversion constants
    # angle-energy conversion constants

    self.distEnergyK_cc = self.K_cc * self.kcal2MU / (self.N * self.A2m ** 2)
    self.distEnergyK_ch = self.K_ch * self.kcal2MU / (self.N * self.A2m ** 2)
    self.angleEnergyK_ccc = self.K_ccc * self.kcal2MU / self.N
    self.angleEnergyK_hch = self.K_hch * self.kcal2MU / self.N
    self.angleEnergyK_cch = self.K_cch * self.kcal2MU / self.N
    self.angleEnergyK_ccTorsional = self.K_ccTorsional * self.kcal2MU / self.N
    
    self.initAtomArrays()
    self.initPairs()
    self.initTriples()
    self.initQuads()
    self.initMatrices()
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

    if len(self.ccPairs) == 0:
      self.ccPairs = np.full((0, 2), 0)

    if len(self.chPairs) == 0:
      self.chPairs = np.full((0, 2), 0)

    self.atomPairs = np.concatenate((self.ccPairs, self.chPairs), axis = 0)
    self.pairEnergyConstants = np.concatenate((
      np.full((1, len(self.ccPairs)), self.distEnergyK_cc),
      np.full((1, len(self.chPairs)), self.distEnergyK_ch)),
      axis = 1)
      
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

    if len(self.cccTriples) == 0:
      self.cccTriples = np.full((0, 3), 0)

    if len(self.hchTriples) == 0:
      self.hchTriples = np.full((0, 3), 0)

    if len(self.cchTriples) == 0:
      self.cchTriples = np.full((0, 3), 0)

    self.atomTriples = np.concatenate((self.cccTriples, self.hchTriples, self.cchTriples), axis = 0)
    self.triplesAngleEneryConstants = np.concatenate((
      np.full((1, len(self.cccTriples)), self.angleEnergyK_ccc),
      np.full((1, len(self.hchTriples)), self.angleEnergyK_hch),
      np.full((1, len(self.cchTriples)), self.angleEnergyK_cch)),
      axis = 1)

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
    self.posMatrix = np.array([([pos * self.A2m for pos in atom[1]["Position"]]) for atom in self.atomArray])
    # angstroms/second
    self.velMatrix = np.zeros((len(self.atomArray), 3))
    # angstroms/second^2
    self.accelMatrix = np.zeros((len(self.atomArray), 3))
    self.prevAccelMatrix = np.zeros((len(self.atomArray), 3))
    # newtons
    self.forceMatrix = np.zeros((len(self.atomArray), 3))
    # atomc masses
    self.massMatrix = np.array([atom[1]["Type"].value * self.amu2kg for atom in self.atomArray])
    # joules
    self.potential = 0
    # seconds
    self.t = 0
    # tick index
    self.currTick = 0
    
    self.M_cc = np.zeros((len(self.ccPairs), 1)) + self.X_cc
    self.M_ch = np.zeros((len(self.chPairs), 1)) + self.X_ch
    self.M_ccc = np.zeros((len(self.cccTriples), 1)) + self.X_ccc
    self.M_hch = np.zeros((len(self.hchTriples), 1)) + self.X_hch
    self.M_cch = np.zeros((len(self.cchTriples), 1)) + self.X_cch

    self.M_pairs = np.concatenate((self.M_cc, self.M_ch), axis = 0).squeeze()
    self.M_triples = np.concatenate((self.M_ccc, self.M_hch, self.M_cch), axis = 0).squeeze()

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
    global jit_funcs
    jit_funcs = False
    self.kineticAtom_v = self.vmap(self.kineticAtom_, in_axes = (0, 0))
    self.distance_v = self.jit(self.vmap(self.distance_, in_axes = (0, )))
    self.angle_v = self.jit(self.vmap(self.angle_, in_axes = (0, )))
    self.cosTorsionalAngle_v = self.jit(self.vmap(self.cosTorsionalAngle_(True), in_axes = (0, )))
    self.accelAtom_v = self.jit(self.vmap(self.accelAtom_, in_axes = (0, 0)))
    self.posAtom_v = self.jit(self.vmap(self.posAtom_, in_axes = (0, 0)))
    self.velAtom_v = self.jit(self.vmap(self.velAtom_, in_axes = (0, 0)))
    self.updatePosition_v = self.jit(self.vmap(self.updatePosition, in_axes = (0, 0, 0, 0, 0)))
    # self.calcPotential_j = self.jit(self.calcPotential)
    self.getCalcPotential_ = self.getCalcPotential(False)
    self.gradient_j = self.jit(jax.grad(self.getCalcPotential(False)))

    self.calcPotential_j = self.jit(self.getCalcPotential(False))
    self.calcKinetic_j = self.jit(self.calcKinetic_)
    self.calcEnergy_j = self.jit(self.calcEnergy_)

    jit_funcs = True
    self.update_j = self.jit(self.update)
    self.record_j = self.jit(self.record)

  # calculates length AB given positions

  def distance_(self, P):
    p0 = P[...,[0],[0,1,2]]
    p1 = P[...,[1],[0,1,2]]

    r = p0 - p1
    r_mag = np.sqrt(np.sum(np.square(r)))
    return r_mag

  # calculates cosine of angle ABC given positions

  def cosAngle_(self, P):
    p0 = P[...,[0],[0,1,2]]
    p1 = P[...,[1],[0,1,2]]
    p2 = P[...,[2],[0,1,2]]

    r1 = p0 - p1
    r2 = p2 - p1
    dot = np.sum(np.multiply(r1, r2), axis = 1)
    r1_mag = np.sqrt(np.sum(np.square(r1), axis = 1))
    r2_mag = np.sqrt(np.sum(np.square(r2), axis = 1))
    return dot / (r1_mag * r2_mag)

  # calculates angle ABC given positions

  def angle_(self, P):
    return np.arccos(self.cosAngle_(P))

  # calculates torsional angle ABCD given positions

  def cosTorsionalAngle_(self, use_v):
    cosAngle_ = self.cosAngle_
    transposeShape = [1, 0] if use_v else [0, 2, 1]

    def internal(P):
      p0 = P[...,[0],[0,1,2]]
      p1 = P[...,[1],[0,1,2]]
      p2 = P[...,[2],[0,1,2]]
      p3 = P[...,[3],[0,1,2]]

      r1 = p0 - p1
      r2 = p1 - p2
      r3 = p3 - p2
      cp_12 = np.cross(r1, r2)
      cp_32 = np.cross(r3, r2)
      cp = np.dstack((cp_12, np.zeros(cp_12.shape), cp_32)) \
        .squeeze() \
        .transpose(transposeShape)
      cosTorsionalAngle = cosAngle_(cp)
      return cosTorsionalAngle

    return internal

  # calculates potential energy of molecule given positions

  def getCalcPotential(self, use_v):
    atomPairs = self.atomPairs
    pairEnergyConstants = self.pairEnergyConstants
    atomTriples = self.atomTriples
    triplesAngleEneryConstants = self.triplesAngleEneryConstants
    quads = self.quads
    angleEnergyK_ccTorsional = self.angleEnergyK_ccTorsional
    distance_v = self.distance_v
    M_pairs = self.M_pairs
    M_triples = self.M_triples

    if use_v:
      angle_v = self.angle_v
      cosTorsionalAngle_v = self.cosTorsionalAngle_v
    else:
      angle_v = self.angle_
      cosTorsionalAngle_v = self.cosTorsionalAngle_(False)

    def calcPotential_(pos):
      potential_0 = 0.5 * np.sum(
        np.multiply(
          np.square(distance_v(pos[atomPairs]) - M_pairs),
          pairEnergyConstants))

      potential_0 += 0.5 * np.sum(
        np.multiply(
          np.square(angle_v(pos[atomTriples]) - M_triples),
          triplesAngleEneryConstants))

      cosAngle = cosTorsionalAngle_v(pos[quads])
      potential_0 += 0.5 \
        * np.sum(1 + 4 * cosAngle ** 3 - 3 * cosAngle) \
        * angleEnergyK_ccTorsional

      return potential_0
    return calcPotential_

  # calculates kinetic energy of single atom given mass and velocity

  def kineticAtom_(self, M, V):
    return 0.5 * np.multiply(M, np.sum(np.square(V))) * self.A2m ** 2 # * self.amu2kg

  # calculates kinetic energy of molecule given masses and velocities

  def calcKinetic(self, V):
    return np.sum(self.kineticAtom_v(self.massMatrix, V))

  def calcKinetic_(self, V):
    sq = np.square(V).transpose()
    sq_sum = np.sum(sq, axis = 0)
    mv2 = np.sum(np.multiply(self.massMatrix, sq_sum), axis = 0)
    return mv2 * 0.5 * (self.A2m ** 2) # * self.amu2kg

  def calcEnergy_(self, P, V):
    return (self.getCalcPotential_(P), self.calcKinetic_(V))

  # calculates force matrix

  def calcForce(self, P):
    return -1 * self.gradient_j(P) / self.A2m

  # calculates acceleration matrix of single atom given mass and force

  def accelAtom_(self, M, F):
    return F / M # / self.amu2kg

  # calculates acceleration matrix gives masses and forces

  def calcAccel(self, M, F):
    return self.accelAtom_v(M, F)

  # calculates position matrix of single atom given position and velocity

  def posAtom_(self, P, V, A):
    return P + V * self.dt + A * self.dt ** 2 / 2

  # calculates position matrix given positions and velocities

  def calcPos(self, P, V):
    return self.posAtom_v(P, V)

  # calculates velocity matrix of single atom given velocity and acceleration

  def velAtom_(self, V, A):
    return V + A * self.dt

  # calculates velocity matrix given velocities and accelerations

  def calcVel(self, V, A):
    return self.velAtom_v(V, A)

  def updatePosition(self, P, V, A, pA, dt):
    # using dA improves our speed by 30X
    # we get same amount of error in total energy after 10,000 ticks with dt=3e-18 and without dA as with dt=1e-16 and with dA
    dA = A - pA
    P = P + V * dt + A * (dt * dt / 2) + dA * (dt * dt / 3)
    V = V + A * dt + dA * (dt / 2)
    return (P, V)

  def update(self, accel, vel, pos):
    prevAccel = accel
    forceMatrix = self.calcForce(pos)
    accel = self.calcAccel(self.massMatrix, forceMatrix)
    (pos, vel) = self.updatePosition(pos, vel, accel, prevAccel, self.dt)
    return (accel, vel, pos)

  def print(self):
    # self.potential = self.calcPotential_j(self.posMatrix)
    # kineticE = self.calcKinetic(self.velMatrix)
    # print("t:")
    # print(self.t)
    # print("potential:")
    # print(self.potential)
    # print("kinetic:")
    # print(kineticE)
    # print("total:")
    # print(self.potential + kineticE)
    # print()
    # print("posMatrix:")
    # print(self.posMatrix)
    # print("velMatrix:")
    # print(self.velMatrix)
    # print("accelMatrix")
    # print(self.accelMatrix)

    # print()

    # print(str(int(100 * (self.currTick / (totalTicks / scale)))) + "%")
    print("--- %s%% %s seconds ---" % (str(int(100 * (self.currTick / (totalTicks / scale)))), time.perf_counter() - start_time))

  def record(self, t, pos, vel):
    atoms = len(self.atomArray)
    tM = np.full((atoms, 1), t)
    idx = np.array([range(atoms)]).transpose()

    arr = np.concatenate((tM, idx, pos), axis = 1)

    (potential, kinetic) = self.calcEnergy_j(pos, vel)
    return (arr,
      np.array([[
        t,
        potential,
        kinetic,
        np.average(self.distance_v(pos[self.ccPairs])),
        np.average(self.distance_v(pos[self.chPairs]))]]))


print("--- 0 seconds ---")

sim = mol(molecule, dt)

start_time = time.perf_counter()

pos = sim.posMatrix
vel = sim.velMatrix
accel = sim.accelMatrix
stabilized = False

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
    positionHistoryArr = jax.ops.index_update(
      positionHistoryArr,
      jax.ops.index[(sim.currTick * natoms):(sim.currTick * natoms + natoms)],
      res[0])
    tickHistoryArray = jax.ops.index_update(
      tickHistoryArray,
      jax.ops.index[sim.currTick: sim.currTick + 1],
      res[1])
    # positionHistoryArr = np.concatenate((positionHistoryArr, res[0]))
    # tickHistoryArray = np.concatenate((tickHistoryArray, np.array((res[1]))))

    # if stabilized == False and res[1][0][1] * 1e-9 > res[1][0][2]:
    #   pos = sim.posMatrix
    # else:
    #   stabilized = True

    sim.currTick += 1


print("--- %s seconds ---" % (time.perf_counter() - start_time))

with open('positionHistory.csv', mode='w') as posHistory:
  posWriter = csv.writer(posHistory, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  posWriter.writerow([len(molecule)])
  posWriter.writerow([scale])
  for i, (k, v) in enumerate(molecule.items()):
    posWriter.writerow([v['Type']])

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

plt.figure(figsize = (20, 5))
plt.scatter(tickHistDf["time"], tickHistDf["potentialE"], label = 'Potential')
plt.scatter(tickHistDf["time"], tickHistDf["kineticE"], label = 'Kinetic')
plt.scatter(tickHistDf["time"], tickHistDf["potentialE"] + tickHistDf["kineticE"], label = 'Total')

plt.title("Ethane Energy Over Time for " + str(totalTicks) + " Ticks, dt = " + str(dt * time_unit) + "s")
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.legend()
plt.savefig('energyPlot.png')

plt.figure(figsize = (10, 5))
plt.plot([tickHistDf["time"][0], totalTicks * dt * time_unit], [1.455, 1.455], color = 'blue', linestyle = ':')
plt.plot([tickHistDf["time"][0], totalTicks * dt * time_unit], [1.099, 1.099], color = 'orange', linestyle = ':')

plt.scatter(tickHistDf["time"], tickHistDf["CC_Bonds"], label = 'Average CC Bond Length')
plt.scatter(tickHistDf["time"], tickHistDf["CH_Bonds"], label = 'Average CH Bond Length')

plt.title("Average Ethane Bond Lengths Over Time for " + str(totalTicks) + " Ticks, dt = " + str(dt * time_unit) + "s")
plt.xlabel('Time (s)')
plt.ylabel('Bond Length (Å)')
plt.legend()
plt.savefig('bondLengthPlot.png')