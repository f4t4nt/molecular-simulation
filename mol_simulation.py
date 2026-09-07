import argparse as ap
import csv
import jax.numpy as np
import math as math
import os
import time as time

from constants import time_unit
from history import get_df_posHistoryArr, get_df_tickHistoryArr
from molecules import molecules
from plotting import draw_energy, draw_bond, draw_bond_histogram
from simulation import mol

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

  output_dir = os.path.join('output', input_mol)
  os.makedirs(output_dir, exist_ok = True)
  out_path = lambda name: os.path.join(output_dir, name)

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

      positionHistoryArr = positionHistoryArr.at[
        (startIdx * natoms):(startIdx * natoms + natoms)
      ].set(res[0])

      # time (ps), energy (zJ), bond lengths (Å)

      tickHistoryArray = tickHistoryArray.at[
        (startIdx): (startIdx + 1)
      ].set(res[1])

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

  with open(out_path(input_mol + '.csv'), mode='w') as molInfo:
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

  with open(out_path(input_mol + '_positionHistory.csv'), mode='w') as posHistory:
    posHistoryDf.to_csv(posHistory)

  ####################
  # prints csv files #
  ####################

  with open(out_path(input_mol + '_energyHistory.csv'), mode='w') as energyHistory:
    tickHistoryDf[["time", "potentialE", "kineticE"]].to_csv(energyHistory)

  with open(out_path(input_mol + '_bondLengthHistory.csv'), mode='w') as bondLengthHistory:
    tickHistoryDf[["time", "CC_Bonds", "CH_Bonds"]].to_csv(bondLengthHistory)

  if len(ranges) == 1:
    ############################
    # prints full energy plot #
    ############################

    draw_energy(tickHistoryDf, input_mol, input_ticks, dt, time_unit, 0, 4, out_path(input_mol + '_energyPlot.png'))

    #################################
    # prints Q1 of full energy plot #
    #################################

    draw_energy(tickHistoryDf, input_mol, input_ticks, dt, time_unit, 0, 1, out_path(input_mol + '_energyPlotQ1.png'), " (Q1)")

    #################################
    # prints Q4 of full energy plot #
    #################################

    draw_energy(tickHistoryDf, input_mol, input_ticks, dt, time_unit, 3, 4, out_path(input_mol + '_energyPlotQ4.png'), " (Q4)")

    ###########################
    # prints bond length plot #
    ###########################

    draw_bond(tickHistoryDf, input_mol, input_ticks, dt, time_unit, 0, 4, out_path(input_mol + '_bondLengthPlot.png'))

    ################################
    # prints bond length histogram #
    ################################

    draw_bond_histogram(tickHistoryDf, input_mol, input_ticks, dt, time_unit, 0.001, out_path(input_mol + '_bondLengthHist.png'))

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
