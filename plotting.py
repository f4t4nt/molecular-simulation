import matplotlib.pyplot as plt
import numpy as np

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
