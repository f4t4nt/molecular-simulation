from collections import namedtuple
import csv
import time as time
import vpython as vp
import numpy as np

atoms = []
bonds = []
bondInfo = []
atomIdx = 0
bondIdx = 0
totalAtoms = 0
totalBonds = 0

fps = 60000
scale = 0
tickTime = 0
scene = vp.canvas(background = vp.vec(0, 0, 0), width = 2000, height = 1000)

readPhase = 0

with open('moleculeInformation.csv', newline = '') as molInfo:
    molReader = csv.reader(molInfo, delimiter = ',', quotechar = '"')

    for row in molReader:
        if readPhase == 0:
            totalAtoms = int(row[0])
            readPhase += 1
        elif readPhase == 1:
            scale = int(row[0])
            tickTime = scale / fps
            readPhase += 1
        elif readPhase == 2:
            atom = vp.sphere()
            atom.pos = vp.vector(0, 0, 0)

            # https://sciencenotes.org/wp-content/uploads/2019/07/CPK-Jmol.png
            # https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
            if row[0] == "atoms.H":
                atom.color = vp.vec(1, 1, 1)
                atom.radius = 0.125
            elif row[0] == "atoms.C":
                atom.color = vp.vec(0.565, 0.565, 0.565)
                atom.radius = 0.35
            
            atoms.append(atom)
            atomIdx += 1

            if atomIdx == totalAtoms:
                readPhase += 1
                atomIdx = 0
        elif readPhase == 3 or bondIdx == totalBonds:
            totalBonds = int(row[0])
            readPhase += 1
        elif readPhase == 4:
            bond = vp.cylinder()
            bond.pos = vp.vector(0, 0, 0)
            bond.radius = 0.05
            bond.axis = vp.vector(1, 0, 0)
            bond.length = 1
            bond.color = atoms[atomIdx].color

            bond.atom1 = atomIdx
            bond.atom2 = int(row[0])

            bondInfo.append([atomIdx, int(row[0])])

            bonds.append(bond)

            bondIdx += 1

            if bondIdx == totalBonds:
                atomIdx += 1
                bondIdx = 0
                readPhase -= 1

contentLine = False

with open('positionHistory.csv', newline = '') as posHistory:
    posReader = csv.reader(posHistory, delimiter = ',', quotechar = '"')

    for row in posReader:
        if contentLine:
            atomIdx = int(row[2])
            x = float(row[3])
            y = float(row[4])
            z = float(row[5])

            atoms[atomIdx].pos = vp.vector(x, y, z)

            if atomIdx == totalAtoms - 1:
                time.sleep(tickTime)

                for bond in bonds:
                    atom1 = atoms[bond.atom1]
                    atom2 = atoms[bond.atom2]

                    bond.pos = atom1.pos
                    bond.axis = atom2.pos - atom1.pos
                    bond.length = (vp.mag(atom2.pos - atom1.pos) - atom2.radius + atom1.radius) / 2
        else:
            contentLine = True