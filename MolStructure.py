import numpy as np
import pickle as pl
from enum import Enum

def VecRotationMatrix(vect, angle, isRad = False):
    angle = angle if isRad else angle * np.pi / 180
    mag = np.linalg.norm(vect)
    l = vect[0, 0] / mag 
    m = vect[0, 1] / mag
    n = vect[0, 2] / mag
    cos = np.cos(angle)
    omc = 1 - cos
    sin = np.sin(angle)
    return np.matrix([
        [l * l * omc + cos, m * l * omc - n * sin, n * l * omc + m * sin, 0],
        [l * m * omc + n * sin, m * m * omc + cos, m * n * omc - l * sin, 0],
        [l * n * omc - m * sin, m * n * omc + l * sin, n * n * omc + cos, 0],
        [0, 0, 0, 1]])

def VecTranslationMatrix(vect):
    return np.matrix([
        [1, 0, 0, vect[0, 0]],
        [0, 1, 0, vect[0, 1]],
        [0, 0, 1, vect[0, 2]],
        [0, 0, 0, 1]])

def SphericalRotationMatrix(theta, phi, isRad = False):
    thetaRotationMatrix = VecRotationMatrix(np.matrix([0, 1, 0]), theta, isRad)
    phiRotationMatrix = VecRotationMatrix( np.matrix([1, 0, 0]), phi, isRad)

    return phiRotationMatrix * thetaRotationMatrix

origin = np.matrix([[0], [0], [0], [1]])
xunit = np.matrix([[1], [0], [0], [1]])
yunit = np.matrix([[0], [1], [0], [1]])
zunit = np.matrix([[0], [0], [1], [1]])
mxunit = np.matrix([[-1], [0], [0], [1]])
myunit = np.matrix([[0], [-1], [0], [1]])
mzunit = np.matrix([[0], [0], [-1], [1]])

class atoms(Enum):
  C = 0
  H = 1

ethane = {
    # "Info" : {
    #     "Atoms" : [
    #         "C0", 
    #         "C1",
    #         "H0",
    #         "H1",
    #         "H2",
    #         "H3",
    #         "H4",
    #         "H5"
    #     ]
    # },
    "C0" : {
        "Type" : atoms.C,
        "Mass" : 10,
        "NeighArray" : [
            "H0",
            "H1",
            "H2",
            "C1"
        ],
        "NeighEnum" : {
            "H0" : {
                "d": 1.9,
                "Theta": 109,
                "Phi" : 0,
            },
            "H1" : {
                "d": 1.9,
                "Theta": 109,
                "Phi" : 120,
            },
            "H2" : {
                "d": 1.9,
                "Theta": 109,
                "Phi" : -120,
            },
            "C1" : {
                "d": 1.9,
                "Theta": 0,
                "Phi" : 0,
            }
        }
    },
    "C1" : {
        "Type" : atoms.C,
        "Mass" : 10,
        "NeighArray" : [
            "H3",
            "H4",
            "H5",
            "C1"
        ],
        "NeighEnum" : {
            "H3" : {
                "d": 1.9,
                "Theta": 180 - 109,
                "Phi" : 180,
            },
            "H4" : {
                "d": 1.9,
                "Theta": 180 - 109,
                "Phi" : 60,
            },
            "H5" : {
                "d": 1.9,
                "Theta": 180 - 109,
                "Phi" : -60,
            },
            "C0" : {
                "d": 1.9,
                "Theta": 180,
                "Phi" : 0,
            }
        }
    },
    "H0" : {
        "Type" : atoms.H,
        "Mass" : 1,
        "NeighArray" : [
            "C0"
        ],
        "NeighEnum" : {
            "C0": {
                "d": 1.9,
                "Theta": 0,
                "Phi": 0
            }
        }
    },
    "H1" : {
        "Type" : atoms.H,
        "Mass" : 1,
        "NeighArray" : [
            "C0"
        ],
        "NeighEnum" : {
            "C0": {
                "d": 1.9,
                "Theta": 0,
                "Phi": 0
            }
        }
    },
    "H2" : {
        "Type" : atoms.H,
        "Mass" : 1,
        "NeighArray" : [
            "C0"
        ],
        "NeighEnum" : {
            "C0": {
                "d": 1.9,
                "Theta": 0,
                "Phi": 0
            }
        }
    },
    "H3" : {
        "Type" : atoms.H,
        "Mass" : 1,
        "NeighArray" : [
            "C1"
        ],
        "NeighEnum" : {
            "C1": {
                "d": 1.9,
                "Theta": 0,
                "Phi": 0
            }
        }
    },
    "H4" : {
        "Type" : atoms.H,
        "Mass" : 1,
        "NeighArray" : [
            "C1"
        ],
        "NeighEnum" : {
            "C1": {
                "d": 1.9,
                "Theta": 0,
                "Phi": 0
            }
        }
    },
    "H5" : {
        "Type" : atoms.H,
        "Mass" : 1,
        "NeighArray" : [
            "C1"
        ],
        "NeighEnum" : {
            "C1": {
                "d": 1.9,
                "Theta": 0,
                "Phi": 0
            }
        }
    }
}

simMol = {
    "C0" : {
        "NeighEnum": {
            "C1": {
                "d": 1,
                "Theta": 0,
                "Phi": 0
            }
        }
    },
    "C1" : {
        "NeighEnum": {
            "C0": {
                "d": 1,
                "Theta": 0,
                "Phi": 0
            },
            "C2": {
                "d" : 1,
                "Theta": 90,
                "Phi": 0
            }
        }
    },
    "C2" : {
        "NeighEnum": {
            "C1": {
                "d": 1,
                "Theta": 0,
                "Phi": 0
            },
            "C3": {
                "d": 1,
                "Theta": -90,
                "Phi": 0
            }
        }
    },
    "C3" : {
        "NeighEnum": {
            "C2": {
                "d": 1,
                "Theta": -90,
                "Phi": 0
            }
        }
    },
}

duoMol = {
    "C0" : {
        "NeighEnum": {
            "C1": {
                "d": 1,
                "Theta": 90,
                "Phi": 90
            }
        }
    },
    "C1" : {
        "NeighEnum": {
            "C0": {
                "d": 1,
                "Theta": 0,
                "Phi": 0
            },
        }
    },
}

# Initializes default position of all atoms relative to themselves

def createStruct(mol):
    atoms = [k for k, v in mol.items()]

    posArray = [None for atom in atoms]
    posVisited = [False for atom in atoms]
    posDict = { atoms[i]: i for i in range(len(atoms)) }
    
    recCreateStruct(mol, posDict, posArray, posVisited, "C0")

    origin = np.matrix([[0], [0], [0], [1]])
    for idx in range(len(atoms)):
        transform = posArray[idx]
        print(atoms[idx])
        pos = transform * origin
        print("( %3.2f, %3.2f, %3.2f)" %(pos[0, 0], pos[1, 0], pos[2, 0]))
        print()

def recCreateStruct(mol, posDict, posArray, posVisited, currAtom):
    curIdx = posDict[currAtom]
    posArray[curIdx] = np.matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    posVisited[curIdx] = True

    rv = []

    for atom, info in mol[currAtom]["NeighEnum"].items():
        distance = info["d"]
        theta = info["Theta"]
        phi = info["Phi"]
        neighIdx = posDict[atom]  
        reverseInfo = mol[atom]["NeighEnum"][currAtom]
        if distance != reverseInfo["d"]:
            Exception("Bad data")

        if not posVisited[neighIdx]:
            posVisited[neighIdx] = True

            children = recCreateStruct(mol, posDict, posArray, posVisited, atom)

            # Align to parent's axis
            translationMatrix = VecTranslationMatrix(np.matrix([distance, 0, 0]))
            backRotatinMatrix = SphericalRotationMatrix(reverseInfo["Theta"], reverseInfo["Phi"])

            rotationMatrix = SphericalRotationMatrix(theta, phi)
            finalTransform = np.linalg.inv(backRotatinMatrix) * rotationMatrix * backRotatinMatrix * translationMatrix

            # Move to position based on atom's directions towards currAtom
            children.append(atom)
            for child in children:
                childIdx = posDict[child]
                posArray[childIdx] = finalTransform * posArray[childIdx]

            rv += children
    return rv

createStruct(ethane)
print("Done")