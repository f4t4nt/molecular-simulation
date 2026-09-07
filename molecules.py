from enum import Enum
import jax as jax
import jax.numpy as np

# jax_enable_x64 must be set before any JAX array is created, including the
# molecule position arrays defined below.
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
