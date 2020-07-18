
class atoms(Enum):
  C = 0
  H = 1

ethane = {
    "C1" : {
            "type"      : atoms.C,
            "position"  : [0,0,0],
            "neighbors" : ["H0", "H1", "H2", "C2"],
            #"distances": [1, 2, 3,],
            #"angles"  : [np.pi/2, np.pi, np.pi ....]
            }
    }

lookup = {}
all_positions = []

for i,k,v in enumerate(ethane.iteritems()):
  lookup[k] = i
  all_positions.append(np.array(ethane[k]["position"]))

all_positions = np.array(all_positions)

lookup_k_all = {}
for k,v in ethane.iteritems():
  for atm_n in k["neighbors"]:
    if v.type.name + ethane[atm_n].type.name not in lookup_k_all:
      lookup_k_all[v.type.name + ethane[atm_n].type.name] = len(self.K_all)
      self.K_all.append(coeffs[v.type.name, ethane[atm_n]])

nn_indices = [[] for i in range(len(self.K_all))]
for k,v in ethane.iteritems():
  for atm_n in k["neighbors"]:
    ind_k = lookup_k_all[v.type.name + ethane[atm_n].type.name] # CH vs HC
    nn_indices[ind_k].append(np.array([lookup[k], lookup[atm_n]])

           
            
            
all_cc_dists = dist_matrix[cc_indices]
V_cc = K_cc*np.sum(np.square(all_cc_dists))

def distances_by_nn_type(R):
  V_nn = 0
  for k,inds in zip(self.K_all, nn_indices):
    V_nn += k*np.sum(np.square(all_cc_dists))
    

  

def calc_nn(R):
  return self.K_all

def V(R):
  V = 0
  V += calc_nn(R)
  return V

F_fxn = -1*jax.grad(V)
F = F_fxn(R)

R = V*dt + 0.5*F*dt**2/m




def main():
  md = model(ethane)

  for tm in times:
    md.update()
