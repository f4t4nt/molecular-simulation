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
