# scaled units to prevent overflow

time_unit = 1e-12
dist_unit = 1e-10
mass_unit = 1e-20

# kcal/Å^2
# kcal/rad^2

# All bond/angle/torsion force constants (alkane and aromatic alike) are
# sourced from a single force field, AMBER GAFF (gaff.dat), rather than
# mixing sources, so every term below is on the same footing. Doubled from
# GAFF's E=K(r-r0)^2 / E=K(theta-theta0)^2 bond/angle convention to match
# this file's E=0.5*K(...)^2 convention.

# sp3 alkane carbons: gaff.dat atom types c3 (sp3 C), hc (H on sp3 C).
# c3-c3 bond: K=303.1, r0=1.5350
K_cc = 606.2
# c3-hc bond: K=337.3, r0=1.0920
K_ch = 674.6
# c3-c3-c3 angle: K=63.21, theta0=110.63 deg
K_ccc = 126.42
# hc-c3-hc angle: K=39.43, theta0=108.35 deg
K_hch = 78.86
# c3-c3-hc angle: K=46.37, theta0=110.05 deg
K_cch = 92.74
# hc-c3-c3-hc torsion: IDIVF=1, PK=0.15, PHASE=0, PN=3 (gaff.dat also lists
# hc-c3-c3-c3 at PK=0.16, close enough to treat as the same constant here,
# consistent with this file not otherwise splitting torsions by terminal
# atom identity). Doubled from GAFF's E=(PK/IDIVF)*(1+cos(n*phi-phase))
# convention to match this file's 0.5*K*(1+cos(n*phi)) convention.
K_ccTorsional = 0.30

# aromatic (e.g. benzene ring) variants: sp2/trigonal-planar carbons have a
# stiffer, shorter C-C bond and a ~120 deg (vs. ~109.5-111 deg sp3) bond
# angle, so the alkane constants above don't apply to them.
# gaff.dat atom types ca (aromatic C), ha (H on aromatic C).
# ca-ca bond: K=478.4, r0=1.3870
K_cc_aromatic = 956.8
# ca-ca-ca angle: K=67.18, theta0=119.97 deg
K_ccc_aromatic = 134.36
# ca-ca-ha angle: K=48.46, theta0=120.01 deg
K_cch_aromatic = 96.92

# ring torsion: the alkane K_ccTorsional above is a 3-fold (staggered-minima)
# potential and gets applied to every C-C-C-C quad including aromatic rings,
# which instead need a 2-fold potential with planar (0/180 deg) minima.
# X-ca-ca-X torsion: IDIVF=4, PK=14.500, PHASE=180.0, PN=2.0; annotated in
# gaff.dat as "intrpol.bsd.on C6H6" (interpolated based on benzene). Per-quad
# barrier is PK/IDIVF = 3.625 kcal/mol, giving
# E=(PK/IDIVF)*(1+cos(2*phi-180)) = 3.625*(1-cos(2*phi)); doubled to
# K_ccTorsional_aromatic=7.25 to match this file's 0.5*K*(...) convention.
K_ccTorsional_aromatic = 7.25

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
X_cc = 1.5350 * A2m
X_ch = 1.0920 * A2m
X_ccc = 1.9308577514813268
X_hch = 1.8910642445358559
X_cch = 1.9207348418197596

# Å
# rad (119.97 deg, 120.01 deg)
X_cc_aromatic = 1.3870 * A2m
X_ccc_aromatic = 2.093871503617597
X_cch_aromatic = 2.094569635318395

# distance-energy conversion constants
# angle-energy conversion constants

distEnergyK_cc = K_cc * kcal2MU / (N * A2m ** 2)
distEnergyK_ch = K_ch * kcal2MU / (N * A2m ** 2)
angleEnergyK_ccc = K_ccc * kcal2MU / N
angleEnergyK_hch = K_hch * kcal2MU / N
angleEnergyK_cch = K_cch * kcal2MU / N
angleEnergyK_ccTorsional = K_ccTorsional * kcal2MU / N

distEnergyK_cc_aromatic = K_cc_aromatic * kcal2MU / (N * A2m ** 2)
angleEnergyK_ccc_aromatic = K_ccc_aromatic * kcal2MU / N
angleEnergyK_cch_aromatic = K_cch_aromatic * kcal2MU / N
angleEnergyK_ccTorsional_aromatic = K_ccTorsional_aromatic * kcal2MU / N

# jit/vmap switch

jit_funcs = True
vmap_funcs = True
