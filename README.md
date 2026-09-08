> **Update (2026-09-07):** Revisited this project and fixed a bug in the potential-energy calculations that was producing inaccurate dynamics, most visibly benzene's ring bending out of its planar shape instead of staying rigid. The original version of this repo remains available at [`5fd783c`](https://github.com/f4t4nt/molecular-simulation/tree/5fd783c7d19ee6d77d2b2c405a6d21602bd6cb93).

# Hydrocarbon Molecular Simulation

We developed a program to simulate the dynamics of hydrocarbon molecules using the consistent force field method devised by Shneior Lifson and Arieh Warshel.

# Setup

```
pip install -r requirements.txt
```

JAX runs on CPU by default. If you have an NVIDIA GPU, you can install CUDA support instead for faster runs:

```
pip install "jax[cuda12]"
```

# File Descriptions

`mol_simulation.py` runs the simulation for one of `ethane`, `propane`, `isobutane`, or `benzene` and writes its output (topology, energy/bond history, plots) to `output/<molecule>/`, e.g.:

```
python mol_simulation.py ethane
```

By default it runs 10,000 iterations; pass `--duration <picoseconds>` instead to control how much simulated time to run, e.g. `--duration 0.01`. Run `python mol_simulation.py --help` for the full list of options (timestep, randomization, etc).

`mol_display.py` reads a molecule's output directory and renders it as a self-contained, interactive HTML viewer (3D molecule view plus energy and bond-length charts), e.g.:

```
python mol_display.py ethane
```

# Links

- Paper: [paper.pdf](paper.pdf)
- Original consistent force field paper: S. Lifson and A. Warshel, ["Consistent Force Field for Calculations of Conformations, Vibrational Spectra, and Enthalpies of Cycloalkane and n-Alkane Molecules"](https://doi.org/10.1063/1.1670007), J. Chem. Phys. 49, 5116 (1968)
- CSV files (7zip or equivalent required to open): [Google Drive](https://drive.google.com/file/d/1DCI9PWngpDCOwGWAZNWLLJBHga-x_eFh)
- Screen recording of the original VPython viewer: [YouTube](https://youtu.be/iCUkThONkhc)
