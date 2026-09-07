# MolSim

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

`mol_display.py` reads a molecule's output directory and renders it with VPython, which we screen-recorded and have posted on YouTube, also linked below:

```
python mol_display.py ethane
```

# Links

Paper: [paper.pdf](paper.pdf)

Original consistent force field paper: S. Lifson and A. Warshel, "Consistent Force Field for Calculations of Conformations, Vibrational Spectra, and Enthalpies of Cycloalkane and n-Alkane Molecules," J. Chem. Phys. 49, 5116 (1968). https://doi.org/10.1063/1.1670007

CSV files (7zip or equivalent required to open): https://drive.google.com/file/d/1DCI9PWngpDCOwGWAZNWLLJBHga-x_eFh

YouTube: https://youtu.be/iCUkThONkhc
