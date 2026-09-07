# MolSim

We developed a program to simulate the dynamics of hydrocarbon molecules using the consistent force field method devised by Shneior Lifson and Arieh Warshel.

# Setup

```
pip install -r requirements.txt
```

# File Descriptions

`mol_simulation.py` runs the simulation for one of `ethane`, `propane`, `isobutane`, or `benzene` and writes its output (topology, energy/bond history, plots) to `output/<molecule>/`, e.g.:

```
python mol_simulation.py ethane
```

`mol_notebook.ipynb` converts these CSV files to readable graphs, some of which were used in our paper linked below.

`mol_display.py` reads a molecule's output directory and renders it with VPython, which we screen-recorded and have posted on YouTube, also linked below:

```
python mol_display.py ethane
```

# Links

Paper: https://drive.google.com/file/d/1iTxnto4CWynYNn1BvOq8889JQxFhdTYQ

CSV files (7zip or equivalent required to open): https://drive.google.com/file/d/1DCI9PWngpDCOwGWAZNWLLJBHga-x_eFh

YouTube: https://youtu.be/iCUkThONkhc
