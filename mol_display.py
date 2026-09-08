import argparse as ap
import csv
import json
import os
import shutil
import subprocess
import webbrowser

from constants import X_cc, X_cc_aromatic, X_ch

parser = ap.ArgumentParser(description = "Render an interactive HTML viewer for a molecule simulated by mol_simulation.py")
parser.add_argument('molecule', help = "molecule name (must match an output/<molecule> directory produced by mol_simulation.py)")
args = parser.parse_args()

output_dir = os.path.join('output', args.molecule)
mol_info_path = os.path.join(output_dir, args.molecule + '.csv')
position_history_path = os.path.join(output_dir, args.molecule + '_positionHistory.csv')
energy_history_path = os.path.join(output_dir, args.molecule + '_energyHistory.csv')
bond_length_history_path = os.path.join(output_dir, args.molecule + '_bondLengthHistory.csv')

atomTypes = []
bondPairs = []
atomIdx = 0
bondIdx = 0
totalAtoms = 0
totalBonds = 0

readPhase = 0

with open(mol_info_path, newline = '') as molInfo:
    molReader = csv.reader(molInfo, delimiter = ',', quotechar = '"')

    for row in molReader:
        if readPhase == 0:
            totalAtoms = int(row[0])
            readPhase += 1
        elif readPhase == 1:
            # scale (ticks per recorded frame) -- not needed for the static
            # viewer, kept only so this loop's phase structure matches the
            # format mol_simulation.py actually writes.
            readPhase += 1
        elif readPhase == 2:
            atomTypes.append(row[0].split('.')[-1])
            atomIdx += 1

            if atomIdx == totalAtoms:
                readPhase += 1
                atomIdx = 0
        elif readPhase == 3 or bondIdx == totalBonds:
            totalBonds = int(row[0])
            readPhase += 1
        elif readPhase == 4:
            bondPairs.append((atomIdx, int(row[0])))
            bondIdx += 1

            if bondIdx == totalBonds:
                atomIdx += 1
                bondIdx = 0
                readPhase -= 1

# each true bond appears twice in bondPairs, once from each endpoint's
# neighbor list -- collapse to the unique undirected edges.
edges = []
seenEdges = set()
for a, b in bondPairs:
    key = frozenset((a, b))
    if key not in seenEdges:
        seenEdges.add(key)
        edges.append((a, b))

# the simulated data model (molecules.py/simulation.py) only tracks bond
# distances, not bond order, so bond order for display is inferred purely
# from connectivity: a 6-membered all-carbon ring is drawn as a stylized
# Kekule structure with alternating double/single bonds, same as benzene's
# real ring. molecules with no such ring (ethane, propane, ...) are
# unaffected and every bond stays single.
carbonAdjacency = {}
for a, b in edges:
    if atomTypes[a] == 'C' and atomTypes[b] == 'C':
        carbonAdjacency.setdefault(a, set()).add(b)
        carbonAdjacency.setdefault(b, set()).add(a)

def find_six_rings(adjacency):
    rings = {}
    for start in adjacency:
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nxt in adjacency.get(node, ()):
                if nxt == start and len(path) == 6:
                    rings[frozenset(path)] = path
                elif nxt not in path and len(path) < 6:
                    stack.append((nxt, path + [nxt]))
    return list(rings.values())

edgeOrder = {}
edgeRing = {}
for ring in find_six_rings(carbonAdjacency):
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        key = frozenset((a, b))
        edgeOrder[key] = 2 if i % 2 == 0 else 1
        edgeRing[key] = ring

hasAromaticRing = len(edgeRing) > 0

bonds = []
for a, b in edges:
    key = frozenset((a, b))
    bonds.append({
        'a': a,
        'b': b,
        'order': edgeOrder.get(key, 1),
        'ring': edgeRing.get(key),
    })

positionFrames = []
frameTimes = []

with open(position_history_path, newline = '') as posHistory:
    posReader = csv.reader(posHistory, delimiter = ',', quotechar = '"')
    next(posReader)

    frame = [None] * totalAtoms
    frameTime = None
    for row in posReader:
        idx = int(row[2])
        frameTime = float(row[1])
        frame[idx] = [float(row[3]), float(row[4]), float(row[5])]

        if idx == totalAtoms - 1:
            positionFrames.append(frame)
            frameTimes.append(frameTime)
            frame = [None] * totalAtoms

def read_history(path):
    with open(path, newline = '') as f:
        reader = csv.reader(f, delimiter = ',', quotechar = '"')
        next(reader)
        return list(reader)

energyRows = read_history(energy_history_path)
bondRows = read_history(bond_length_history_path)

# position-history rows and tick-history (energy/bond) rows are recorded
# once per iteration of the same simulation loop in mol_simulation.py, so
# they line up 1:1 by index -- even though the "time" columns are in
# different units (seconds vs. picoseconds; see record() in simulation.py).
# Use the position-history time (seconds, matches the viewer's fmtTime/
# xFmtT conventions below) as the merged frame's canonical time.
nFrames = min(len(positionFrames), len(energyRows), len(bondRows))

frames = []
for i in range(nFrames):
    frames.append({
        't': frameTimes[i],
        'pe': float(energyRows[i][2]),
        'ke': float(energyRows[i][3]),
        'cc': float(bondRows[i][2]),
        'ch': float(bondRows[i][3]),
        'pos': positionFrames[i],
    })

ccTarget = X_cc_aromatic if hasAromaticRing else X_cc
chTarget = X_ch

def fmt_time_label(seconds):
    fs = seconds * 1e15
    if fs < 1000:
        return f'{fs:.2f} fs'
    return f'{seconds * 1e12:.4g} ps'

dtSeconds = frames[1]['t'] - frames[0]['t'] if len(frames) > 1 else 0
spanSeconds = frames[-1]['t'] - frames[0]['t'] if frames else 0

paramRows = [
    ('Atoms', str(totalAtoms)),
    ('Bonds', str(len(edges))),
    ('Frame &Delta;t', fmt_time_label(dtSeconds)),
    ('Frames', str(len(frames))),
    ('Span', fmt_time_label(spanSeconds)),
]
paramsHtml = ''.join(
    f'<div><dt>{k}</dt><dd>{v}</dd></div>' for k, v in paramRows
)

TYPE_LABELS = {'C': 'Carbon', 'H': 'Hydrogen'}
TYPE_DOT_CLASS = {'C': 'carbon', 'H': 'hydrogen'}
presentTypes = sorted(set(atomTypes), key = lambda t: atomTypes.index(t))
legendHtml = ''.join(
    '<span><i class="dot {}"></i>{}</span>'.format(
        TYPE_DOT_CLASS.get(t, 'carbon'), TYPE_LABELS.get(t, t))
    for t in presentTypes
)

moleculeTitle = args.molecule.capitalize()

data = {
    'meta': {
        'ccTarget': ccTarget,
        'chTarget': chTarget,
    },
    'types': atomTypes,
    'bonds': [
        {'a': b['a'], 'b': b['b'], 'order': b['order'], 'ring': list(b['ring']) if b['ring'] else None}
        for b in bonds
    ],
    'frames': frames,
}

_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mol_display_template.html')
with open(_TEMPLATE_PATH) as f:
    _TEMPLATE = f.read()

html = (_TEMPLATE
    .replace('@@TITLE_NAME@@', moleculeTitle)
    .replace('@@PARAMS@@', paramsHtml)
    .replace('@@LEGEND@@', legendHtml)
    .replace('@@DATA_JSON@@', json.dumps(data))
)

out_html_path = os.path.join(output_dir, args.molecule + '_viewer.html')
with open(out_html_path, 'w') as f:
    f.write(html)

abs_path = os.path.abspath(out_html_path)
print(f"Viewer written to {out_html_path}")

opened = False
if shutil.which('wslpath') and shutil.which('explorer.exe'):
    win_path = subprocess.run(
        ['wslpath', '-w', abs_path], capture_output = True, text = True, check = True
    ).stdout.strip()
    print(f"Opening {win_path}")
    subprocess.run(['explorer.exe', win_path], check = False)
    opened = True

if not opened:
    url = 'file://' + abs_path
    print(f"Opening {url}")
    webbrowser.open(url)
