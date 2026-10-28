"""Portable grid inputs, verified A* preprocessing, and route expansion."""
from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np

from .runtime import GridMap, Instance, astar, route_time


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Dataset paths must stay inside their manifest directory')
    return path


def _validate_arrays(name, T, blocked, cells):
    if blocked.ndim != 2 or not np.isin(blocked, (0, 1)).all():
        raise ValueError(f'{name}: occupancy must be a binary 2D array')
    if (T.dtype != np.float64 or T.ndim != 2 or T.shape[0] != T.shape[1]
            or T.shape[0] != len(cells) or len(cells) < 3):
        raise ValueError(f'{name}: require a float64 (N+1)x(N+1) matrix and depot plus tasks')
    if not np.isfinite(T).all() or (T < 0).any() or np.any(T.diagonal()) or not np.array_equal(T, T.T):
        raise ValueError(f'{name}: require finite nonnegative symmetric costs with zero diagonal')
    if len(set(cells)) != len(cells):
        raise ValueError(f'{name}: cells must be unique')
    h, w = blocked.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or blocked[r, c]:
            raise ValueError(f'{name}: each depot/task cell must be unblocked and in bounds')


def load_dataset(manifest_path):
    """Return instances, declared pairs, and manifest; validate file identities."""
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = read_json(manifest_path)
    if manifest.get('schema') != 'llm-alns-b.dataset.v1':
        raise ValueError('Unsupported dataset schema')
    instances = {}
    for entry in manifest['instances']:
        name = entry['id']
        if name in instances:
            raise ValueError('Duplicate instance ID')
        folder = _within(root, entry['directory'])
        occupancy = _within(root, entry['occupancy'])
        if sha256(occupancy) != entry['occupancy_sha256']:
            raise ValueError(f'{name}: occupancy identity mismatch')
        for filename in ('nodes.json', 'node_cells.npy', 'distance_matrix.npy', 'shortest_paths.json.gz'):
            if sha256(_within(folder, filename)) != entry['files'][filename]:
                raise ValueError(f'{name}: file identity mismatch: {filename}')
        raw_cells = np.load(folder / 'node_cells.npy', allow_pickle=False)
        if raw_cells.ndim != 2 or raw_cells.shape[1] != 2 or not np.issubdtype(raw_cells.dtype, np.integer):
            raise ValueError(f'{name}: cells must be an integer (N+1)x2 array')
        cells = [tuple(map(int, p)) for p in raw_cells.tolist()]
        nodes = read_json(folder / 'nodes.json')
        if nodes['id'] != name or nodes['cells_row_col'] != [list(p) for p in cells]:
            raise ValueError(f'{name}: node metadata differs from cell array')
        T = np.load(folder / 'distance_matrix.npy', allow_pickle=False)
        blocked = np.load(occupancy, allow_pickle=False)
        _validate_arrays(name, T, blocked, cells)
        if entry.get('N', len(cells) - 1) != len(cells) - 1:
            raise ValueError(f'{name}: task-count metadata differs')
        T.flags.writeable = False
        blocked.flags.writeable = False
        instance_meta = dict(entry, path_file=str(folder / 'shortest_paths.json.gz'))
        instances[name] = Instance(name, T, blocked, cells, instance_meta)
    seeds = manifest.get('solver_seeds', [])
    if any(isinstance(s, bool) or not isinstance(s, int) for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError('Solver seeds must be unique integers')
    pairs = [dict(instance=name, seed=s) for name in instances for s in seeds]
    return instances, pairs, manifest


def load_policy(manifest_path, config):
    """Load exact frozen source files with file-hash verification."""
    from .runtime import Components
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    if manifest.get('schema') != 'llm-alns-b.policy.v1':
        raise ValueError('Unsupported policy schema')
    sources = {}
    for kind in ('D', 'Q'):
        source = _within(path.parent, manifest[kind]['file'])
        if sha256(source) != manifest[kind]['sha256']:
            raise ValueError(f'Frozen {kind} source identity mismatch')
        sources[kind] = source.read_text(encoding='utf-8')
    return Components(**sources, config=config), manifest


def independent_distances(blocked, start):
    """Independent Dijkstra check, with the same permitted grid movements."""
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    h, w = blocked.shape
    distances = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        distance, (r, c) = heapq.heappop(heap)
        if distance != distances[(r, c)]:
            continue
        for dr, dc in steps:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w) or blocked[rr, cc]:
                continue
            if dr and dc and (blocked[r, cc] or blocked[rr, c]):
                continue
            point = (rr, cc)
            new = distance + (math.sqrt(2) if dr and dc else 1.0)
            if new < distances.get(point, math.inf):
                distances[point] = new
                heapq.heappush(heap, (new, point))
    return distances


def validate_grid_path(blocked, path, start, goal, expected):
    path = [tuple(p) for p in path]
    if not path or path[0] != tuple(start) or path[-1] != tuple(goal):
        raise ValueError('Grid-path endpoints differ')
    costs = []
    h, w = blocked.shape
    for r, c in path:
        if not (0 <= r < h and 0 <= c < w) or blocked[r, c]:
            raise ValueError('Grid path crosses a blocked/out-of-bounds cell')
    for (r, c), (rr, cc) in zip(path, path[1:]):
        dr, dc = abs(rr - r), abs(cc - c)
        if max(dr, dc) != 1 or (dr and dc and (blocked[r, cc] or blocked[rr, c])):
            raise ValueError('Illegal grid step or corner crossing')
        costs.append(math.sqrt(2) if dr and dc else 1.0)
    total = math.fsum(costs)
    if not math.isclose(total, float(expected), rel_tol=0, abs_tol=1e-10):
        raise ValueError('Expanded grid-path cost differs from the matrix')
    return total


def expand_route(instance, route, path_file=None):
    """Expand depot -> service permutation -> depot into obstacle-safe grid cells."""
    objective = route_time(route, instance)
    path_file = path_file or instance.manifest.get('path_file')
    if not path_file:
        raise ValueError('A precomputed path file is required')
    with gzip.open(path_file, 'rt', encoding='utf-8') as stream:
        rows = json.load(stream)
    index = {(row['i'], row['j']): row for row in rows}
    expanded, segments = [], []
    order = [0, *map(int, route), 0]
    for a, b in zip(order, order[1:]):
        if a == b:
            path = [instance.cells[a]]
        else:
            row = index[min(a, b), max(a, b)]
            path = [tuple(p) for p in row['path']]
            if a > b:
                path.reverse()
        cost = validate_grid_path(instance.blocked, path, instance.cells[a], instance.cells[b], instance.T[a, b])
        expanded.extend(path if not expanded else path[1:])
        segments.append(dict(from_node=a, to_node=b, cost=cost, cells=[list(p) for p in path]))
    if not math.isclose(math.fsum(s['cost'] for s in segments), objective, rel_tol=0, abs_tol=1e-9):
        raise ValueError('Expanded route cost mismatch')
    return dict(service_order=order, objective=objective, cells=[list(p) for p in expanded], segments=segments)


def precompute(occupancy_path, nodes_path, output, *, name=None, split='CUSTOM', seeds=()):
    """Prepare a new self-contained dataset from binary occupancy and ordered cells."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory')
    blocked = np.load(occupancy_path, allow_pickle=False)
    metadata = read_json(nodes_path)
    raw_cells = metadata['cells_row_col']
    if any(len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in p) for p in raw_cells):
        raise ValueError('cells_row_col must contain integer [row, col] pairs')
    cells = [tuple(p) for p in raw_cells]
    name = name or metadata['id']
    T = np.zeros((len(cells), len(cells)), dtype=np.float64)
    _validate_arrays(name, T, blocked, cells)
    host_grid = GridMap(blocked.astype(bool), cells[0])
    paths = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            distance, path = astar(host_grid, cells[i], cells[j])
            if not math.isfinite(distance):
                raise ValueError(f'Unreachable tasks: {i}, {j}')
            validate_grid_path(blocked, path, cells[i], cells[j], distance)
            T[i, j] = T[j, i] = distance
            paths.append(dict(i=i, j=j, distance=distance, path=path))
    max_error = 0.0
    for i, start in enumerate(cells):
        distances = independent_distances(blocked, start)
        for j, end in enumerate(cells):
            error = abs(float(T[i, j]) - distances[end])
            max_error = max(max_error, error)
            if error > 1e-10:
                raise ValueError('A* and independent Dijkstra disagree')
    folder = output / 'instance'
    folder.mkdir(parents=True, exist_ok=False)
    np.save(output / 'occupancy.npy', blocked, allow_pickle=False)
    np.save(folder / 'node_cells.npy', np.asarray(cells, dtype=np.int64), allow_pickle=False)
    np.save(folder / 'distance_matrix.npy', T, allow_pickle=False)
    write_json(folder / 'nodes.json', dict(id=name, split=split, cells_row_col=raw_cells, N=len(cells) - 1, base=raw_cells[0]))
    with gzip.GzipFile(filename=str(folder / 'shortest_paths.json.gz'), mode='xb', mtime=0) as stream:
        stream.write(json.dumps(paths, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8'))
    files = {file.name: sha256(file) for file in folder.iterdir()}
    entry = dict(id=name, split=split, N=len(cells)-1, directory='instance', occupancy='occupancy.npy',
                 occupancy_sha256=sha256(output / 'occupancy.npy'), files=files)
    manifest = dict(schema='llm-alns-b.dataset.v1', split=split, instances=[entry], solver_seeds=list(seeds),
                    movement='8-neighbor; no corner crossing', cost_unit='normalized grid distance',
                    verification=dict(astar_pairs=len(paths), dijkstra_sources=len(cells), max_abs_error=max_error))
    write_json(output / 'manifest.json', manifest)
    return output / 'manifest.json'

