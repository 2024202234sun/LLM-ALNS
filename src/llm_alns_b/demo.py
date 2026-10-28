"""Synthetic demonstration assets, unrelated to the private research inputs."""
import hashlib
from pathlib import Path

import numpy as np

from .data import precompute, read_json, sha256, write_json


def make_demo(output):
    """Generate a small obstacle map, public example policies and experiment."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'inputs'
    source.mkdir()
    blocked = np.zeros((18, 22), dtype=np.uint8)
    blocked[4:14, 10] = 1
    blocked[8, 10] = 0
    cells = [[1, 1], [2, 4], [3, 8], [2, 15], [5, 18], [8, 5], [8, 16],
             [11, 3], [12, 7], [14, 12], [15, 18], [16, 4], [9, 19]]
    np.save(source / 'occupancy.npy', blocked, allow_pickle=False)
    write_json(source / 'nodes.json', dict(id='synthetic_demo', cells_row_col=cells))
    dataset = precompute(source / 'occupancy.npy', source / 'nodes.json', output / 'dataset', seeds=(17,))
    parameters_source = Path(__file__).resolve().parent / 'demo_parameters.json'
    parameters = read_json(parameters_source)
    write_json(output / 'parameters.json', parameters)
    models = output / 'policies'
    models.mkdir()
    # This elementary illustrative pair is not a trained policy.
    d_code = 'def destroy(route, q, dist, state, rng):\n    return random_remove(route, q, dist, rng)\n'
    q_code = 'def degree(state, rng):\n    return 0.8\n'
    (models / 'destroy.py').write_text(d_code, encoding='utf-8')
    (models / 'degree.py').write_text(q_code, encoding='utf-8')
    write_json(models / 'manifest.json', dict(schema='llm-alns-b.policy.v1',
        description='Synthetic illustration only; no trained research policy.',
        D=dict(file='destroy.py', sha256=sha256(models / 'destroy.py')),
        Q=dict(file='degree.py', sha256=sha256(models / 'degree.py'))))
    write_json(output / 'experiment.json', dict(schema='llm-alns-b.experiment.v1',
        description='Synthetic demonstration, not a paper experiment.',
        parameters='parameters.json', data_manifest='dataset/manifest.json', policy_manifest='policies/manifest.json'))
    return output / 'experiment.json'
