"""Independent synthetic tests. No private experiment asset is imported."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from llm_alns_b.data import (expand_route, independent_distances, load_dataset, load_policy,
                             read_json, validate_grid_path)
from llm_alns_b.demo import make_demo
from llm_alns_b.experiment import load_experiment, run_experiment
from llm_alns_b.runtime import GridMap, ProgramError, SafeProgram, SearchConfig, astar, solve
from llm_alns_b.runtime.components import weight_update


class PublicSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.experiment = make_demo(self.root / 'synthetic')
        self.config, self.instances, self.components, self.pairs, _ = load_experiment(self.experiment)
        self.instance = self.instances['synthetic_demo']

    def tearDown(self):
        self.temp.cleanup()

    def test_external_configuration_is_complete_and_strict(self):
        values = self.config.to_dict()
        self.assertEqual(SearchConfig.from_dict(values), self.config)
        missing = dict(values)
        del missing['temperature_decay']
        with self.assertRaises(ValueError):
            SearchConfig.from_dict(missing)
        with self.assertRaises(ValueError):
            SearchConfig.from_dict(dict(values, undeclared_control=True))
        with self.assertRaises(ValueError):
            SearchConfig.from_dict(dict(values, iterations=True))
        with self.assertRaises(ValueError):
            SearchConfig.from_dict(dict(values, initial_destroy_weights=[]))

    def test_actual_search_runs_and_is_stepwise_deterministic(self):
        first = solve(self.instance, self.pairs[0]['seed'], self.config, self.components)
        second = solve(self.instance, self.pairs[0]['seed'], self.config, self.components)
        for field in ('wall_seconds', 'cpu_seconds'):
            first.pop(field)
            second.pop(field)
        self.assertEqual(first, second)
        self.assertEqual(first['evaluations'], self.config.iterations)
        self.assertEqual(sorted(first['best_route']), list(range(1, self.instance.n + 1)))
        self.assertLessEqual(first['best_time'], first['initial_time'])
        self.assertEqual(len(first['records']), self.config.iterations)

    def test_recent_statistics_follow_external_window(self):
        result = solve(self.instance, self.pairs[0]['seed'], self.config, self.components)
        records = result['records']
        for i, row in enumerate(records):
            previous = records[max(0, i - self.config.recent_window):i]
            expected = sum(r['accepted'] and r['changed_route'] for r in previous) / max(1, len(previous))
            self.assertAlmostEqual(row['state']['recent_acceptance_rate'], expected)

    def test_weight_reaction_uses_the_supplied_configuration(self):
        destroy, repair = [2.0, 4.0], [3.0, 5.0]
        old_d, old_r = destroy[:], repair[:]
        reward = self.config.reward_best
        weight_update(destroy, repair, 1, 0, reward, config=self.config)
        self.assertEqual(destroy[0], old_d[0])
        self.assertEqual(repair[1], old_r[1])
        self.assertAlmostEqual(destroy[1], (1-self.config.weight_reaction)*old_d[1] + self.config.weight_reaction*reward)
        self.assertAlmostEqual(repair[0], (1-self.config.weight_reaction)*old_r[0] + self.config.weight_reaction*reward)

    def test_results_include_replay_inputs_trace_and_valid_grid_route(self):
        output = self.root / 'run'
        report = run_experiment(self.experiment, output)
        self.assertEqual(report['network_requests'], 0)
        self.assertEqual(read_json(output / 'parameters.json'), self.config.to_dict())
        result_folder = next(p for p in output.iterdir() if p.is_dir())
        result = read_json(result_folder / 'solution.json')
        expanded = read_json(result_folder / 'expanded_route.json')
        self.assertAlmostEqual(expanded['objective'], result['best_time'])
        self.assertEqual(expanded['cells'][0], list(self.instance.cells[0]))
        self.assertEqual(expanded['cells'][-1], list(self.instance.cells[0]))
        self.assertEqual(len((result_folder / 'trace.jsonl').read_text().splitlines()), self.config.iterations)
        with self.assertRaises(FileExistsError):
            run_experiment(self.experiment, output)

    def test_grid_geometry_independently_agrees(self):
        grid = GridMap(self.instance.blocked.astype(bool), self.instance.cells[0])
        source = self.instance.cells[0]
        distances = independent_distances(self.instance.blocked, source)
        for target in self.instance.cells:
            distance, path = astar(grid, source, target)
            self.assertAlmostEqual(distance, distances[target])
            validate_grid_path(self.instance.blocked, path, source, target, distance)
        blocked = np.array([[0, 1], [1, 0]], dtype=bool)
        distance, path = astar(GridMap(blocked, (0, 0)), (0, 0), (1, 1))
        self.assertFalse(np.isfinite(distance))
        self.assertEqual(path, [])

    def test_changed_input_or_policy_is_rejected(self):
        policy = self.experiment.parent / 'policies/destroy.py'
        policy.write_text(policy.read_text() + '\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            load_experiment(self.experiment)

    def test_policy_cannot_execute_arbitrary_python(self):
        invalid_sources = (
            'import os\ndef degree(state, rng):\n    return 1\n',
            'def degree(state, rng):\n    return rng.random()\n',
            'def degree(state, rng):\n    return random_remove([], 1, [], rng)\n',
        )
        for source in invalid_sources:
            with self.subTest(source=source), self.assertRaises(ProgramError):
                SafeProgram(source, 'Q')

    def test_unknown_and_duplicate_pairs_are_rejected(self):
        declared = read_json(self.experiment)
        declared['pairs'] = [self.pairs[0], self.pairs[0]]
        self.experiment.write_text(json.dumps(declared), encoding='utf-8')
        with self.assertRaises(ValueError):
            load_experiment(self.experiment)
        declared['pairs'] = [dict(instance='missing', seed=17)]
        self.experiment.write_text(json.dumps(declared), encoding='utf-8')
        with self.assertRaises(ValueError):
            load_experiment(self.experiment)


if __name__ == '__main__':
    unittest.main()
