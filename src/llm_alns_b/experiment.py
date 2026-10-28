"""Run frozen-policy experiments using explicitly supplied external assets."""
from pathlib import Path
from urllib.parse import quote

from .data import expand_route, load_dataset, load_policy, read_json, sha256, write_json
from .runtime import SearchConfig, solve


def load_experiment(experiment_path):
    path = Path(experiment_path).resolve()
    declared = read_json(path)
    if declared.get('schema') != 'llm-alns-b.experiment.v1':
        raise ValueError('Unsupported experiment schema')
    required = {'schema', 'parameters', 'data_manifest', 'policy_manifest'}
    if not required <= declared.keys() or declared.keys() - (required | {'pairs', 'description'}):
        raise ValueError('Missing or unknown experiment field')
    parameters_path = path.parent / declared['parameters']
    data_path = path.parent / declared['data_manifest']
    policy_path = path.parent / declared['policy_manifest']
    config = SearchConfig.from_dict(read_json(parameters_path))
    instances, default_pairs, manifest = load_dataset(data_path)
    components, policy_manifest = load_policy(policy_path, config)
    pairs = declared.get('pairs', default_pairs)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError('Declare at least one experiment pair')
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {'instance', 'seed'}:
            raise ValueError('Each pair requires instance and seed')
        if pair['instance'] not in instances or type(pair['seed']) is not int or pair['seed'] < 0:
            raise ValueError('Unknown instance or invalid seed')
        key = (pair['instance'], pair['seed'])
        if key in seen:
            raise ValueError('Duplicate experiment pair')
        seen.add(key)
    identities = dict(experiment=sha256(path), parameters=sha256(parameters_path),
                      dataset=sha256(data_path), policy=sha256(policy_path))
    return config, instances, components, pairs, identities


def run_experiment(experiment_path, output):
    config, instances, components, pairs, identities = load_experiment(experiment_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'parameters.json', config.to_dict())
    write_json(output / 'input_identities.json', identities)
    summaries = []
    for pair in pairs:
        instance = instances[pair['instance']]
        folder = output / (quote(instance.name, safe='') + '-' + str(pair['seed']))
        folder.mkdir()
        result = solve(instance, pair['seed'], config, components, trace_path=folder / 'trace.jsonl')
        result.pop('records', None)
        write_json(folder / 'solution.json', result)
        write_json(folder / 'expanded_route.json', expand_route(instance, result['best_route']))
        summaries.append({key: result[key] for key in ('instance', 'seed', 'budget', 'best_time', 'best_route')})
    report = dict(schema='llm-alns-b.search-report.v1', pairs=summaries,
                  input_identities=identities, external_assets=True, network_requests=0)
    write_json(output / 'summary.json', report)
    return report
