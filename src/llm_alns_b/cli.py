"""Public commands; model generation and training are intentionally absent."""
import argparse
import json
from pathlib import Path
import sys

from .data import precompute
from .demo import make_demo
from .experiment import load_experiment, run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description='Configurable B search with external experiment assets')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Generate synthetic inputs and run the actual search')
    demo.add_argument('--output', required=True)
    run = commands.add_parser('run', help='Reproduce search experiments using externally supplied private assets')
    run.add_argument('--experiment', required=True)
    run.add_argument('--output', required=True)
    validate = commands.add_parser('validate', help='Validate external inputs without running search')
    validate.add_argument('--experiment', required=True)
    prep = commands.add_parser('precompute', help='Build A* costs and verify paths for your own inputs')
    prep.add_argument('--occupancy', required=True)
    prep.add_argument('--nodes', required=True)
    prep.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'demo':
            output = Path(args.output).resolve()
            output.mkdir(parents=True, exist_ok=False)
            experiment = make_demo(output / 'synthetic_assets')
            report = run_experiment(experiment, output / 'results')
            print(json.dumps(dict(status='PASS', synthetic_demo=True, pairs=len(report['pairs']),
                                  network_requests=0, output=str(output)), ensure_ascii=False, indent=2))
        elif args.command == 'run':
            report = run_experiment(args.experiment, args.output)
            print(json.dumps(dict(status='PASS', pairs=len(report['pairs']), network_requests=0), indent=2))
        elif args.command == 'validate':
            _, instances, _, pairs, _ = load_experiment(args.experiment)
            print(json.dumps(dict(status='PASS', instances=len(instances), pairs=len(pairs), network_requests=0), indent=2))
        elif args.command == 'precompute':
            print(precompute(args.occupancy, args.nodes, args.output))
        return 0
    except (ValueError, TypeError, KeyError, FileExistsError, FileNotFoundError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
