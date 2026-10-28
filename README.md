# B: configurable USV route-search runtime

This public edition retains the executable route-search rules, obstacle-aware preprocessing and experiment runner. **Research data, tuned parameters, trained policies, recorded results and policy-evolution implementation are not distributed.**

The search runner can reproduce frozen-policy routing experiments when their original private inputs are supplied. A self-contained synthetic demonstration also runs without those inputs. The public files alone cannot reproduce the withheld paper's numerical results or retrain its policies.

[中文说明](docs/USAGE_ZH.md) · [External asset interface](docs/EXTERNAL_ASSETS.md)

## Install and run

Use a fresh environment for this public edition:

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python -m llm_alns_b demo --output runs/demo
```

The demo creates synthetic occupancy and task cells, prepares A* costs, constructs elementary illustrative policies, runs the actual search, and expands the service permutation into a depot-to-depot grid path. Its parameters were selected for demonstration; they are not the research configuration, and its policies are not trained research artifacts. The package makes no network request.

## Reproduce a search experiment with private inputs

Keep private inputs outside the public checkout and supply an experiment manifest:

```sh
python -m llm_alns_b validate --experiment /path/to/private/experiment.json
python -m llm_alns_b run --experiment /path/to/private/experiment.json --output /path/to/local/results
```

The manifest refers to a required parameter JSON, a dataset manifest and a frozen-policy manifest. Paths are relative to the experiment file. The runner verifies asset hashes and records the effective parameters, input identities, full iteration traces, solutions and expanded grid paths. Output directories must be new.

Exact search reproduction requires the same data, task ordering, instance name, policies, parameters, seed and compatible numerical environment. Instance names participate in random-stream derivation. Keeping executable search rules does not supply withheld inputs automatically.

## Search rules retained

The runtime uses a depot-closed service permutation and nearest-neighbor initialization. Each iteration performs operator selection, removal-count evaluation, task removal, repair, best-route update, simulated-annealing acceptance and adaptive-weight update. Recent search statistics are exposed to the supplied policies. Random streams are separated by instance, seed, iteration and role.

The trusted route primitives, R1–R4 repair implementations, restricted policy interpreter and deterministic tie rules are included. A* uses eight-neighbor movement with no blocked-corner crossing; complete node-pair costs are independently checked. Grid-distance units are not calibrated physical seconds.

All experiment controls—including iteration budget, random namespace, weights, acceptance schedule, reward values, history window and removal-degree mapping—must be supplied explicitly. There are no fallback research values. Interpreter safety limits and elementary geometric constants remain implementation safeguards rather than research configuration.

## Custom inputs

```sh
python -m llm_alns_b precompute --occupancy my_inputs/occupancy.npy --nodes my_inputs/nodes.json --output runs/prepared
```

`occupancy.npy` is a binary 2D NumPy array (`1=blocked`, `0=water`). `nodes.json` contains an `id` and `cells_row_col`, with the depot first and tasks following in ID order. See `docs/EXTERNAL_ASSETS.md` for policy and experiment contracts.

## Public contents

```text
src/llm_alns_b/runtime/     executable search rules with required external configuration
src/llm_alns_b/data.py      generic input validation, A* and path expansion
src/llm_alns_b/experiment.py
src/llm_alns_b/demo.py      independent synthetic illustration
docs/                      public interfaces and usage
tests/                     synthetic checks only
```

This edition has its own release manifest. It contains no private input identities or archived result hashes. Generated run files and private assets are excluded from version control by default. Do not put private inputs in the upload directory. No project-wide license has been assigned by this preparation.
