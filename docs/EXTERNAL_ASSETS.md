# External experiment contracts

Private assets live outside the public checkout. Relative references in an experiment file resolve against that file's directory.

## Experiment manifest

```json
{
  "schema": "llm-alns-b.experiment.v1",
  "parameters": "parameters.json",
  "data_manifest": "dataset/manifest.json",
  "policy_manifest": "policies/manifest.json"
}
```

An optional `pairs` array declares objects containing `instance` and integer `seed`. Without it, the dataset's declared instance/seed panel is used. Duplicate pairs are rejected. The actual research instance IDs, seeds and panel membership are private inputs.

## Parameters

`runtime/config.py::SearchConfig` defines the strict required schema. It has no research defaults. The fields are:

- Iteration budget, RNG namespace, destroy/repair slot counts and initial weights.
- Uniform-selection mixing, weight reaction and rewards for best improvement, parent improvement, accepted changed route and other outcomes.
- Initial temperature fraction, multiplicative decay, minimum temperature and acceptance-exponent floor.
- Recent-statistics window and objective-comparison tolerances.
- Allowed Q alpha levels and the removal fraction.

`src/llm_alns_b/demo_parameters.json` is a complete synthetic example, independent of the research settings. Use it to understand the schema, not to claim paper reproduction. Missing and unknown fields are rejected.

## Dataset

Schema: `llm-alns-b.dataset.v1`. Each instance declares an ID, directory, occupancy path and file hashes. Its directory contains:

- `nodes.json`: ID, depot-first ordered `cells_row_col`.
- `node_cells.npy`: integer coordinate array.
- `distance_matrix.npy`: finite symmetric float64 costs with zero diagonal.
- `shortest_paths.json.gz`: precomputed undirected node-pair grid paths.

`data.py::precompute` writes this format from user-supplied occupancy and nodes. Its matrices are independently checked and become read-only at load time. The manifest can include solver seeds for an experiment panel. Dataset paths cannot escape the manifest directory.

## Frozen policy

Schema: `llm-alns-b.policy.v1`. Entries `D` and `Q` each contain a relative `file` and its `sha256`. Hash values and policy code are supplied privately; none of the paper's trained-policy hashes are included in this release.

D signature: `destroy(route, q, dist, state, rng)` returns `(kept, removed)`. Removed tasks must be unique, exclude the depot, and contain exactly q tasks; kept tasks retain their relative order.

Q signature: `degree(state, rng)` returns one of the externally declared alpha values. The host maps alpha to `floor(removal_fraction * N * alpha + 0.5)`, clamped to the valid removal-count interval. All mapping parameters are external inputs.

Policy sources are interpreted by the bounded AST interpreter. Arbitrary Python imports, attributes, loops and execution are rejected. Trusted primitive arguments controlling selection behavior must be supplied explicitly.

## Public versus private evidence

The public validation report describes synthetic functional checks and whether a local private-asset compatibility check passed. It discloses neither private objective values nor private input signatures. Full result traces and parameter snapshots are generated only in the selected local output directory.
