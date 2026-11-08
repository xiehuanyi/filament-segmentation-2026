# Analysis scripts (not part of the final pipelines)

| script | what it computes |
|---|---|
| `oracle.py` | upper bound of the selection stage: keep candidates by their true q, per level and over all levels |
| `noise_ceiling.py` | agreement between two annotators on the q of the same candidate (how predictable q is) |
| `human_ceiling.py` | human baselines (another annotator, intersection / union of two) vs our OOF predictions on 3-record frames |

Other development experiments (dropped model variants, selector variants) are not included.
