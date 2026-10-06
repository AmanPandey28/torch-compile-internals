# Data-dependent graph-break case

## Question

What happens when a Python branch depends on a runtime tensor value, and how
does expressing that branch with `torch.cond` change Dynamo capture?

| Check | Python `if` | `torch.cond` |
|---|---:|---:|
| Correct for positive input | True | True |
| Correct for negative input | True | True |
| Captured graphs across both paths | 3 | 1 |
| Accepted by `fullgraph=True` | no | yes |

The Python form fails full-graph capture with
`Data-dependent branching`. Dynamo captures a predicate graph
and specializes separate continuation graphs for the two Python branches. The
`torch.cond` form represents the branch inside one captured graph.

This is a structural result, not a performance claim. `torch.cond` avoids the
graph break, but its runtime benefit still depends on backend support, branch
cost, shapes, and hardware.
