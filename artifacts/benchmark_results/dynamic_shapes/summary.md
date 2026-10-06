# Dynamic-shape specialization analysis

This experiment varies only the sequence dimension of a gated transformer MLP
and runs each `torch.compile` policy in a separate process. A pass-through
backend counts Dynamo graph compilations without adding Inductor code-generation
effects.

| Mode | `dynamic` argument | Backend compilations | Guard-triggered recompiles |
|---|---:|---:|---:|
| Static | `False` | 3 | 2 |
| Automatic | `None` | 2 | 1 |
| Upfront dynamic | `True` | 1 | 0 |

| Call | Sequence length | Static | Automatic | Upfront dynamic |
|---:|---:|---:|---:|---:|
| 1 | 32 | compile G1 | compile G1 | compile G1 |
| 2 | 64 | compile G2 | compile G2 | cache reuse |
| 3 | 128 | compile G3 | cache reuse | cache reuse |
| 4 | 32 (repeat) | cache reuse | cache reuse | cache reuse |
| 5 | 64 (repeat) | cache reuse | cache reuse | cache reuse |

Static mode compiled 3 graphs for 3 unique sequence lengths and logged 2 guard-triggered recompiles. Automatic mode compiled 2 graphs; its final graph signature contains 1 symbolic integer input. Upfront dynamic mode introduced the symbolic dimension on its first capture and handled the full sequence with one graph.

Repeated lengths reuse cached graphs rather than compiling again.

All outputs matched eager execution. These counts describe Dynamo capture for
this workload and input sequence; they do not establish that a more dynamic
graph is faster. Dynamic kernels can trade fewer compilations for broader guards
or less specialized generated code.
