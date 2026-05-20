# scm

This package provides a Set-Covering Machine (SCM) implementation optimized in C++ with a Python interface (via pybind11). It replaces the original NumPy/Numba code with a single-threaded C++ core.

## Installation

Install via pip (requires a C++ compiler and CMake):

```bash
pip install --no-build-isolation .
```

## Usage

```python
import numpy as np
from scm import SCMModel, fit

# Example data (dummy)
nodes_start = np.array([0, 1], dtype=np.uint64)
nodes_stop = np.array([1, 2], dtype=np.uint64)
kmers_assembly_idx = np.array([0, 1], dtype=np.uint16)
is_target = np.array([1, 0], dtype=np.uint8)

models = fit(
    nodes_start,
    nodes_stop,
    kmers_assembly_idx,
    is_target,
    max_rules=10,
    p=1.0,
    disjunction=True,
    top_n=3,
)
print(models)
```

`fit()` always returns `list[SCMModel]`, even when `top_n=1`.

`top_n` controls how many top models are returned. Returned models are ordered by the
solver's existing state ordering and are not deduplicated.

Each `SCMModel` includes `risk`, the solver's internal weighted training error:

```text
risk = p * transformed positives removed + transformed negatives remaining
```

For details, see the docstrings and source code.
