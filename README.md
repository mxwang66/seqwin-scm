# seqwin_scm

This package provides a Set-Covering Machine (SCM) implementation optimized in C++ with a Python interface (via pybind11).  It replaces the original NumPy/Numba code with a single-threaded C++ core. 

## Installation

Install via pip (requires a C++ compiler and CMake):

```bash
pip install .
```

## Usage

```python
import numpy as np
from seqwin_scm import SCMModel, fit

# Example data (dummy)
node_start = np.array([0, 1], dtype=np.uint64)
node_stop = np.array([1, 2], dtype=np.uint64)
kmer_assembly_idx = np.array([0, 1], dtype=np.uint16)
is_target = np.array([True, False], dtype=np.bool_)

model = fit(node_start, node_stop, kmer_assembly_idx, is_target, max_rules=10, p=1.0, disjunction=True)
print(model)
```

For details, see the docstrings and source code.
