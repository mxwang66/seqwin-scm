from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

# Import the compiled extension (built by pybind11)
from ._core import fit_native

@dataclass(frozen=True, slots=True)
class SCMModel:
    disjunction: bool
    nodes: NDArray[np.int64]
    polarities: NDArray[np.bool_]
    pred: NDArray[np.bool_]

def fit(
    node_start: NDArray[np.uint64],
    node_stop: NDArray[np.uint64],
    kmer_assembly_idx: NDArray[np.uint16],
    is_target: NDArray[np.bool_],
    max_rules: int,
    p: float = 1.0,
    disjunction: bool = True
) -> SCMModel:
    """
    Fit the SCM model. Inputs must be 1D NumPy arrays (C-contiguous) of the correct dtype.
    """
    dis, nodes, pol, pred = fit_native(
        node_start, node_stop, kmer_assembly_idx, is_target,
        max_rules, p, disjunction
    )
    return SCMModel(disjunction=dis, nodes=nodes, polarities=pol, pred=pred)
