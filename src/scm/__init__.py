"""
SCM
===

Set Covering Machine (SCM) model fitting. It is designed to fit SCM models on Seqwin-derived arrays. 

Dependencies:
-------------
- numpy

Classes:
----------
- SCMModel

Functions:
----------
- fit
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._core import fit_native

@dataclass(frozen=True, slots=True)
class SCMModel:
    """Fitted Set Covering Machine model.

    Attributes:
        disjunction (bool): `True` for a disjunction model, `False` for a conjunction model. 
        nodes (NDArray[np.int64]): Indices of selected nodes (one per rule). 
        polarities (NDArray[np.uint8]): Rule polarity per selected node: `0` for presence, `1` for absence. 
        pred (NDArray[np.uint8]): Final per-assembly predictions as `0`/`1` labels (non-target/target). 
    """
    disjunction: bool
    nodes: NDArray[np.int64]
    polarities: NDArray[np.uint8]
    pred: NDArray[np.uint8]

def fit(
    nodes_start: NDArray[np.uint64], 
    nodes_stop: NDArray[np.uint64], 
    kmers_assembly_idx: NDArray[np.uint16], 
    is_target: NDArray[np.uint8], 
    max_rules: int, 
    p: float = 1.0, 
    disjunction: bool = True
) -> SCMModel:
    """Fit a Set Covering Machine model. 

    The input arrays are expected to come from Seqwin structures and use
    exact dtypes for direct transfer to the C++ backend. 

    Args:
        nodes_start (NDArray[np.uint64]): C-contiguous copy of `nodes['start']`. 
        nodes_stop (NDArray[np.uint64]): C-contiguous copy of `nodes['stop']`. 
        kmers_assembly_idx (NDArray[np.uint16]): C-contiguous copy of `kmers['assembly_idx']`. 
        is_target (NDArray[np.uint8]): `np.uint8` array converted from `assemblies['is_target']`. 
        max_rules (int): Maximum number of rules to include in the fitted model. 
        p (float, optional): Utility penalty for removed positive examples (typically >= 1.0). [1.0]
        disjunction (bool, optional): If `True`, fit a disjunction; if `False`, fit a conjunction. [True]

    Returns:
        SCMModel: Immutable model object containing selected rule nodes,
        polarities, and per-assembly predictions. 
    """
    dis, nodes, pol, pred = fit_native(
        nodes_start, nodes_stop, kmers_assembly_idx, is_target,
        max_rules, p, disjunction
    )
    return SCMModel(disjunction=dis, nodes=nodes, polarities=pol, pred=pred)
