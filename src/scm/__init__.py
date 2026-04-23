"""
SCM
===

Fit set covering machine (SCM) models on Seqwin data structures (`kmers` and `nodes`).  

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
    """The SCM model class. 

    Attributes:
        disjunction (str): True for a disjunction model. False for a conjunction model. 
        nodes (NDArray[np.int64]): Indices of the selected nodes. 
        polarities (NDArray[np.uint8]): SCM rule type of each selected node. 0 for presence, 1 for absence. 
        pred (NDArray[np.uint8]): Predictions of the Seqwin assemblies. 0 for non-targets, 1 for targets. 
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
    """Fit the SCM model. Rules are labeled with node indices

    Args:
        nodes_start (NDArray[np.uint64]): C-contiguous copy of `nodes['start']`. 
        nodes_stop (NDArray[np.uint64]): C-contiguous copy of `nodes['stop']`. 
        kmers_assembly_idx (NDArray[np.uint16]): C-contiguous copy of `kmers['assembly_idx']`. 
        is_target (NDArray[np.uint8]): `np.uint8` array converted from `assemblies['is_target']`. 
        max_rules (int): Maximum number of rules for the SCM model. 
        p (float, optional): SCM hyperparameter (penalty for the utility function). Should be no less than 1. [1.0]
        disjunction (bool, optional): True to fit a disjunction model, False to fit a conjunction model. [True]

    Returns:
        SCMModel: The SCM model. Only works for the provided Seqwin data. 
    """
    dis, nodes, pol, pred = fit_native(
        nodes_start, nodes_stop, kmers_assembly_idx, is_target,
        max_rules, p, disjunction
    )
    return SCMModel(disjunction=dis, nodes=nodes, polarities=pol, pred=pred)
