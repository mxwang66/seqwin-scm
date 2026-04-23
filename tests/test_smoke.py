import numpy as np
import pytest
from scm import SCMModel, fit

def test_smoke():
    # Simple example with 2 assemblies and 1 node
    node_start = np.array([0], dtype=np.uint64)
    node_stop = np.array([2], dtype=np.uint64)
    # Two kmers both in assembly 0 and 1
    kmer_assembly_idx = np.array([0, 1], dtype=np.uint16)
    is_target = np.array([1, 0], dtype=np.uint8)

    model = fit(node_start, node_stop, kmer_assembly_idx, is_target, max_rules=5, p=1.0, disjunction=True)
    
    # Check return type and structure
    assert isinstance(model, SCMModel)
    assert isinstance(model.disjunction, (bool, np.bool_))
    assert isinstance(model.nodes, np.ndarray) and model.nodes.dtype == np.int64
    assert isinstance(model.polarities, np.ndarray) and model.polarities.dtype == np.uint8
    assert isinstance(model.pred, np.ndarray) and model.pred.dtype == np.uint8

    # Check array shapes match expectations (nodes and polarities should have same length)
    assert model.nodes.shape == model.polarities.shape
    assert model.pred.shape[0] == is_target.shape[0]
