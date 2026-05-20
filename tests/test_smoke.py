import numpy as np
from scm import SCMModel, fit


def _toy_data():
    nodes_start = np.array([0, 2], dtype=np.uint64)
    nodes_stop = np.array([2, 4], dtype=np.uint64)
    kmers_assembly_idx = np.array([0, 1, 1, 2], dtype=np.uint16)
    is_target = np.array([1, 0, 1], dtype=np.uint8)
    return nodes_start, nodes_stop, kmers_assembly_idx, is_target


def test_fit_returns_list_default_top1():
    nodes_start, nodes_stop, kmers_assembly_idx, is_target = _toy_data()

    models = fit(nodes_start, nodes_stop, kmers_assembly_idx, is_target, max_rules=5, p=1.0, disjunction=True)

    assert isinstance(models, list)
    assert len(models) == 1

    model = models[0]
    assert isinstance(model, SCMModel)
    assert isinstance(model.disjunction, (bool, np.bool_))
    assert isinstance(model.nodes, np.ndarray) and model.nodes.dtype == np.int64
    assert isinstance(model.polarities, np.ndarray) and model.polarities.dtype == np.uint8
    assert isinstance(model.pred, np.ndarray) and model.pred.dtype == np.uint8
    assert hasattr(model, "risk")
    assert isinstance(float(model.risk), float)

    assert model.nodes.shape == model.polarities.shape
    assert model.pred.shape[0] == is_target.shape[0]


def test_fit_top_n_sorted_by_risk_non_decreasing():
    nodes_start, nodes_stop, kmers_assembly_idx, is_target = _toy_data()

    models = fit(
        nodes_start,
        nodes_stop,
        kmers_assembly_idx,
        is_target,
        max_rules=5,
        p=1.0,
        disjunction=True,
        beam_width=3,
        branch_width=3,
        top_n=2,
    )

    assert isinstance(models, list)
    assert 0 < len(models) <= 2
    assert all(isinstance(model, SCMModel) for model in models)

    risks = [float(model.risk) for model in models]
    assert risks == sorted(risks)

    for model in models:
        assert model.nodes.shape == model.polarities.shape
        assert model.pred.shape[0] == is_target.shape[0]
