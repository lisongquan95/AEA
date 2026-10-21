import numpy as np
import pytest
from scipy.io import savemat
from aea.data import ChannelScaler, inner_splits, load_data
from aea.metrics import aggregate_metrics, classification_metrics


def test_subject_disjoint_partitions():
    ids = np.repeat([2, 7, 19, 23], 4)
    splits = list(inner_splits(ids, 19))
    assert [s[0] for s in splits] == [2, 7, 23]
    for _, train, val in splits:
        assert 19 not in ids[train] and 19 not in ids[val]
        assert set(ids[train]).isdisjoint(ids[val])
        assert len(train) + len(val) == 12


def test_scaler_training_statistics_only():
    source = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    scaler = ChannelScaler().fit(source)
    before = scaler.mean.copy()
    scaler.transform(np.full((3, 2, 3), 1e6, dtype=np.float32))
    np.testing.assert_equal(before, scaler.mean)
    np.testing.assert_allclose(scaler.transform(source).mean(axis=(0, 2)), 0, atol=1e-7)
    np.testing.assert_equal(ChannelScaler().fit(np.ones((2, 2, 3))).scale, 1)


def test_mat_npz_contract(tmp_path):
    arrays = dict(EEGsample=np.ones((6, 3, 16)), substate=np.array([0, 1] * 3)[:, None],
                  subindex=np.repeat([1, 4, 9], 2)[None, :])
    for suffix, save in [('.mat', savemat), ('.npz', lambda p, d: np.savez(p, **d))]:
        p = tmp_path / ('data' + suffix)
        save(p, arrays)
        data = load_data(p)
        assert data.x.shape == (6, 3, 16) and data.y.shape == (6,)
    p = tmp_path / 'bad.npz'
    for replacement in [dict(substate=np.array([1, 2] * 3)),
                        dict(EEGsample=np.ones((3, 16, 6))),
                        dict(subindex=np.array([1.5] * 6)),
                        dict(EEGsample=np.full((6, 3, 16), np.nan))]:
        np.savez(p, **{**arrays, **replacement})
        with pytest.raises(ValueError):
            load_data(p)


def test_metrics_definitions_and_missing_class():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([[.9, .1], [.4, .6], [.2, .8], [.3, .7]])
    m = classification_metrics(y, probabilities)
    assert m['accuracy'] == .75 and m['drowsy_recall'] == 1
    assert m['confusion_matrix'] == [[1, 1], [0, 2]]
    one = classification_metrics(np.zeros(2, dtype=int), probabilities[:2])
    assert one['roc_auc'] is None
    assert one['balanced_accuracy'] == .5
    assert aggregate_metrics([one])['roc_auc']['n_defined_folds'] == 0
