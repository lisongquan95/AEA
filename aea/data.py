"""Explicit sample-first data contract. No subject-wise/global pre-normalization."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from scipy.io import loadmat


@dataclass
class EEGData:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_data(path):
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as source:
            arrays = {k: source[k] for k in ('EEGsample', 'substate', 'subindex')}
    elif path.suffix.lower() == '.mat':
        try:
            arrays = loadmat(path)
        except NotImplementedError as exc:
            raise ValueError('MAT v7.3 is unsupported; convert explicitly to sample-first NPZ') from exc
    else:
        raise ValueError('data must be .mat (pre-v7.3) or .npz')
    x = np.asarray(arrays['EEGsample'])
    y = np.asarray(arrays['substate']).reshape(-1)
    subjects = np.asarray(arrays['subindex']).reshape(-1)
    if x.ndim != 3 or x.shape[0] != len(y) or len(y) != len(subjects) or min(x.shape) < 1:
        raise ValueError('expected EEGsample (N,C,T) and N labels/subject IDs; axes are never guessed')
    for name, values in [('EEGsample', x), ('substate', y), ('subindex', subjects)]:
        if values.dtype.kind not in 'biuf' or not np.isfinite(values).all():
            raise ValueError(f'{name} must contain finite real numeric values')
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('substate must contain both 0=alert and 1=drowsy; no automatic relabeling')
    if not np.equal(subjects, np.floor(subjects)).all() or np.any(np.abs(subjects) >= 2**53):
        raise ValueError('subindex must contain exactly represented integer subject IDs')
    if len(np.unique(subjects)) < 3:
        raise ValueError('nested subject-disjoint LOSO requires at least 3 subjects')
    x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('EEGsample overflows float32')
    return EEGData(x, y.astype(np.int64), subjects.astype(np.int64))


class ChannelScaler:
    def __init__(self, mode='channel'):
        if mode not in ('none', 'channel'):
            raise ValueError('normalization must be none or channel')
        self.mode = mode

    def fit(self, x):
        if self.mode == 'channel':
            self.mean = x.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
            self.scale = x.std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
            self.scale[self.scale < 1e-6] = 1.0
        else:
            self.mean = np.zeros((1, x.shape[1], 1), dtype=np.float32)
            self.scale = np.ones_like(self.mean)
        return self

    def transform(self, x):
        result = ((x - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('non-finite normalized data')
        return result


def inner_splits(subjects, outer_test, limit=0):
    eligible = np.ones(len(subjects), dtype=bool) if outer_test is None else subjects != outer_test
    source = np.unique(subjects[eligible])
    # IDs only: no label-dependent validation subject selection.
    for held_out in source[:limit or len(source)]:
        train = np.flatnonzero(eligible & (subjects != held_out))
        val = np.flatnonzero(subjects == held_out)
        assert not set(subjects[train]) & set(subjects[val])
        yield int(held_out), train, val
