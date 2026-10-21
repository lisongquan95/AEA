import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from aea.experiment import run_loso, validate_candidates
from evaluate import evaluate
from train import parser


@pytest.fixture
def synthetic(tmp_path):
    rng = np.random.default_rng(123)
    x = rng.normal(size=(24, 3, 16)).astype(np.float32)
    y = np.tile([0, 1, 0, 1, 0, 1], 4)
    subjects = np.repeat([2, 7, 19, 23], 6)
    path = tmp_path / 'synthetic.npz'
    np.savez(path, EEGsample=x, substate=y, subindex=subjects)
    grid = tmp_path / 'grid.json'
    grid.write_text(json.dumps([{'N1': 2, 'd': 1, 'kernelLength': 4, 'batch_size': 5},
                                {'N1': 2, 'd': 1, 'kernelLength': 4, 'batch_size': 8}]))
    return path, grid, x, y, subjects


def arguments(path, grid, output, *extra):
    return parser().parse_args(['--data', str(path), '--grid', str(grid), '--output', str(output),
                               '--epochs', '2', '--patience', '1', *extra])


def test_full_nested_loso_cli_exports_and_checkpoint(synthetic, tmp_path):
    path, grid, _, _, _ = synthetic
    output = tmp_path / 'full'
    command = [sys.executable, 'train.py', '--data', str(path), '--grid', str(grid),
               '--output', str(output), '--epochs', '2', '--patience', '1']
    subprocess.run(command, cwd=Path(__file__).parents[1], check=True, capture_output=True, text=True)
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['complete_loso'] and summary['n_test_subjects'] == 4
    assert summary['pooled_samples']['n_samples'] == 24
    assert summary['subject_macro']['accuracy']['n_defined_folds'] == 4
    rows = (output / 'predictions.csv').read_text().splitlines()
    assert len(rows) == 25
    assert [int(row.split(',')[0]) for row in rows[1:]] == list(range(24))
    for subject in [2, 7, 19, 23]:
        selection = json.loads((output / f'subject_{subject}/selection.json').read_text())
        assert subject not in selection['source_subjects']
        assert set(selection['source_indices']).isdisjoint(selection['test_indices'])
        for candidate in selection['candidates']:
            assert len(candidate['inner_folds']) == 3
            for fold in candidate['inner_folds']:
                assert subject not in fold['training_subjects']
                assert fold['validation_subject'] not in fold['training_subjects']
                assert fold['validation_subject'] != subject
    reevaluation = tmp_path / 'reevaluation'
    subprocess.run([sys.executable, 'evaluate.py', '--checkpoint', str(output / 'subject_2/model.pt'),
                    '--data', str(path), '--output', str(reevaluation)],
                   cwd=Path(__file__).parents[1], check=True, capture_output=True)
    assert (reevaluation / 'predictions.csv').read_bytes() == (output / 'subject_2/predictions.csv').read_bytes()
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'completed'
    with pytest.raises(ValueError, match='empty'):
        run_loso(arguments(path, grid, output))


def test_test_labels_and_signals_cannot_change_selection_or_training(synthetic, tmp_path):
    torch.set_num_threads(1)
    path, grid, x, y, subjects = synthetic
    for name, change in [('original', False), ('changed', True), ('repeat', False)]:
        p = path
        if change:
            p = tmp_path / 'changed.npz'
            changed_y, changed_x = y.copy(), x.copy()
            changed_y[subjects == 2] = 1 - changed_y[subjects == 2]
            changed_x[subjects == 2] *= 100
            np.savez(p, EEGsample=changed_x, substate=changed_y, subindex=subjects)
        run_loso(arguments(p, grid, tmp_path / name, '--test-subjects', '2'))
    original = tmp_path / 'original/subject_2'
    for name in ['changed', 'repeat']:
        other = tmp_path / name / 'subject_2'
        assert (original / 'selection.json').read_bytes() == (other / 'selection.json').read_bytes()
        a = torch.load(original / 'model.pt', weights_only=True)
        b = torch.load(other / 'model.pt', weights_only=True)
        for k in a['state_dict']:
            assert torch.equal(a['state_dict'][k], b['state_dict'][k]), k
        assert torch.equal(a['mean'], b['mean'])
        assert torch.equal(a['scale'], b['scale'])
        if name == 'repeat':
            assert (original / 'predictions.csv').read_bytes() == (other / 'predictions.csv').read_bytes()
    with pytest.raises(ValueError, match='hash'):
        evaluate(original / 'model.pt', tmp_path / 'changed.npz', tmp_path / 'wrong')


def test_candidate_validation():
    for grid in [[], [{'lr': -1}], [{'batch_size': 0}], [{'kernelLength': 100}],
                 [{'unknown': 1}], [{'attention_kernel': 2}], [{'lr': float('nan')}]]:
        with pytest.raises(ValueError):
            validate_candidates(grid, 80)
