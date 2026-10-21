"""Nested LOSO selection and source-only refitting. Outer labels never enter selection."""
import csv
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import ChannelScaler, file_sha256, inner_splits, load_data
from .metrics import aggregate_metrics, classification_metrics
from .model import AEA_ICNN


DEFAULT_CANDIDATE = {'batch_size': 24, 'lr': 0.001, 'weight_decay': 0.0,
                     'N1': 16, 'd': 2, 'kernelLength': 64, 'attention_kernel': 3}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def seed_everything(seed, deterministic=True):
    # Must be set before CUDA initializes; unsupported deterministic operations fail loudly.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def derived_seed(seed, *parts):
    return int(np.random.SeedSequence([seed, *parts]).generate_state(1)[0])


def validate_candidates(candidates, time_points):
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('grid must be a nonempty JSON list of candidate objects')
    validated = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) - set(DEFAULT_CANDIDATE):
            raise ValueError(f'allowed candidate keys: {list(DEFAULT_CANDIDATE)}')
        c = {**DEFAULT_CANDIDATE, **candidate}
        for key in ('batch_size', 'N1', 'd', 'kernelLength', 'attention_kernel'):
            if type(c[key]) is not int or c[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('lr', 'weight_decay'):
            if type(c[key]) not in (int, float) or not np.isfinite(c[key]):
                raise ValueError(f'{key} must be finite numeric')
        if c['lr'] <= 0 or c['weight_decay'] < 0:
            raise ValueError('lr must be positive and weight_decay nonnegative')
        if c['kernelLength'] >= time_points or c['attention_kernel'] % 2 == 0:
            raise ValueError('kernelLength must be < T (safe singleton batches); attention_kernel must be odd')
        validated.append(c)
    return validated


def make_loader(x, y, batch_size, shuffle=False, seed=0):
    dataset = TensorDataset(torch.from_numpy(np.ascontiguousarray(x[:, None])), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=0, generator=generator)


def predict(model, x, batch_size, device):
    model.eval()
    probabilities = []
    # No target labels are needed by the prediction path.
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(x[:, None]))),
                                   batch_size=batch_size, shuffle=False, num_workers=0):
            probabilities.append(model(batch.to(device)).exp().cpu().numpy())
    return np.concatenate(probabilities)


def fit_model(train_x, train_y, candidate, model_kwargs, seed, epochs, patience,
              selection_metric, device, eval_batch_size, normalization,
              deterministic=True, validation=None, schedule_epochs=None):
    if set(np.unique(train_y)) != {0, 1}:
        raise ValueError('each training partition must contain both classes; change subject protocol explicitly')
    seed_everything(seed, deterministic)
    scaler = ChannelScaler(normalization).fit(train_x)
    train_x = scaler.transform(train_x)
    val_x = scaler.transform(validation[0]) if validation is not None else None
    model = AEA_ICNN(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate['lr'], weight_decay=candidate['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_epochs or epochs)
    loader = make_loader(train_x, train_y, candidate['batch_size'], shuffle=True, seed=seed)
    loss_fn = nn.NLLLoss()
    best_score, best_epoch, stale = -float('inf'), 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        record = {'epoch': epoch, 'train_nll': total_loss / len(train_y),
                  'lr': optimizer.param_groups[0]['lr']}
        scheduler.step()
        if validation is not None:
            metrics = classification_metrics(validation[1], predict(model, val_x, eval_batch_size, device))
            score = metrics[selection_metric]
            record['validation'] = metrics
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
        history.append(record)
        if validation is not None and stale >= patience:
            break
    return model, scaler, history, best_score, best_epoch


def model_options(candidate, channels, time_points, batchnorm_mode):
    return {'classes': 2, 'sampleChannel': channels, 'sampleLength': time_points,
            **{k: candidate[k] for k in ('N1', 'd', 'kernelLength', 'attention_kernel')},
            'batchnorm_mode': batchnorm_mode}


def select_on_sources(x, y, subjects, candidates, *, seed, epochs, patience,
                      selection_metric, device, eval_batch_size, normalization,
                      deterministic, batchnorm_mode, inner_folds):
    """Receives source arrays ONLY; test samples/labels are absent by construction."""
    records = []
    for candidate_index, candidate in enumerate(candidates):
        folds = []
        # Sentinel outside the source ID set: all source subjects remain eligible.
        for inner_index, (held_out, train_idx, val_idx) in enumerate(inner_splits(subjects, None, inner_folds)):
            run_seed = derived_seed(seed, 1, inner_index)  # same initialization seed across candidates
            _, _, history, score, best_epoch = fit_model(
                x[train_idx], y[train_idx], candidate,
                model_options(candidate, x.shape[1], x.shape[2], batchnorm_mode), run_seed,
                epochs, patience, selection_metric, device, eval_batch_size, normalization,
                deterministic, validation=(x[val_idx], y[val_idx]))
            folds.append({'validation_subject': held_out,
                          'training_subjects': np.unique(subjects[train_idx]).tolist(),
                          'seed': run_seed, 'best_epoch': best_epoch, 'best_score': score,
                          'history': history})
        records.append({'candidate_index': candidate_index, 'parameters': candidate,
                        'mean_validation_score': float(np.mean([f['best_score'] for f in folds])),
                        'inner_folds': folds})
    # Stable candidate-order tie break, no test scores consulted.
    selected = max(range(len(records)), key=lambda i: records[i]['mean_validation_score'])
    best_epochs = [f['best_epoch'] for f in records[selected]['inner_folds']]
    refit_epochs = max(1, int(np.floor(np.median(best_epochs) + 0.5)))
    return records, selected, refit_epochs


def save_predictions(path, indices, subject_ids, y, probabilities):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_index', 'subject_id', 'y_true', 'y_pred', 'prob_alert', 'prob_drowsy'])
        for idx, subject, label, prob in zip(indices, subject_ids, y, probabilities):
            writer.writerow([int(idx), int(subject), int(label), int(prob.argmax()), float(prob[0]), float(prob[1])])


def run_loso(args):
    if args.epochs < 1 or args.patience < 1 or args.eval_batch_size < 1 or args.inner_folds < 0:
        raise ValueError('epochs, patience, eval batch size must be positive; inner folds >= 0')
    if not 0 <= args.seed < 2**32:
        raise ValueError('seed must be in [0, 2**32)')
    data = load_data(args.data)
    candidates = validate_candidates(json.loads(Path(args.grid).read_text()) if args.grid else [{}], data.x.shape[2])
    subjects = np.unique(data.subjects)
    test_subjects = args.test_subjects or subjects.tolist()
    if len(set(test_subjects)) != len(test_subjects) or not set(test_subjects) <= set(subjects):
        raise ValueError('test subjects must be distinct IDs present in the data')
    if args.inner_folds > len(subjects) - 1:
        raise ValueError('--inner-folds exceeds the number of source subjects')
    device = torch.device(args.device)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty: never overwrite an experiment')
    output.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        git_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_dirty = None, None
    data_hash = file_sha256(args.data)
    manifest = {'status': 'running', 'arguments': vars(args), 'candidates': candidates,
                'data_sha256': data_hash, 'data_shape': list(data.x.shape),
                'subjects': subjects.tolist(), 'test_subjects': test_subjects,
                'complete_loso': set(test_subjects) == set(subjects),
                'protocol': 'nested_loso_source_only_refit', 'git_commit': git_commit, 'git_dirty': git_dirty,
                'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                             'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
                             'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()},
                'device': str(device), 'torch_threads': torch.get_num_threads()}
    write_json(output / 'manifest.json', manifest)
    fold_metrics, all_indices, all_probabilities = [], [], []
    for test_subject in test_subjects:
        fold_dir = output / f'subject_{test_subject}'
        fold_dir.mkdir()
        source_idx = np.flatnonzero(data.subjects != test_subject)
        test_idx = np.flatnonzero(data.subjects == test_subject)
        fold_seed = derived_seed(args.seed, 0, int(np.flatnonzero(subjects == test_subject)[0]))
        print(f'Outer test subject {test_subject}: selecting on {len(subjects)-1} source subjects', flush=True)
        records, selected, refit_epochs = select_on_sources(
            data.x[source_idx], data.y[source_idx], data.subjects[source_idx], candidates,
            seed=fold_seed, epochs=args.epochs, patience=args.patience,
            selection_metric=args.selection_metric, device=device, eval_batch_size=args.eval_batch_size,
            normalization=args.normalization, deterministic=not args.allow_nondeterministic,
            batchnorm_mode=args.batchnorm_mode, inner_folds=args.inner_folds)
        candidate = candidates[selected]
        selection = {'outer_test_subject': test_subject, 'source_subjects': np.unique(data.subjects[source_idx]).tolist(),
                     'source_indices': source_idx.tolist(), 'test_indices': test_idx.tolist(),
                     'selection_metric': args.selection_metric, 'selected_candidate': selected,
                     'refit_epochs': refit_epochs, 'candidates': records}
        write_json(fold_dir / 'selection.json', selection)
        kwargs = model_options(candidate, data.x.shape[1], data.x.shape[2], args.batchnorm_mode)
        refit_seed = derived_seed(fold_seed, 2)
        model, scaler, history, _, _ = fit_model(
            data.x[source_idx], data.y[source_idx], candidate, kwargs, refit_seed,
            refit_epochs, args.patience, args.selection_metric, device, args.eval_batch_size,
            args.normalization, not args.allow_nondeterministic, schedule_epochs=args.epochs)
        write_json(fold_dir / 'refit_history.json', history)
        # Final source-only model: no test-time fitting or checkpoint selection.
        torch.save({'format_version': 1, 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'model_kwargs': kwargs, 'candidate': candidate, 'refit_epochs': refit_epochs,
                    'seed': refit_seed, 'test_subject': int(test_subject), 'data_sha256': data_hash,
                    'normalization': args.normalization, 'mean': torch.from_numpy(scaler.mean),
                    'scale': torch.from_numpy(scaler.scale), 'eval_batch_size': args.eval_batch_size},
                   fold_dir / 'model.pt')
        probabilities = predict(model, scaler.transform(data.x[test_idx]), args.eval_batch_size, device)
        metrics = classification_metrics(data.y[test_idx], probabilities)
        write_json(fold_dir / 'metrics.json', metrics)
        save_predictions(fold_dir / 'predictions.csv', test_idx, data.subjects[test_idx], data.y[test_idx], probabilities)
        fold_metrics.append({'subject_id': int(test_subject), **metrics})
        all_indices.extend(test_idx.tolist())
        all_probabilities.append(probabilities)
        write_json(output / 'fold_metrics.json', fold_metrics)
    indices = np.asarray(all_indices)
    probs = np.concatenate(all_probabilities)
    order = np.argsort(indices)
    save_predictions(output / 'predictions.csv', indices[order], data.subjects[indices[order]],
                     data.y[indices[order]], probs[order])
    summary = {'complete_loso': manifest['complete_loso'], 'n_test_subjects': len(fold_metrics),
               'subject_macro': aggregate_metrics(fold_metrics),
               'pooled_samples': classification_metrics(data.y[indices], probs)}
    write_json(output / 'summary.json', summary)
    with (output / 'fold_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['subject_id', 'n_samples', 'accuracy', 'balanced_accuracy', 'macro_precision',
                  'macro_recall', 'macro_f1', 'drowsy_precision', 'drowsy_recall', 'drowsy_f1', 'roc_auc']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(fold_metrics)
    manifest['status'] = 'completed'
    write_json(output / 'manifest.json', manifest)
    print(f'Completed {len(fold_metrics)} folds; results: {output}', flush=True)
    return summary
