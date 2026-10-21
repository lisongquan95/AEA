"""Command-line entry point: python train.py --help."""
import argparse
from aea.experiment import run_loso


def parser():
    p = argparse.ArgumentParser(description='Source-only nested LOSO EEG training and evaluation')
    p.add_argument('--data', required=True, help='MAT/NPZ with EEGsample, substate, subindex')
    p.add_argument('--output', required=True, help='new or empty output directory')
    p.add_argument('--grid', help='JSON list of candidate parameter dictionaries')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=26)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu', help='cpu, cuda, cuda:0, ...')
    p.add_argument('--eval-batch-size', type=int, default=128)
    p.add_argument('--inner-folds', type=int, default=0, help='0: full source LOSO; K: first K source IDs')
    p.add_argument('--test-subjects', nargs='+', type=int, help='optional partial run; defaults to every subject')
    p.add_argument('--selection-metric', choices=['balanced_accuracy', 'macro_f1', 'accuracy'], default='balanced_accuracy')
    p.add_argument('--normalization', choices=['channel', 'none'], default='channel')
    p.add_argument('--batchnorm-mode', choices=['running', 'batch'], default='running',
                   help='running: strict inductive default; batch: notebook batch-dependent behavior')
    p.add_argument('--allow-nondeterministic', action='store_true')
    p.add_argument('--threads', type=int, default=1, help='PyTorch CPU threads; recorded for repeatability')
    return p


def main():
    import torch
    args = parser().parse_args()
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    torch.set_num_threads(args.threads)
    run_loso(args)


if __name__ == '__main__':
    main()
