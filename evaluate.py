"""Re-evaluate a saved fold on its original held-out subject, without training."""
import argparse
from pathlib import Path
import torch
from aea.data import load_data, file_sha256, ChannelScaler
from aea.experiment import predict, save_predictions, write_json, seed_everything
from aea.metrics import classification_metrics
from aea.model import AEA_ICNN


def evaluate(checkpoint, data_path, output, device='cpu'):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['format_version'] != 1:
        raise ValueError('unsupported checkpoint format')
    if file_sha256(data_path) != saved['data_sha256']:
        raise ValueError('data hash differs from training input; original fold evaluation requires the identical file')
    data = load_data(data_path)
    indices = (data.subjects == saved['test_subject']).nonzero()[0]
    if not len(indices):
        raise ValueError('held-out subject is absent')
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('output directory must be empty')
    destination.mkdir(parents=True, exist_ok=True)
    seed_everything(saved['seed'])
    model = AEA_ICNN(**saved['model_kwargs']).to(device)
    model.load_state_dict(saved['state_dict'])
    scaler = ChannelScaler(saved['normalization'])
    scaler.mean, scaler.scale = saved['mean'].numpy(), saved['scale'].numpy()
    probs = predict(model, scaler.transform(data.x[indices]), saved['eval_batch_size'], device)
    metrics = classification_metrics(data.y[indices], probs)
    write_json(destination / 'metrics.json', metrics)
    save_predictions(destination / 'predictions.csv', indices, data.subjects[indices], data.y[indices], probs)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    torch.set_num_threads(args.threads)
    evaluate(args.checkpoint, args.data, args.output, args.device)
