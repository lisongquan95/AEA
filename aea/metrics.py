import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)


def classification_metrics(y, probabilities):
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1], zero_division=0)
    present = support > 0
    return {
        'n_samples': int(len(y)), 'accuracy': float(accuracy_score(y, predicted)),
        'balanced_accuracy': float(recall[present].mean()),
        'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
        'macro_f1': float(f1.mean()), 'drowsy_precision': float(precision[1]),
        'drowsy_recall': float(recall[1]), 'drowsy_f1': float(f1[1]),
        'roc_auc': float(roc_auc_score(y, probabilities[:, 1])) if present.all() else None,
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        'per_class': [{'label': i, 'precision': float(precision[i]), 'recall': float(recall[i]),
                       'f1': float(f1[i]), 'support': int(support[i])} for i in range(2)],
    }


def aggregate_metrics(folds):
    keys = ('accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
            'drowsy_precision', 'drowsy_recall', 'drowsy_f1', 'roc_auc')
    result = {}
    for key in keys:
        values = [f[key] for f in folds if f[key] is not None]
        result[key] = {'mean': float(np.mean(values)) if values else None,
                       'std_population': float(np.std(values, ddof=0)) if values else None,
                       'n_defined_folds': len(values)}
    return result
