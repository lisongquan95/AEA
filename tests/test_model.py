import json
from pathlib import Path
import pytest
import torch
from aea.model import AEA, AEA_ICNN


def notebook_classes():
    namespace = {'torch': torch, 'nn': torch.nn}
    cells = json.loads((Path(__file__).parents[1] / 'AEA.ipynb').read_text())['cells']
    for cell in cells:
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'code' and source.lstrip().startswith('class '):
            exec(source, namespace)
    return namespace


def test_notebook_effective_forward_and_gradients():
    torch.set_num_threads(1)
    old = notebook_classes()['AEA_ICNN']()
    new = AEA_ICNN(batchnorm_mode='batch')
    active = {k: v for k, v in old.state_dict().items() if 'temporal_fc' not in k}
    new.load_state_dict(active, strict=True)
    x1 = torch.randn(3, 1, 30, 384, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    out1, out2 = old(x1), new(x2)
    torch.testing.assert_close(out1, out2)
    (-out1[:, 0].sum()).backward()
    (-out2[:, 0].sum()).backward()
    torch.testing.assert_close(x1.grad, x2.grad)
    for name, parameter in new.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(old.named_parameters())[name].grad)
    assert old.AEA.temporal_fc.weight.grad is None


def test_running_bn_eval_is_batch_independent():
    model = AEA_ICNN(sampleChannel=3, sampleLength=16, N1=2, d=1, kernelLength=4)
    model(torch.randn(4, 1, 3, 16))  # populate training statistics
    model.eval()
    x = torch.randn(4, 1, 3, 16)
    with torch.no_grad():
        torch.testing.assert_close(model(x[:1]), model(x)[:1])
    weights = model.AEA(x)
    assert weights.shape == (4, 1, 3, 1)
    assert ((weights >= 0) & (weights <= 2)).all()
    assert model.depthwise.groups == 2


@pytest.mark.parametrize('kernel', [1, 3, 5])
def test_attention_shape(kernel):
    assert AEA(E_channels=7, i=kernel)(torch.randn(2, 1, 7, 19)).shape == (2, 1, 7, 1)


def test_bad_dimensions():
    with pytest.raises(ValueError):
        AEA(i=2)
    with pytest.raises(ValueError):
        AEA_ICNN()(torch.randn(2, 1, 29, 384))
