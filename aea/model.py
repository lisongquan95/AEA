"""AEA's effective notebook computation, with explicit normalization semantics."""
import torch
from torch import nn


class AEA(nn.Module):
    def __init__(self, E_channels=30, i=3):
        super().__init__()
        if E_channels < 1 or i < 1 or i % 2 == 0:
            raise ValueError('E_channels must be positive and i must be positive and odd')
        self.E_channels = E_channels
        self.fc1 = nn.Conv2d(1, 32, (i, 1), padding=(i // 2, 0))
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(32, 1, (i, 1), padding=(i // 2, 0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self.E_channels:
            raise ValueError('AEA expects (batch, 1, electrodes, time)')
        amplitude_range = x.amax(dim=-1, keepdim=True) - x.amin(dim=-1, keepdim=True)
        # In the notebook, avg/max pooling on (..., electrodes, 1) are identities.
        # Both branches share weights: their sum is exactly 2 * sigmoid(...).
        return 2 * self.sigmoid(self.fc2(self.relu(self.fc1(amplitude_range))))


class AEA_ICNN(nn.Module):
    def __init__(self, classes=2, sampleChannel=30, sampleLength=384,
                 N1=16, d=2, kernelLength=64, attention_kernel=3,
                 batchnorm_mode='running'):
        super().__init__()
        if min(classes, sampleChannel, sampleLength, N1, d, kernelLength) < 1:
            raise ValueError('model dimensions must be positive')
        if kernelLength > sampleLength:
            raise ValueError('kernelLength must not exceed sampleLength')
        if batchnorm_mode not in ('running', 'batch'):
            raise ValueError('batchnorm_mode must be running or batch')
        self.sampleChannel, self.sampleLength = sampleChannel, sampleLength
        self.AEA = AEA(sampleChannel, attention_kernel)
        self.pointwise = nn.Conv2d(1, N1, (sampleChannel, 1))
        self.depthwise = nn.Conv2d(N1, N1 * d, (1, kernelLength), groups=N1)
        self.activ = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(N1 * d, track_running_stats=batchnorm_mode == 'running')
        self.GAP = nn.AvgPool2d((1, sampleLength - kernelLength + 1))
        self.fc = nn.Linear(N1 * d, classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputdata):
        if inputdata.ndim != 4 or tuple(inputdata.shape[1:]) != (1, self.sampleChannel, self.sampleLength):
            raise ValueError(f'expected (batch, 1, {self.sampleChannel}, {self.sampleLength})')
        x = self.pointwise(inputdata * self.AEA(inputdata))
        x = self.batchnorm(self.activ(self.depthwise(x)))
        return self.softmax(self.fc(self.GAP(x).flatten(1)))
