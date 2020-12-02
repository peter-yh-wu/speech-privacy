"""
Model definition
"""

import random
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


class VGGVoxClf(nn.Module):
    """Based on https://github.com/zimmerrol/vggvox-pytorch"""
    def __init__(self, nOut=1024, log_input=True):
        super(VGGVoxClf, self).__init__()
        self.log_input = log_input

        self.netcnn = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(5,7), stride=(1,2), padding=(2,2)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,2)),

            nn.Conv2d(96, 256, kernel_size=(5,5), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

            nn.Conv2d(256, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

            nn.Conv2d(256, 512, kernel_size=(4,1), padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.encoder = nn.AdaptiveMaxPool2d((1,1))
        out_dim = 512

        self.fc = nn.Linear(out_dim, nOut)
        self.clf = nn.Sequential(nn.ReLU(), nn.Linear(nOut, 2))

        self.instancenorm   = nn.InstanceNorm1d(40)
        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40)
        
    def forward(self, x):
        x = self.torchfb(x)+1e-6
        if self.log_input: x = x.log()
        x = self.instancenorm(x).unsqueeze(1).detach()
        x = self.netcnn(x)
        x = self.encoder(x)
        x = x.view((x.size()[0], -1))
        x = self.fc(x)
        x = self.clf(x)
        return x

    def forward_emb(self, x):
        x = self.torchfb(x)+1e-6
        if self.log_input: x = x.log()
        x = self.instancenorm(x).unsqueeze(1).detach()
        x = self.netcnn(x)
        x = self.encoder(x)
        x = x.view((x.size()[0], -1))
        x = self.fc(x)
        return x
