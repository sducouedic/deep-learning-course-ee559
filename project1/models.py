import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from modelclass import Model


class Baseline(Model):
    def __init__(self, f_gen_data, nb_epochs=25, mini_batch_size=100, learning_rate=1e-3):
        super().__init__(f_gen_data, "Baseline", nb_epochs, mini_batch_size, learning_rate)

        self.fc1 = nn.Linear(392, 2)

    def forward(self, x):
        nb_sample = x.size()[0]

        x = self.fc1(x.view(nb_sample, -1))
        x = F.relu(x)
        return x

