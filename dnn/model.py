import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 100)
        self.relu1 = nn.PReLU()
        self.layer2 = nn.Linear(100, 100)
        self.ss = nn.Softsign()
        self.drop = nn.Dropout()
        self.layer3 = nn.Linear(100, 100)
        self.relu2 = nn.PReLU()
        self.layer4 = nn.Linear(100, 1)

    def forward(self, x):
        z = self.layer1(x)
        z = self.relu1(z)
        z = self.layer2(z)
        z = self.ss(z)
        z = self.drop(z)
        z = self.layer3(z)
        z = self.relu2(z)
        out = self.layer4(z)

        return out
