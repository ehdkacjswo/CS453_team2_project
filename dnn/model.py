import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 100)
        self.lr1 = nn.LeakyReLU()
        #self.bn1 = nn.BatchNorm1d(num_features=100)
        self.layer2 = nn.Linear(100, 100)
        self.lr2 = nn.PReLU()
        #self.bn2 = nn.BatchNorm1d(num_features=100)
        self.layer3 = nn.Linear(100, 100)
        self.lr3 = nn.PReLU()
        #self.bn3 = nn.BatchNorm1d(num_features=100)
        self.layer4 = nn.Linear(100, 1)
        self.lr4 = nn.PReLU()

    def forward(self, x):
        z = self.layer1(x)
        z = nn.PReLU()(z)
        z = self.layer2(z)
        z = nn.Softsign()(z)
        z = self.layer3(z)
        z = nn.PReLU()(z)
        out = self.layer4(z)

        return out
