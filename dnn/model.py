import torch
import torch.nn as nn

class MLP(nn.Module):
	def __init__(self, input_dim):
		super(MLP, self).__init__()
		self.layer1 = nn.Linear(input_dim, 100)
		self.layer2 = nn.Linear(100, 100)
		self.layer3 = nn.Linear(100, 100)
		self.layer4 = nn.Linaer(100, 1)

	def forward(self, x):
		z = self.layer1(x)
		z = self.layer2(z)
		z = self.layer3(z)
		out = self.layer4(z)

		return out
