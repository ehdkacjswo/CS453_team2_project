import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def guided_mutation(args, model, inputs):
	inputs_var = torch.Tensor(inputs).to(device)
	fitness = model(inputs_var)
	grad = torch.autograd.grad(fitness, inputs_var)[0]
	mutated_input = inputs_var.detach() + args.step_size * torch.sign(grad.detach())

	return mutated_input

def train_one_iter(args, model, optimizer, inputs, fitness, iter_num, device):
	inputs_var = torch.Tensor(inputs).to(device)
	target_var = torch.Tensor(fitness).to(device)
	pred = model(inputs_var)

	loss = nn.MSELoss()(pred, fitness)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	print("[{}/{}]: loss={}".format(iter_num, args.niter, loss.item()))

	return loss.item()