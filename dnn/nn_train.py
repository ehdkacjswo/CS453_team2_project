import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def guided_mutation(inputs, args):
	inputs_var = torch.Tensor(inputs).to(args.device)
	inputs_var.requires_grad_(True)
	fitness = args.model(inputs_var)
	grad = torch.autograd.grad(fitness, inputs_var)[0]
	
	mutated_input = inputs_var.detach() + args.step_size * torch.sign(grad.detach())
	return mutated_input[:, :args.input_dim].round()

def train_one_iter(inputs, fitness, args):
	inputs_var = torch.Tensor(inputs).to(args.device)
	target_var = torch.Tensor(fitness).to(args.device)
	pred = args.model(inputs_var)

	loss = nn.MSELoss()(pred, target_var.unsqueeze(1))

	args.optimizer.zero_grad()
	loss.backward()
	args.optimizer.step()

	'''print("[{}/{}]: loss={}".format(iter_num, args.niter, loss.item()))'''

	return loss.item()
