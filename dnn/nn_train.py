import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def guided_mutation(inputs, leaf_ind, args):
	inputs_var = torch.Tensor(inputs).to(args.device)
	inputs_var.requires_grad_(True)
	fitness = args.dnn[leaf_ind][0](inputs_var)
	gradient = torch.autograd.grad(fitness, inputs_var)[0]

	org_input = inputs_var.detach()
	grad = gradient.detach()

	if torch.norm(grad) == 0:
		return org_input[:, :args.input_dim].round()
		
	mutated_input = org_input - grad / torch.norm(grad) * 0.1 * torch.norm(org_input)
	return mutated_input[:, :args.input_dim].round()

def train_one_iter(inputs, fitness, leaf_ind, args):
	inputs_var = torch.Tensor(inputs).to(args.device)
	target_var = torch.Tensor(fitness).to(args.device)
		
	pred = args.dnn[leaf_ind][0](inputs_var)
	#loss = nn.MSELoss()(pred, target_var.unsqueeze(1))
	loss = nn.MSELoss()(pred, target_var)
	
	args.dnn[leaf_ind][1].zero_grad()
	loss.backward()
	args.dnn[leaf_ind][1].step()

	'''print("[{}/{}]: loss={}".format(iter_num, args.niter, loss.item()))'''
	return loss.item()

def forward(inputs, leaf_ind, args):
	inputs_var = torch.Tensor(inputs).to(args.device)
	pred = args.dnn[leaf_ind][0](inputs_var)

	return pred.item()
