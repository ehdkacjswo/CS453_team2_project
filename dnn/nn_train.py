import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def guided_mutation(inputs, args):
    args.model.eval()

    inputs_var = torch.Tensor(inputs).to(args.device)
    inputs_var.requires_grad_(True)
    fitness = args.model(inputs_var)
    gradient = torch.autograd.grad(fitness, inputs_var)[0]

    org_input = inputs_var.detach()
    grad = gradient.detach()

    mutated_input = org_input + grad * args.step_size
    return mutated_input.round()


def train(inputs, fitness, loss_range, args):
    args.model.train()
    
    inputs_var = torch.Tensor(inputs).to(args.device)
    inputs_var.requires_grad_(False)

    target_var = torch.Tensor(fitness).to(args.device)
    target_var.requires_grad_(False)

    
    for epoch in range(args.niter + 1):
        args.opt.zero_grad()
        pred = args.model(inputs_var)
        loss = nn.L1Loss()(pred, target_var)

        if loss.item() < loss_range or epoch == args.niter:
            return epoch, loss.item()

        loss.backward()

        #torch.nn.utils.clip_grad_norm_(args.model.parameters(), 10)
        args.opt.step()

def forward(inputs, leaf_ind, args):
    args.model.eval()
    inputs_var = torch.Tensor(inputs).to(args.device)
    pred = args.model(inputs_var)

    return pred.item()
