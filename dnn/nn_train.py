import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def guided_mutation(inputs, leaf_ind, args):
    args.dnn[leaf_ind][0].eval()

    inputs_var = torch.Tensor(inputs).to(args.device)
    inputs_var.requires_grad_(True)
    fitness = args.dnn[leaf_ind][0](inputs_var)
    gradient = torch.autograd.grad(fitness, inputs_var)[0]

    org_input = inputs_var.detach()
    grad = gradient.detach()

    mutated_input = org_input - grad * 1e6
    return mutated_input[:, :args.input_dim].round()


def train(inputs, fitness, leaf_ind, args):
    args.dnn[leaf_ind][0].train()

    inputs_var = torch.Tensor(inputs).to(args.device)
    target_var = torch.Tensor(fitness).to(args.device)

    #scheduler = optim.lr_scheduler.StepLR(args.dnn[leaf_ind][1], step_size = 25, gamma = 0.95)
    #scheduler = optim.lr_scheduler.CyclicLR(args.dnn[leaf_ind][1], base_lr = 1e-4, max_lr = 0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(args.dnn[leaf_ind][1], factor = 0.95, patience = 25, mode='min')

    for epoch in range(args.niter + 1):
        pred = args.dnn[leaf_ind][0](inputs_var)
        #loss = nn.MSELoss()(pred, target_var.unsqueeze(1))
        loss = nn.L1Loss()(pred, target_var)
        #print(loss)

        if loss.item() < 0.2 or epoch == args.niter:
            return epoch, loss.item()

        loss.backward()
        args.dnn[leaf_ind][1].step()
        args.dnn[leaf_ind][1].zero_grad()

        scheduler.step(loss)


def forward(inputs, leaf_ind, args):
    args.dnn[leaf_ind][0].eval()
    inputs_var = torch.Tensor(inputs).to(args.device)
    pred = args.dnn[leaf_ind][0](inputs_var)

    return pred.item()
