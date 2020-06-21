import random as rand
import math
import torch
import copy
from ga.mutation import doam_mut
from dnn.nn_train import guided_mutation


def doam_cross(output, leaf_ind, special, args):
    pair = []

    # Binary tournament selection
    for i in range(2):
        p1 = rand.choice(output)
        p2 = rand.choice(output)

        if p1[1][leaf_ind] < p2[1][leaf_ind]:
            pair.append(p1)
        elif p1[1][leaf_ind] > p2[1][leaf_ind]:
            pair.append(p2)
        else:
            pair.append(rand.choice([p1, p2]))

    if pair[0][0] == pair[1][0]:
        return doam_cross(output, leaf_ind, special, args)

    score1 = pair[0][1][leaf_ind]
    score2 = pair[1][1][leaf_ind]

    children = []

    # Single point crossover
    if rand.random() < 0.9 and len(pair[0][0]) > 1:
        cross_point = rand.randint(1, len(pair[0][0]) - 1)

        children.append(pair[0][0][:cross_point] + pair[1][0][cross_point:])
        children.append(pair[1][0][:cross_point] + pair[0][0][cross_point:])

    # If single point crossover is unvailable, mutate
    else:
        children.append(doam_mut(copy.deepcopy(
            pair[0][0]), special, 1.0, args.alpha, args.beta))
        children.append(doam_mut(copy.deepcopy(
            pair[1][0]), special, 1.0, args.alpha, args.beta))

    # Dnn help
    if args.use_dnn:
        for i in range(2):
            children.append(guided_mutation(
                [pair[i][0]], args).tolist()[0])
            children[-1] = [int(inp) for inp in children[-1]]

    for ind in range(len(children)):
        child = children[ind]

        if rand.random() < args.pm or child == pair[0][0] or child == pair[1][0]:
            children[ind] = doam_mut(
                child, special, args.pm, args.alpha, args.beta)

    return children
