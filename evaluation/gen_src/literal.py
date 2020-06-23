import ast
import random

def make_node_num(max_depth=None):
    n = random.randrange(-10000, 10000)
    return ast.Num(n)

def make_node_literal(vctx, max_depth=None):
    choices = [
        make_node_num,
    ]

    if max_depth >= 1:
        choices += [
        ]

    return random.choice(choices)(max_depth=max_depth)