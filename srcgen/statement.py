import ast
import random

from expression import make_node_expression
from variable import make_node_lhs_variable

def make_node_assign(vctx, max_depth=None):
    value = make_node_expression(vctx, max_depth=max_depth)
    targets = [make_node_lhs_variable(vctx, max_depth=max_depth)]
    return ast.Assign(targets, value)

def make_node_loop_end(vctx, max_depth=None):
    choices = [
        make_node_return,
        make_node_break,
        make_node_continue,
    ]
    return random.choice(choices)(vctx, max_depth=max_depth)

def make_node_pass(max_depth=None):
    return ast.Pass()

def make_node_break(vctx, max_depth=None):
    return ast.Break()

def make_node_continue(vctx, max_depth=None):
    return ast.Continue()

def make_node_return(vctx, max_depth=None):
    return ast.Return(make_node_expression(vctx, max_depth=max_depth - 1))