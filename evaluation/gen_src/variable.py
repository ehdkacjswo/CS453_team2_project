import ast
import random

def make_node_lhs_variable(vctx, max_depth=None):
    ctx = ast.Store()
    if len(vctx) == 0 or random.choice([True, False]):
        for _ in range(5):
            name = generate_variable_name()
            if name not in vctx:
                vctx.add(name)
                return ast.Name(name, ctx)
    return make_node_rhs_variable(vctx, max_depth=None)

def make_node_rhs_variable(vctx, max_depth=None):
    assert(len(vctx) > 0)
    ctx = ast.Load()
    name = random.choice(tuple(vctx))
    return ast.Name(name, ctx)

def generate_variable_name():
    return random.choice(list('abcdefxyz'))