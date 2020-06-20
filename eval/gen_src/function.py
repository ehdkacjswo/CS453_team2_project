import ast
import random

from variable import generate_variable_name
from expression import make_node_expression
from block import make_node_block

def make_node_function_def(max_depth=None, num_body_statements=None):
    vctx = set()
    name = 'fun_' + generate_variable_name()

    args = make_node_arguments(vctx, name, max_depth=1)
    body = make_node_block(vctx, max_depth=max_depth, num_statements=num_body_statements)

    decorator_list = []
    returns = None

    return ast.FunctionDef(name, args, body, decorator_list, returns)


def make_node_arguments(vctx, fun_name, max_depth=None):
    num_args = random.randrange(1, 4)
    names = set(generate_variable_name() for _ in range(num_args))
    args = [ast.arg(name, None) for name in names]
    vctx = vctx.union(names)
    
    defaults = []
    kwarg = None
    vararg = None
    kwonlyargs = None
    kw_defaults = None

    return ast.arguments(args=args, vararg=vararg, kwonlyargs=kwonlyargs, kwarg=kwarg, defaults=defaults, kw_defaults=kw_defaults)

