import ast
import astor
import random

from variable import make_node_rhs_variable
from literal import make_node_literal

binary_ops = [
    ast.Add,
    ast.Sub,
    ast.Mult,
    # ast.Div,
    # ast.FloorDiv,
    # ast.Mod,
]
bool_ops = [ast.Or, ast.And]
comparisons = [
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
]


def make_node_expression(vctx, max_depth=None, numeric=False):
    choices = [
        make_node_literal,
    ]
    if len(vctx) > 0:
        choices += [
            make_node_rhs_variable,
        ]
    if max_depth >= 1:
        choices += [
            make_node_binary_op,
            make_node_binary_op,
        ]
        if not numeric:
            choices += [
                make_node_bool_op,
                make_node_comparison,
            ]

    return random.choice(choices)(vctx, max_depth=max_depth)

def make_node_test_expression(vctx, max_depth=None):
    choices = [
        make_node_bool_op,
        make_node_comparison,
        make_node_comparison,
    ]
    ret = random.choice(choices)(vctx, max_depth=max_depth)
    return ret

def make_node_binary_op(vctx, max_depth=None):
    left = make_node_expression(vctx, max_depth=max_depth - 1, numeric=True)
    right = make_node_expression(vctx, max_depth=max_depth - 1, numeric=True)
    op = random.choice(binary_ops)
    return ast.BinOp(left, op(), right)


def make_node_bool_op(vctx, max_depth=None):
    op = random.choice(bool_ops)
    length = max(2, random.randrange(0, 4))

    values = [make_node_expression(vctx, max_depth=max_depth - 1) for _ in range(length)]
    ret = ast.BoolOp(op(), values)
    return ret


def make_node_comparison(vctx, max_depth=None):
    length = 2

    ops = [op() for op in random.choices(comparisons, k=length - 1)]
    left, *comparators = [make_node_expression(vctx, max_depth=max_depth - 1) for _ in range(length)]
    ret = ast.Compare(left, ops, comparators)
    return ret
