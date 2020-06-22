import ast
import random
import copy

from gen_src.expression import make_node_expression, make_node_test_expression
from gen_src.statement import make_node_assign, make_node_loop_end, make_node_pass

MAX_NUM_STATEMENTS = 4

def make_node_block(vctx, max_depth=None, block_pass=True, num_statements=None):
    if num_statements is None and block_pass and random.choice([True] + [False] * 10):
        return [make_node_pass(max_depth=max_depth)]
    else:
        if num_statements is None:
            num_statements = random.randrange(1, MAX_NUM_STATEMENTS)
        choices = [
            make_node_assign,
        ]
        if max_depth >= 1:
            choices += [
                make_node_if,
            ]

        vctx_copy = copy.copy(vctx)
        node_funs = random.choices(choices, k=num_statements)

        ret = []

        for _ in range(num_statements - 1):
            if len(vctx) > 0:
                fun = random.choice(choices)
            else:
                fun = make_node_assign
            ret.append(fun(vctx, max_depth=max_depth - 1))

        if len(vctx) > 0 and max_depth >= 1:
            fun = make_node_if
        else:
            fun = make_node_assign
        ret.append(fun(vctx, max_depth=max_depth - 1))

        vctx = vctx_copy
        return ret

def make_node_loop_block(vctx, max_depth=None):
    if random.choice([True] + [False] * 10):
        return [make_node_pass(max_depth=max_depth)]
    else:
        block = make_node_block(vctx, max_depth=max_depth, block_pass=False)
        if random.choice([True] + [False] * 4):
            block.append(make_node_loop_end(vctx, max_depth=max_depth))
        return block


def make_node_if(vctx, max_depth=None):
    test = make_node_test_expression(vctx, max_depth=min(max_depth, 1))
    body = make_node_block(copy.copy(vctx), max_depth=max_depth)
    orelse = random.choice([make_node_block(copy.copy(vctx), max_depth=max_depth), []])
    return ast.If(test, body, orelse)

def make_node_while(vctx, max_depth=None):
    test = make_node_test_expression(vctx, max_depth=min(max_depth, 1))
    body = make_node_loop_block(vctx, max_depth=max_depth)
    orelse = random.choice([make_node_block(vctx, max_depth=max_depth), []])
    return ast.While(test, body, orelse)