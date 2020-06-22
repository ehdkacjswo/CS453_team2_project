import ast
import astor

from evaluation.gen_src.function import make_node_function_def

def generate():
    fun_def = make_node_function_def(max_depth=3, num_body_statements=4)
    expr = ast.Module(body=[fun_def])
    src = astor.code_gen.to_source(expr)
    return src