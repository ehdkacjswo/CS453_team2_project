import ast
import astor

from function import make_node_function_def

'''
python srcgen.py > make_node_source_file.py
Reference : https://github.com/radomirbosak/random-ast.git
'''

def main():
    fun_def = make_node_function_def(max_depth=4, num_body_statements=4)
    expr = ast.Module(body=[fun_def])
    src = astor.code_gen.to_source(expr)
    print(src)

if __name__ == '__main__':
    main()
