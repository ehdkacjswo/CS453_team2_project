import ast, astor

print(astor.dump_tree(astor.code_to_ast.parse_file('test.py')))
