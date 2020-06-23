import ast
import astor

# Find ast.Num node
def find_num(node):
	if isinstance(node, ast.Num):
		return [node.n]
	
	rt = []

	try:
		for field in node._fields:
			rt.extend(find_num(getattr(node, field)))
	except AttributeError:
		if isinstance(node, list):
			for child in node:
				rt.extend(find_num(child))
	
	return rt

# Maximum length of names
def name_len(node):
	if isinstance(node, str):
		return len(node)
    
	rt = 0
    
	try:
		for field in node._fields:
			rt = max(rt, name_len(getattr(node, field)))
	except AttributeError:
		if isinstance(node, list):
			for child in node:
				rt = max(rt, name_len(child))
	
	return rt

# Get branch distance for given if statement
# 0(Eq, LtE, GtE), 1(NotEq, Lt, Gt)
def branch_dist_comp(test, args):
	if isinstance(test.ops[0], ast.Eq):
		op_type = 0
		br_dist = ast.Call(func=ast.Name(id='abs'),
							args=[ast.BinOp(left=test.left, op=ast.Sub(), right=test.comparators[0])],
							keywords=[],
							starags=None,
							kwargs=None)
	
	elif isinstance(test.ops[0], ast.NotEq):
		op_type = 1
		br_dist = ast.UnaryOp(op=ast.USub(),
								operand=ast.Call(func=ast.Name(id='abs'),
													args=[ast.BinOp(left=test.left, op=ast.Sub(),
															right=test.comparators[0])],
													keywords=[],
													starags=None,
													kwargs=None))
	
	elif isinstance(test.ops[0], ast.Lt):
		op_type = 1
		br_dist = ast.BinOp(left=test.left, op=ast.Sub(), right=test.comparators[0])

	elif isinstance(test.ops[0], ast.LtE):
		op_type = 0
		br_dist = ast.BinOp(left=test.left, op=ast.Sub(), right=test.comparators[0])
	
	elif isinstance(test.ops[0], ast.Gt):
		op_type = 1
		br_dist = ast.BinOp(left=test.comparators[0], op=ast.Sub(), right=test.left)

	elif isinstance(test.ops[0], ast.GtE):
		op_type = 0
		br_dist = ast.BinOp(left=test.comparators[0], op=ast.Sub(), right=test.left)
	
	return ast.Call(func=ast.Lambda(args=ast.arguments(args=[ast.arg(arg=args.lambda_arg, annotation=None)],
														vararg=None,
														kwonlyargs=[],
														kw_defaults=[],
														kwarg=None,
														defaults=[]),
									 body=ast.IfExp(test=ast.Compare(left=ast.Name(id=args.lambda_arg),
									 									ops=[ast.LtE() if op_type == 0 else ast.Lt()],
																		comparators=[ast.Num(n=0)]),
													body=ast.Name(id=args.lambda_arg),
													orelse=ast.BinOp(left=ast.Name(id=args.lambda_arg),
																		op=ast.Add(),
																		right=ast.Num(args.k)))),
					args=[br_dist],
					keywords=[])

# Branch distance for boolean operator (and, or)
def branch_dist_boolop(op, values, args):
	if len(values) == 1:
		return branch_dist(values[0], args)

	else:
		return ast.Call(func=ast.Lambda(args=ast.arguments(args=[ast.arg(arg=args.lambda_arg, annotation=None)],
															vararg=None,
															kwonlyargs=[],
															kw_defaults=[],
															kwarg=None,
															defaults=[]),
										body=ast.IfExp(test=ast.Compare(left=ast.Name(id=args.lambda_arg),
																		ops=[ast.Gt() if isinstance(op,ast.And) else ast.LtE()],
																		comparators=[ast.Num(n=0)]),
														body=ast.Name(id=args.lambda_arg),
														orelse=branch_dist_boolop(op, values[1:], args))),
						args=[branch_dist(values[0], args)],
						keywords=[])

# Branch distance for not operator
def branch_dist_not(test, args):
	return ast.BinOp(left=ast.Num(n=args.k), op=ast.Sub(), right=branch_dist(test.operand, args))

def branch_dist(test, args):
	if isinstance(test, ast.Compare):
		return branch_dist_comp(test, args)
	
	elif isinstance(test, ast.BoolOp):
		return branch_dist_boolop(test.op, test.values, args)
	
	elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
		return branch_dist_not(test, args)
	
	else:
		return test

class branch:
	br_list = [None]
	
	# parent: index of parent, op_type:
	def __init__(self, parent, lineno, reach):
		self.ind = len(branch.br_list)
		self.lineno = lineno

		# Ind of parent(if on true branch positive, elsewise negative)
		self.parent = parent
		
		# Whether it has child on true, false branch
		self.true = False
		self.false = False

		# Whether it's rechable systatically
		self.reach = reach
		
		# Add itself to parent's branch if reachable
		if reach:
			if parent > 0:
				branch.br_list[parent].true = True
			elif parent < 0:
				branch.br_list[-parent].false = True

		branch.br_list.append(self)


# Find branch of code from function body ast
def find_if(body, parent, args, reach):
	try:
		for field in body._fields:
			find_if(getattr(body, field), parent, args, reach)

	except AttributeError:
		if isinstance(body, list):
			ind = 0

			while ind in range(len(body)):
				line = body[ind]

				if isinstance(line, ast.Return):
					reach = False

				elif isinstance(line, ast.If) or isinstance(line, ast.While):
					node = branch_dist(line.test, args)
					new_branch = branch(parent, line.lineno, reach)

					# Assign branch distance to temporary variable
					body.insert(ind, ast.Assign(targets=[ast.Name(id=args.temp_name)],
												value=node))

					# Print branch_id, op_type, branch distance in order
					body.insert(ind + 1, ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=args.file_name),
																				attr='write'),
																				args=[ast.Call(func=ast.Attribute(value=ast.Str(s='{} {}\n'),
																												attr='format'),
																							args=[ast.Num(n=new_branch.ind), ast.Name(id=args.temp_name)],
																							keywords=[],
																							starargs=None,
																							kwargs=None)],
																				keywords=[],
																				starargs=None,
																				kwargs=None)))

					line.test = ast.Compare(left=ast.Name(id=args.temp_name), ops=[ast.LtE()], comparators=[ast.Num(n=0)])
					
					if isinstance(line, ast.While):
						line.body.append(body[ind])
						line.body.append(body[ind + 1])

					find_if(line.body, new_branch.ind, args, reach)
					find_if(line.orelse, -new_branch.ind, args, reach)
					
					ind += 2

				else:
					find_if(line, parent, args, reach)

				ind += 1
