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
def branch_dist(test):
	if isinstance(test.ops[0], ast.Eq):
		return 0, ast.Call(func=ast.Name(id='abs'),
							args=[ast.BinOp(left=test.left, op=ast.Sub(), right=test.comparators[0])],
							keywords=[],
							starags=None,
							kwargs=None)
	
	elif isinstance(test.ops[0], ast.NotEq):
		return 1, ast.UnaryOp(op=ast.USub(),
								operand=ast.Call(func=ast.Name(id='abs'),
													args=[ast.BinOp(left=test.left, op=ast.Sub(),
															right=test.comparators[0])],
													keywords=[],
													starags=None,
													kwargs=None))
	
	elif isinstance(test.ops[0], ast.Lt):
		return 1, ast.BinOp(left=test.left, op=ast.Sub(), right=test.comparators[0])

	elif isinstance(test.ops[0], ast.LtE):
		return 0, ast.BinOp(left=test.left, op=ast.Sub(), right=test.comparators[0])
	
	elif isinstance(test.ops[0], ast.Gt):
		return 1, ast.BinOp(left=test.comparators[0], op=ast.Sub(), right=test.left)

	elif isinstance(test.ops[0], ast.GtE):
		return 0, ast.BinOp(left=test.comparators[0], op=ast.Sub(), right=test.left)


class branch:
	br_list = [None]
	
	# parent: index of parent, op_type:
	def __init__(self, parent, op_type, lineno, reach):
		self.ind = len(branch.br_list)
		self.lineno = lineno

		# Ind of parent(if on true branch positive, elsewise negative)
		self.parent = parent
		self.op_type = op_type
		
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
					op_type, node = branch_dist(line.test)
					new_branch = branch(parent, op_type, line.lineno, reach)

					# Assign branch distance to temporary variable
					body.insert(ind, ast.Assign(targets=[ast.Name(id=args.temp_name)],
												value=node))

					# Print branch_id, op_type, branch distance in order
					body.insert(ind + 1, ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id=args.file_name),
																				attr='write'),
																				args=[ast.Call(func=ast.Attribute(value=ast.Str(s='{} {} {}\n'),
																												attr='format'),
																							args=[ast.Num(n=new_branch.ind), ast.Num(n=op_type), ast.Name(id=args.temp_name)],
																							keywords=[],
																							starargs=None,
																							kwargs=None)],
																				keywords=[],
																				starargs=None,
																				kwargs=None)))

					line.test.left = ast.Name(id=args.temp_name)
					line.test.comparators = [ast.Num(n=0)]

					if op_type == 0:
						line.test.ops = [ast.LtE()]
					else:
						line.test.ops = [ast.Lt()]
					
					if isinstance(line, ast.While):
						line.body.append(body[ind])
						line.body.append(body[ind + 1])

					find_if(line.body, new_branch.ind, args, reach)
					find_if(line.orelse, -new_branch.ind, args, reach)
					
					ind += 2

				else:
					find_if(line, parent, args, reach)

				ind += 1
