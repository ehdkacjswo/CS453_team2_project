def ff(a, b, c):
	if a == b and b == c:
		return 1
	elif a == b or c == a or b == c:
		return 2
	else:
		return 3
