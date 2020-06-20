import math

def f(a, b, c):
	d = int(math.floor(math.sqrt(a)))
	e = a % b
	f = pow(a, b, c)

	if d < a:
		if e == d:
			if f == e:
				return 1
		elif e > d:
			if f < e:
				return 2
		else:
			if f > e:
				return 3
