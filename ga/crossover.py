import random as rand
from ga.mutation import doam_mut

def doam_cross(output, leaf_ind, special, pm, alpha, beta):
	pair = []
	
	# Binary tournament selection
	for k in range(2):
		p1 = rand.choice(output)
		p2 = rand.choice(output)

		if p1[1][leaf_ind] < p2[1][leaf_ind]:
			pair.append(p1)
		elif p1[1][leaf_ind] > p2[1][leaf_ind]:
			pair.append(p2)
		else:
			pair.append(rand.choice([p1, p2]))

	if pair[0][0] == pair[1][0]:
		return doam_cross(output, leaf_ind, special, pm, alpha, beta)

	score1 = pair[0][1][leaf_ind]
	score2 = pair[1][1][leaf_ind]

	children = []

	# Single point crossover
	if len(pair[0][0]) > 1:
		cross_point = rand.randint(1, len(pair[0][0]) - 1)

		children.append(pair[0][0][:cross_point] + pair[1][0][cross_point:])
		children.append(pair[1][0][:cross_point] + pair[0][0][cross_point:])

	# If single point crossover is unvailable, mutate
	else:
		children.append(doam_mut(pair[0][0], special, 1.0, alpha, beta))
		children.append(doam_mut(pair[1][0], special, 1.0, alpha, beta))
			
	# Secant method
	if score1 != score2:
		children.append([(pair[0][0][k] * (score2 + 1) - pair[1][0][k] * (score1 + 1)) / (score2 - score1) for k in range(len(pair[0][0]))])
	
	for child in children:
		if rand.random() < pm or child == pair[0][0] or child == pair[1][0]:
			child = doam_mut(child, special, pm, alpha, beta)

	return children

