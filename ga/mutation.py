import random as rand
import math

# Mutate given input
def doam_mut(test, special, pm, alpha, beta):
	same = 0
	rt = []
	pm = 1.0 / len(test)

	for ind in range(len(test)):
		prob = rand.random()
		if prob <= pm / 4:
			test[ind] = rand.choice(special)

		elif prob <= pm:
			test[ind] += (math.floor(rand.gammavariate(alpha, beta)) + 1) * rand.choice([-1, 1])
			#test[ind] += rand.choice([-1, 1])
		else:
			same += 1
	
	# If none is changed, muteate again
	if same == len(test):
		return doam_mut(test, special, pm, alpha, beta)
	
	return test
