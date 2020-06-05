import random as rand
import math

# Mutate given input
def doam_mut(test, special, pm, sigma):
	same = 0

	for ind in range(len(test)):
		prob = rand.random()
		if prob <= pm / 4:
			test[ind] = rand.choice(special)

		elif prob <= pm:
			test[ind] += rand.gauss(0, sigma)

		else:
			same += 1
	
	# If none is changed, muteate again
	if same == len(test):
		return doam_mut(test, special, pm, sigma)
	
	return test
