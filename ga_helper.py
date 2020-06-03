import random as rand
import math

# Check whether new test is in the given test_list
def in_test(test_list, new_test):
	for test in test_list:
		if test == new_test:
			return True
	
	return False

# If non of given test is identical to new test, add it
def add_test(test_list, new_test):
	if in_test(test_list, new_test):
		return test_list
	
	return test_list + [new_test]

# Mutate given input
def mutate(test, special, pm, alpha, beta):
	same = 0

	for ind in range(len(test)):
		prob = rand.random()
		if prob <= pm / 4:
			test[ind] = rand.choice(special)

		elif prob <= pm:
			add = int(math.floor(rand.gammavariate(alpha, beta))) + 1
			sign = rand.choice([-1, 1])

			test[ind] += sign * add

		else:
			same += 1
	
	# If none is changed, muteate again
	if same == len(test):
		return mutate(test, special, pm, alpha, beta)
	
	return test
