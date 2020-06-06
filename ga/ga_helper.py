import random as rand
import math

# Check whether new test is in the given test_list
def in_test(test_list, new_test):
	'''for test in test_list:
		if test == new_test:
			return True'''

	for test in test_list:
		same = True

		for i in range(len(test)):
			if abs(test[i] - new_test[i]) > 1e-5:
				same = False
				break

		if same:
			return True
	
	return False

# If non of given test is identical to new test, add it
def add_test(test_list, new_test):
	if in_test(test_list, new_test):
		return test_list
	
	return test_list + [new_test]
