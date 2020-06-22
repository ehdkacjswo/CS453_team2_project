import random as rand
import math

# Mutate given input


def doam_mut(test, special, pm, alpha, beta):
    same = 0
    pm = 1.0 / len(test)

    for ind in range(len(test)):
        prob = rand.random()
        org = test[ind]
        
        if prob <= pm / 4:
            test[ind] = rand.choice(special)
            
        elif prob <= pm / 2:
            test[ind] = rand.randint(-10000, 10000)

        elif prob <= pm:
            test[ind] += int(math.floor(rand.gammavariate(alpha, beta)) + 1) * rand.choice([-1, 1])

        if org == test[ind]:
            same += 1

    # If none is changed, muteate again
    if same == len(test):
        return doam_mut(test, special, pm, alpha, beta)

    return test
