#
from random import seed
from random import random
from matplotlib import pyplot
import numpy as np

seed(1)
random_walk = list()
random_walk.append(0)

for i in range (1,1000):
	next_step = -1 if random() < 0.5 else 1
	value = random_walk[i-1] + next_step
	random_walk.append(value)

pyplot.plot(random_walk)
pyplot.show()
np.savetxt('rw_seed_1.txt', random_walk, delimiter = ',')
