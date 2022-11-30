import numpy as np
import simdefy
import time
import scipy


x = np.random.random((1000,100000))
x = np.array(x, dtype=np.float32) + 10

simdefy.init()

start = time.time()
for i in range(0,1000):
	y = simdefy.log_gamma(x[i])
simdefy_time = time.time() - start

start = time.time()
for i in range(0,1000):
	y2 = scipy.special.loggamma(x[i])
numpy_time = time.time() - start

print(f'simdefy : {simdefy_time}')
print(f'numpy : {numpy_time}')

print(f'simdefy is {numpy_time / simdefy_time} times faster')
