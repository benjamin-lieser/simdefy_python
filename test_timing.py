import numpy as np
import simdefy
import time


x = np.random.random((1000,10000))
x = np.array(x, dtype=np.double)


start = time.time()
for i in range(0,1000):
	y = simdefy.log1exp(x[i])
simdefy_time = time.time() - start

start = time.time()
for i in range(0,1000):
	y2 = np.log(1+np.exp(x[i]))
numpy_time = time.time() - start

print(f'simdefy : {simdefy_time}')
print(f'numpy : {numpy_time}')

print(f'simdefy is {numpy_time / simdefy_time} times faster')
