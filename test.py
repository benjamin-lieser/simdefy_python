import numpy as np
import simdefy
import time


x = np.random.random((100,100))
x = np.array(x, dtype=np.double)



y = simdefy.log1exp(x)
y2 = np.log(1+np.exp(x))

print(np.isclose(y,y2).all())
