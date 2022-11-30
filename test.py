import numpy as np
import simdefy
import time
import scipy


x = np.random.random((5))
x = np.array(x, dtype=np.float32)
x = x + 10


simdefy.init()
y = simdefy.log_gamma(x)
print(y)

print(scipy.special.loggamma(x))

#print(np.isclose(y,y2).all())
