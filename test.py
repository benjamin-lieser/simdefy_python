import numpy as np
import simdefy
import time
import scipy


x = np.random.random((100,100))
x = np.array(x, dtype=np.float32)
x = (x + 3) * 10


simdefy.init()
y = simdefy.log_gamma(x)
y_scipy = scipy.special.loggamma(x)

print("Maximal abluste difference to scipy is")
print(np.max(np.abs(y - y_scipy)))

y_avx = simdefy.log_gamma_avx2(x)
print("Maximal abluste difference of avx2 to scipy is")
print(np.max(np.abs(y_avx - y_scipy)))

