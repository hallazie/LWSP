import numpy as np

a = np.array([1,2,3,4,3,2,1])
b = 255*(a-np.amin(a))/(np.amax(a)-np.amin(a))
print b