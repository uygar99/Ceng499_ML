import numpy as np
a = np.array(((1,2,3,4),(4,5,6,7),(6,5,6,7)))
b = np.array(((1,1,1,1),(0,0,0,0),(1,1,1,1)))
print(np.stack((a, b), axis=0))
print(np.log(a))
print(np.log2(a))