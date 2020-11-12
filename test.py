import sys
import numpy as np

import FocusAnalysis

sys.path.append('../')
from Utils import utils

a = np.array([  [1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16] ])

print(a)
print(a[1][3])
print("a.size: ", a.size)
b = a.flatten()
print(b)
print("b.size: ", b.size)
c = b[b != 8]
print(c)
print("c.size: ", c.size)


#a[1,:] = -1
b = np.append(b,100)
print(b)

print(np.percentile(b,95))

#utils.overSampleBlockSumGrid(a, target_height, target_width)

e = np.array([30,33,43,53,56,67,68,72])
print(np.percentile(e,25))

print(a)
a[a < 10] = 0
print(a)
