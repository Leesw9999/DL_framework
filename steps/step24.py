if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

def beale(x, y):
   
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2


x = Variable(np.array(3.0))
y = Variable(np.array(0.5))

z = beale(x, y)
z.backward() 

print("Beale function gradient at (3, 0.5):")
print("x.grad =", x.grad)
print("y.grad =", y.grad)
