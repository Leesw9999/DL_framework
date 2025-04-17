import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - x ** 2
    return y

def gx2(x):
    return 12 * x ** 2 - 2

x = Variable(np.array(2.0)) # 초깃값
iters = 10 # 반복 횟수

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)
