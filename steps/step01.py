import numpy as np

class Variable:
    def __init__(self, data): # __init__ : 초기화 함수, data : 인스턴스 변수
        self.data = data

data = np.array(1.0)
x = Variable(data) # x : Variable의 인스턴스 변수
print(x.data)
# 1.0

x.data = np.array(2.0)
print(x.data)
# 2.0
