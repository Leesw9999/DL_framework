import numpy as np

class Variable:
    def __init__(self, data): # __init__ : 초기화 함수, data : 인스턴스 변수
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data # 데이터를 꺼냄
        y = self.forward(x) # 실제 계산
        output = Variable(y) # Variable 형태로 되돌림
        return output

    def forward(self, x): # Function 클래스의 forward 메서드는 예외를 발생시킴
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b) # (e^x^2)^2

print(y.data)
