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
    
def numerical_diff(f, x, eps=1e-4): # eps(h) : 앱실론
    x0 = Variable(x.data - eps) # x - h
    x1 = Variable(x.data + eps) # x + h
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

f = Square() # x^2
x = Variable(np.array(2.0))
dy = numerical_diff(f, x) # 2x -> 2*2.0

print(dy)

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x))) # (e^x^2)^2

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)
