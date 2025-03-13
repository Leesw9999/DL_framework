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

x = Variable(np.array(10))
f = Square()
y = f(x)

print(type(y)) # type() 함수는 객체의 클래스를 알려줌
print(y.data)
