import numpy as np

class Variable:
    def __init__(self, data): # __init__ : 초기화 함수, data : 인스턴스 변수
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func):
        self.creator = func

    def backward(self): # 반복
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 함수를 가져옴
            x, y = f.input, f.output # 함수의 입력과 출력을 가져옴
            x.grad = f.backward(y.grad) # 함수의 backward 메서드 호출

            if x.creator is not None:
                funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가

class Function:
    def __call__(self, input):
        x = input.data # 데이터를 꺼냄
        y = self.forward(x) # 실제 계산
        output = Variable(y) # Variable 형태로 되돌림
        output.set_creator(self) # 출력 변수에 창조자를 설정
        self.input = input # 입력 변수를 기억(보관)
        self.output = output # 출력 저장
        return output

    def forward(self, x): # Function 클래스의 forward 메서드는 예외를 발생시킴
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy # x제곱 미분 = 2x
        return gx
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# 순전파
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 역전파
y.grad = np.array(1.0)
y.backward()
print(a.grad)
print(x.grad)
