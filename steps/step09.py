import numpy as np

class Variable:
    def __init__(self, data): # __init__ : 초기화 함수, data : 인스턴스 변수
        if data is not None:
            if not isinstance(data, np.ndarray): # ndarray만 취급
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
            
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func):
        self.creator = func

    def backward(self): # 반복
        if self.grad is None:
            self.grad = np.ones_like(self.data) # y.grad = np.array(1.0)
            
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 함수를 가져옴
            x, y = f.input, f.output # 함수의 입력과 출력을 가져옴
            x.grad = f.backward(y.grad) # 함수의 backward 메서드 호출

            if x.creator is not None:
                funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가

def as_array(x): # 스칼라 값을 ndarray로 변환
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input):
        x = input.data # 데이터를 꺼냄
        y = self.forward(x) # 실제 계산
        output = Variable(as_array(y)) # Variable 형태로 되돌림
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
    
def square(x):
    # f = Square()
    # return f(x)
    return Square()(x)

def exp(x):
    # f = Exp()
    # return f(x)
    return Exp()(x)

x = Variable(np.array(0.5))
y = square(exp(square(x))) # a = Square(x), b = Exp(a), y = Square(b)
y.backward()
print(x.grad)

x = Variable(np.array(1.0)) # O
x = Variable(None) # O
x = Variable(1.0) # X, TypeError
