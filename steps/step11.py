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
    def __call__(self, inputs):
        xs = [x.data for x in inputs] # x = input.data # 데이터를 꺼냄
        ys = self.forward(xs) # y = self.forward(x) # 실제 계산
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self, xs): # Function 클래스의 forward 메서드는 예외를 발생시킴
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)
    
xs = [Variable(np.array(2)), Variable(np.array(3))] # 리스트로 준비
f = Add()
ys = f(xs) # ys 튜플
y = ys[0]
print(y.data)
