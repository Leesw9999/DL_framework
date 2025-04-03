import numpy as np

class Variable:
    def __init__(self, data): # __init__ : 초기화 함수, data : 인스턴스 변수
        if data is not None:
            if not isinstance(data, np.ndarray): # ndarray만 취급
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
            
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self): # 필요시 호출하여 grad값을 리셋
        self.grad = None
        
    def backward(self): # 반복
        if self.grad is None:
            self.grad = np.ones_like(self.data) # y.grad = np.array(1.0)
            
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation) # funcs를 항상 정렬 상태로 유지

        add_func(self.creator)
        
        while funcs:
            f = funcs.pop() # 함수를 가져옴
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator) # 수정 전 funcs.append(x.creator)

def as_array(x): # 스칼라 값을 ndarray로 변환
    if np.isscalar(x):
        return np.array(x)
    return x
    
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 호출시 별표를 붙이면 리스트 언팩(리스트의 원소를 낱개로 풀어서 전달)
        if not isinstance(ys, tuple): # 튜플이 아닌 경우 추가 지원
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0] # 리스트의 원소가 하나라면 첫 번째 원소를 반환

    def forward(self, xs): # Function 클래스의 forward 메서드는 예외를 발생시킴
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
def add(x0, x1):
    return Add()(x0, x1)

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data # 수정 전 -> x = self.input.data
        gx = 2 * x * gy
        return gx

def square(x):
    f = Square()
    return f(x)

x = Variable(np.array(2))
a = square(x)
y = add(square(a), square(a)) # (a^2)^2 + (a^2)^2 = 2a^4
y.backward()

print(y.data)
print(x.grad)
