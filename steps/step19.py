import numpy as np
import weakref # weak reference (약한 참조)
import contextlib

class Config:
    enable_backprop = True
    
@contextlib.contextmanager # 전처리와 후처리 구현 가능
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value) # 전처리
    try:
        yield
    finally:
        setattr(Config, name, old_value) # 후처리

def no_grad(): # 기울기가 필요없을 때는 no_grad 함수 호출
    return using_config('enable_backprop', False)
    
class Variable:
    def __init__(self, data, name=None): # __init__ : 초기화 함수, data : 인스턴스 변수
        if data is not None:
            if not isinstance(data, np.ndarray): # ndarray만 취급
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
            
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
        
    @property # property 덕분에 shape를 인스턴스 변수처럼 사용 가능
    def shape(self):
        return self.data.shape # self.data는 np.ndarray임

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self): # Variable 인스턴스에 대해서도 len 함수 사용 가능
        return len(self.data)
        
    def __repr__(self): # Variable 내용을 쉽게 확인할 수 있는 기능(print 함수를 사용하여 Variable 안의 데이터 내용을 출력)
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
        
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self): # 필요시 호출하여 grad값을 리셋
        self.grad = None
        
    def backward(self, retain_grad=False): # 반복
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
            gys = [output().grad for output in f.outputs]
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

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y는 weakref
                    
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

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            
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

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = Variable(None)
x.name = 'x'

print(x.name) # x
print(x.shape) # (2, 3)
print(len(x)) # 2
print(x) # Variable([[1, 2, 3] [4, 5, 6]])
print(y) # Variable(None)
