import numpy as np
import sys

"""
x.shape: (100,8)
y.shape: (100,)

khold = Khold(x,y,10)
for x_train,y_train,x_test,y_test in khold:
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    # iterate 10 times
"""
class Khold():
    indexArr = None
    y = None
    x = None
    k = None
    i = 0
    dataNum = None
    trainDataNum = None
    def __init__(self,x,y,k):
        if(len(x) != len(y)):
            print("Khold input data error\ninput array must have same length\ny:",len(y),"x:",len(x),file=sys.stderr)
            sys.exit(1)
        self.dataNum = len(x)
        self.indexArr = np.array([i for i in range(self.dataNum)])
        np.random.shuffle(self.indexArr)
        self.y = y
        self.x = x
        self.k = k
        self.trainDataNum = int(self.dataNum/self.k)

    def __iter__(self):
        return self

    def __next__(self):
        if(self.i == self.k):
            raise StopIteration()
        x_test = np.array([self.x[self.indexArr[i]] for i in range(self.i*self.trainDataNum,(self.i+1)*self.trainDataNum)])
        y_test = np.array([self.y[self.indexArr[i]] for i in range(self.i*self.trainDataNum,(self.i+1)*self.trainDataNum)])
        x_train = np.array([self.x[self.indexArr[i]] for i in list(range(0,self.i*self.trainDataNum))+list(range((self.i+1)*self.trainDataNum,self.dataNum))])
        y_train = np.array([self.y[self.indexArr[i]] for i in list(range(0,self.i*self.trainDataNum))+list(range((self.i+1)*self.trainDataNum,self.dataNum))])
        self.i += 1
        return x_train,y_train,x_test,y_test