import numpy as np
import sys

class Khold():
    indexArr = None
    y = None
    x = None
    k = None
    i = 0
    dataNum = None
    testDataNum = None
    def __init__(self,x,y,k):
        if(len(x) != len(y)):
            print("input array must have same element num",file=sys.stderr)
            sys.exit(1)
        self.dataNum = len(x)
        self.indexArr = np.array([i for i in range(self.dataNum)])
        np.random.shuffle(self.indexArr)
        self.y = y
        self.x = x
        self.k = k
        self.testDataNum = int(self.dataNum/self.k)

    def __iter__(self):
        return self

    def __next__(self):
        if(self.i == self.k):
            raise StopIteration()
        x_train = np.array([self.x[self.indexArr[i]] for i in range(self.i*self.testDataNum,(self.i+1)*self.testDataNum)])
        y_train = np.array([self.y[self.indexArr[i]] for i in range(self.i*self.testDataNum,(self.i+1)*self.testDataNum)])
        x_test = np.array([self.x[self.indexArr[i]] for i in list(range(0,self.i*self.testDataNum))+list(range((self.i+1)*self.testDataNum,self.dataNum))])
        y_test = np.array([self.y[self.indexArr[i]] for i in list(range(0,self.i*self.testDataNum))+list(range((self.i+1)*self.testDataNum,self.dataNum))])
        self.i += 1
        return x_train,y_train,x_test,y_test
