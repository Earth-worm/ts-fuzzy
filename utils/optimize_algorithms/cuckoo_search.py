import math
import numpy as np
import time
from utils.ts_fuzzy.ts_fuzzy import tsfuzzy
from utils.ts_fuzzy.actfunc import ActiveFunc
import copy

class Nest:
  value = None
  eggs = None
  ns = None

  def __init__(self,NestSet):
    self.ns = NestSet
    self.eggs = self.ns.ts.init_par()
    self.value = self.evaluate(self.eggs,self.ns.y_train,self.ns.X_train)

  def __del__(self):
    del self.eggs

  def levy_flight(self):
    nume = math.gamma(1+self.ns.Beta)*math.sin(math.pi * self.ns.Beta /2)
    denom = math.gamma((1+self.ns.Beta)/2)*self.ns.Beta*(2**((self.ns.Beta-1)/2))
    sigma = (nume/denom)**(1/self.ns.Beta)
    u = np.random.normal(0,1)*sigma
    v = np.random.normal(0,1)
    s = u/(abs(v)**1/self.ns.Beta)
    return (self.ns.Alpha * s)

  def replace(self,base):
    newEggs = copy.deepcopy(self.eggs)
    for i in range(self.ns.ts.exVarNum):
      for j in range(self.ns.ts.Node):
        newEggs["alpha"][i][j]=max(min(self.ns.ts.ParMax,base.eggs["alpha"][i][j] + self.levy_flight()),0.001)
        newEggs["beta"][i][j]=max(min(self.ns.ts.ParMax,base.eggs["beta"][i][j] + self.levy_flight()),-self.ns.ts.ParMax)
        newEggs["weight"][i][j]=max(min(self.ns.ts.ParMax,base.eggs["weight"][i][j] + self.levy_flight()),-self.ns.ts.ParMax)
    newValue = self.evaluate(newEggs,self.ns.y_train,self.ns.X_train)
    if(newValue < self.value):
      self.eggs = newEggs
      self.value = newValue

  
  def abandon(self):
    self.eggs = self.ns.ts.init_par()
    self.value = self.evaluate(self.eggs,self.ns.y_train,self.ns.X_train)

  def evaluate(self,eggs,y,x):
    return self.ns.ts.Q(eggs,y,x)


class CS:
  #cuckoo search
  nests = None
  Gen = None
  SetSize = None
  AbaRate = None
  Alpha = None
  Beta = None
  ts = None
  X_train = None
  X_test = None
  y_train = None
  y_test = None

  def __init__(self,Gen,SetSize,AbaRate,Alpha,Beta,ParMax,M,C,
               X_src,y_src,X_train,y_train,X_test,y_test,Node=3,ActFunc = ActiveFunc.sigmoid):
    self.Gen = Gen
    self.SetSize = SetSize
    self.AbaRate = AbaRate
    self.Alpha = Alpha
    self.Beta = Beta
    self.X_test = X_test
    self.y_test = y_test
    self.X_train = X_train
    self.y_train = y_train
    self.ts = tsfuzzy(X_src,y_src,M,C,Node,ParMax,ActFunc)
    self.nests = []

    for i in range(self.SetSize):
      self.nests.append(Nest(self))

  def __del__(self):
    del self.nests
  
  def alternate(self):
    r1 = np.random.randint(0,self.SetSize-1)
    r2 = (r1 + np.random.randint(0,self.SetSize - 2) + 1)%self.SetSize
    self.nests[r2].replace(self.nests[r1])

    for i in range(self.SetSize - int(self.AbaRate * self.SetSize),self.SetSize):
      self.nests[i].abandon()
    
    self.sort(0,self.SetSize-1)
  
  def sort(self,lb,ub):
    if(lb<ub):
      k=int((lb+ub)/2)
      pivot=self.nests[k].value
      i = lb
      j = ub
      while(True):
        while(self.nests[i].value < pivot):
          i+=1
        while(self.nests[j].value > pivot):
          j-=1
        if(i<=j):
          self.nests[i],self.nests[j] = self.nests[j],self.nests[i]
          i+=1
          j-=1
        else:
          break
      self.sort(lb,j)
      self.sort(i,ub)
  
  def learn(self):
    score_best = []
    for i in range(self.Gen):
        score = self.ts.Q(self.nests[0].eggs,self.y_test,self.X_test)
        score_best.append(score)
        self.alternate()
    return self.nests[0].value,score_best