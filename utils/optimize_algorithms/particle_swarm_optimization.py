import copy
from pyexpat.errors import XML_ERROR_TAG_MISMATCH
import time
import sys
from utils.ts_fuzzy.ts_fuzzy import TSFuzzy
from utils.ts_fuzzy.actfunc import ActiveFunc
import random

#PSO
class Particle:
  pBestPos = None #パーソナルベストの解
  pBestValue = None  #パーソナルベストの値
  swarm = None #群れのクラス
  v = None #前回の速度
  pos = None #今の座標
  value = None #今の評価値

  def __init__(self,swarm):
    self.swarm = swarm
    self.init_Pos()
    self.pBestPos = copy.deepcopy(self.pos)
    self.evaluate(self.swarm.y_test,self.swarm.X_test)
    self.pBestValue = self.value

  def init_Pos(self):
    self.pos = self.swarm.ts.init_par()
    self.v = self.swarm.ts.init_par(zero=True)

  def evaluate(self,y,x):
    self.value = self.swarm.ts.Q(self.pos,y,x)

  def move(self):
    for ex in range(self.swarm.ts.exVarNum):
      for node in range(self.swarm.ts.Node):
        self.v.Alpha[ex][node] = self.v.Alpha[ex][node]*self.swarm.W+self.swarm.CG*random.uniform(0,1)*(self.swarm.gBestPos.Alpha[ex][node]-self.pos.Alpha[ex][node])+self.swarm.CP*random.uniform(0,1)*(self.pBestPos.Alpha[ex][node]-self.pos.Alpha[ex][node])
        self.v.Beta[ex][node] = self.v.Beta[ex][node]*self.swarm.W+self.swarm.CG*random.uniform(0,1)*(self.swarm.gBestPos.Beta[ex][node]-self.pos.Beta[ex][node])+self.swarm.CP*random.uniform(0,1)*(self.pBestPos.Beta[ex][node]-self.pos.Beta[ex][node])
        self.v.Weight[ex][node] = self.v.Weight[ex][node]*self.swarm.W+self.swarm.CG*random.uniform(0,1)*(self.swarm.gBestPos.Weight[ex][node]-self.pos.Weight[ex][node])+self.swarm.CP*random.uniform(0,1)*(self.pBestPos.Weight[ex][node]-self.pos.Weight[ex][node])
        self.pos.Alpha[ex][node] += self.v.Alpha[ex][node]
        self.pos.Beta[ex][node] += self.v.Beta[ex][node]
        self.pos.Weight[ex][node] += self.v.Weight[ex][node]
        self.pos.Alpha[ex][node] = min(max(self.pos.Alpha[ex][node],0.001),self.swarm.ts.ParMax)
        self.pos.Beta[ex][node] = min(max(self.pos.Beta[ex][node],-self.swarm.ts.ParMax),self.swarm.ts.ParMax)
        self.pos.Weight[ex][node] = min(max(self.pos.Weight[ex][node],-self.swarm.ts.ParMax),self.swarm.ts.ParMax)
    self.evaluate(self.swarm.y_test,self.swarm.X_test)
    if(self.value < self.pBestValue):
      self.pBestValue = self.value
      self.pBestPos = copy.deepcopy(self.pos)

class PSO:
  Size = None #群れのサイズ
  TMax = None #ステップ数
  W = None #慣性係数
  CP = None #パーソナル加速係数
  CG = None #グローバル加速係数

  particles = None
  gBestValue = None
  gBestPos = None
  ts = None

  X_train = None
  y_train = None
  X_test = None
  y_test = None

  def __init__(self,size,tMax,w,cp,cg,parMax,X_src,y_src,X_tgt,y_tgt,X_test,y_test,M,C,Node=3,ActFunc=ActiveFunc.sigmoid):
    self.Size = size
    self.TMax = tMax
    self.W = w
    self.CP = cp
    self.CG = cg
    self.particles = []
    self.gBestValue = sys.float_info.max
    self.X_test = X_test
    self.y_test = y_test
    self.X_train = X_src
    self.y_train = y_src

    self.ts = TSFuzzy(X_tgt,y_tgt,X_src,y_src,M,C,Node,parMax,ActFunc)
    for i in range(self.Size):
      self.particles.append(Particle(self))
      if(self.gBestValue > self.particles[i].pBestValue):
        self.gBestValue = self.particles[i].pBestValue
        self.gBestPos = copy.deepcopy(self.particles[i].pBestPos)

  def move(self):
    for particle in self.particles:
      particle.move()
      if(particle.pBestValue < self.gBestValue):
        self.gBestValue = particle.pBestValue
        self.gBestPos = copy.deepcopy(particle.pBestPos)

  def learn(self):
    trans = []
    for i in range(self.TMax):
      score = self.ts.Q(self.gBestPos,self.y_test,self.X_test)
      trans.append(score)
      self.move()
    return self.gBestValue,self.ts.predict(self.X_test,self.gBestPos),trans,self.gBestPos,