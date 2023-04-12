import numpy as np
import skfuzzy.cluster as fuzz_cmeans #c-平均法
from sklearn import metrics

class ActFuncPar():
  Alpha = None
  Beta = None
  Weight = None
  def __init__(self,alpha,beta,weight):
    self.Alpha = alpha
    self.Beta = beta
    self.Weight = weight

class TSFuzzy():
    ActFunc = None
    M = None
    C = None
    Node = None
    linear_coef = None
    cntr = None
    X_tgt = None
    y_tgt = None
    exVarNum = None
    tgt_size = None
    ParMax = None

    def __init__(self,X_src,y_src,X_tgt,y_tgt,M,C,Node,ParMax,ActFunc):
      self.M = M
      self.C = C
      self.Node = Node
      self.exVarNum = X_src.shape[1]
      self.tgt_size = X_tgt.shape[0]
      self.X_tgt = X_tgt
      self.y_tgt = y_tgt
      self.ParMax = ParMax
      self.ActFunc = ActFunc
      self.cntr,u = self._cmeans(X_src)
      self.linear_coef = self._init_linear_coef(X_src,y_src,u)

    def init_par(self,zero = False):
      if(zero):
        alpha = np.zeros((self.exVarNum,self.Node))
        beta = np.zeros((self.exVarNum,self.Node))
        weight = np.zeros((self.exVarNum,self.Node))
        return ActFuncPar(alpha,beta,weight)
      alpha = np.random.uniform(0.001,self.ParMax,(self.exVarNum,self.Node))
      beta = np.random.uniform(-self.ParMax,self.ParMax,(self.exVarNum,self.Node))
      weight = np.random.uniform(-self.ParMax,self.ParMax,(self.exVarNum,self.Node))
      return ActFuncPar(alpha,beta,weight)

    def phi(self,x,par): #マッピング関数
      fixed_x = np.zeros_like(x)
      fixed_cntr = np.zeros_like(self.cntr)
      x_num = x.shape[0]
      sum1 = 0
      sum2 = 0
      for n in range(x_num):
        for m in range(self.exVarNum):
          sum1 = 0
          for p in range(self.Node):
            sum1=sum1+par.Weight[m][p]*self.ActFunc(par.Alpha[m][p]*(x[n][m]-par.Beta[m][p]))
          fixed_x[n][m] = sum1

      for c in range(self.C):
        for m in range(self.exVarNum):
          sum2 = 0
          for p in range(self.Node):
            sum2 = sum2 + par.Weight[m][p]*self.ActFunc(par.Alpha[m][p]*(self.cntr[c][m]-par.Beta[m][p]))
          fixed_cntr[c][m] = sum2
      return fixed_x,fixed_cntr

    def predict(self,x,par=None):#t-s fuzzyモデルによる予測
      if par is None:
        fixed_X_tgt = x
        fixed_cntr = self.cntr
      else:
        fixed_X_tgt,fixed_cntr = self.phi(x,par)
      x_num = x.shape[0]
      y = np.zeros(x.shape[0])
      A = self._cmeans_predict(fixed_X_tgt,fixed_cntr)
      for i in range(x_num):
        wk = 0
        for j in range(self.C):
          wk = wk + A[i][j]*self._L(fixed_X_tgt[i],self.linear_coef[j])
          y[i] = wk
      return y

    def Q(self,par,y,x):
      return metrics.mean_absolute_error(y,self.predict(x,par))
      #return metrics.mean_squared_error(y,self.predict(par,x))

    def _init_linear_coef(self,x,y,u): #線形関数の計算
      src_size = len(y)
      f = np.zeros((src_size,self.C*(self.exVarNum+1)))
      for i in range(src_size):
        for j in range(self.C):
          f[i][(self.exVarNum+1)*j] = u[i][j]
          for k in range(self.exVarNum):
            f[i][(self.exVarNum+1)*j+k+1] = u[i][j]*x[i][k]
      a = np.dot(np.dot(np.linalg.pinv(np.dot(f.T,f)),f.T),y)
      return a.reshape(self.C,self.exVarNum+1)

    def _cmeans(self,data): #c平均法モデルの計算
      cntr,u,_,_,_,_,_ = fuzz_cmeans.cmeans(data.T,self.C,self.M,0.005,10000)
      return cntr,u.T

    def _L(self,x,coef): #線形関数の出力を求める関数
      sum = coef[0]
      for i in range(self.exVarNum):
        sum = sum + coef[i+1]*x[i]
      return sum

    def _cmeans_predict(self,data,cntr): #c-means予測 ,x,重心
      u,_,_,_,_,_ = fuzz_cmeans.cmeans_predict(data.T,cntr,self.M,error=0.005,maxiter=10000)
      return u.T