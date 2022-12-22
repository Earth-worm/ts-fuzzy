import numpy as np
import gc
import skfuzzy.cluster as fuzz_cmeans #c-平均法
from sklearn import metrics
from utils.ts_fuzzy.actfunc import ActiveFunc
import sys
import copy

#GA
class GA_TSfuzzy():
  #ファジィ用パラメータ
  M = None #ファジィ度
  C = None #クラスタ数

  #GA用パラメータ
  Gen = None #世代数
  Mut = None #突然変異率
  PopSize = None #解集団数
  EliteNum =None #エリート選択数

  #ニューラルネット用パラメータ
  Node = None #ノード数
  ActFunc = None #活性化関数
  parMax = None #係数の取りえる最大値

  #入力データ
  D =None #入力データ次元数
  X_train = None
  y_train = None
  X_test = None
  y_test =None
  dataNum = None

  ng_pop = None #次世代プール
  pop = None #現在プール
  score_avg = None
  score_best = None

  #ソースデータでの学習結果
  cntr = None #ソースデータの重心
  linear_coef = None #線形関数の係数

  def __init__(self,M,C,Gen,Mut,PopSize,X_src,y_src,X_train,y_train,X_test,y_test,Node = 3,EliteNum = 5,ActFunc=ActiveFunc.sigmoid,parMax =150):
    self.M = M
    self.C = C
    self.Gen = Gen
    self.Mut = Mut
    self.PopSize = PopSize
    self.Node = Node
    self.EliteNum = EliteNum
    self.ActFunc = ActFunc
    self.parMax = parMax
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.D = self.X_test.shape[1]
    self.dataNum =self.X_test.shape[0]
    self.cntr,u = self.fuzzy_cmeans(X_src)
    self.linear_coef = self.opt_linearfunc(X_src,y_src,u)
    self.pop = self.init_actfunc_parameter()
    self.ng_pop = self.init_actfunc_parameter()
  
  def __del(self):
    del self.pop
    del self.ng_pop
    del self.score_avg
    del self.score_best
    del self.linear_coef
    gc.collect()


  def init_actfunc_parameter(self): #解集団の生成
    alpha = np.random.uniform(0.001,self.parMax,(self.PopSize,self.D,self.Node))
    beta = np.random.uniform(-self.parMax,self.parMax,(self.PopSize,self.D,self.Node))
    weight = np.random.uniform(-self.parMax,self.parMax,(self.PopSize,self.D,self.Node))
    return {"alpha":alpha,"beta":beta,"weight":weight}

  def fuzzy_cmeans(self,data): #c-meansの計算 x,重心数,ファジィ度
    cntr,u,_,_,_,_,_ = fuzz_cmeans.cmeans(data.T,self.C,self.M,0.005,10000)
    return cntr,u.T

  def L(self,x,a): #線形関数の出力を求める関数
    sum = a[0]
    for i in range(len(x)):
      sum = sum + a[i+1]*x[i]
    return sum

  def fuzzy_cmeans_predict(self,data,cntr): #c-means予測 ,x,重心,ファジィ度
    u,_,_,_,_,_ = fuzz_cmeans.cmeans_predict(data.T,cntr,self.M,error=0.005,maxiter=10000)
    return u.T

  def ts_fuzzy_predict(self,x,cntr):#t-s fuzzyモデルによる予測
    x_size = len(x)
    y = np.zeros(x_size)
    A = self.fuzzy_cmeans_predict(x,cntr)
    for i in range(x_size):
      wk = 0
      for j in range(self.C):
        wk = wk + A[i][j]*self.L(x[i],self.linear_coef[j])
        y[i] = wk
    return y

  def opt_linearfunc(self,x,y,u): #cNum:クラスターの数
    f = np.zeros((len(y),self.C*(self.D+1)))
    for i in range(len(x)):
      for j in range(self.C):
        f[i][(self.D+1)*j] = u[i][j]
        for k in range(self.D):
          f[i][(self.D+1)*j+k+1] = u[i][j]*x[i][k]
    
    a = np.dot(np.dot(np.linalg.pinv(np.dot(f.T,f)),f.T),y)
    return a.reshape(self.C,self.D+1)

  def Q(self,y_true,y_pred): #目的関数
    return metrics.mean_squared_error(y_true,y_pred) #平均二乗誤差
    #return metrics.mean_absolute_error(y_true,y_pred) #平均絶対誤差)

  def phi(self,alpha,beta,weight,X_data): #マッピング関数
    X_data_size = len(X_data)
    fixed_x = np.zeros_like(X_data)
    fixed_cntr = np.zeros_like(self.cntr)
    sum1 = 0
    sum2 = 0
    for n in range(X_data_size):
      for m in range(self.D):
        sum1 = 0
        for p in range(self.Node):
          sum1=sum1+weight[m][p]*self.ActFunc(alpha[m][p]*(X_data[n][m]-beta[m][p]))
        fixed_x[n][m] = sum1
    
    for c in range(self.C):
      for m in range(self.D):
        sum2 = 0
        for p in range(self.Node):
          sum2 = sum2 + weight[m][p]*self.ActFunc(alpha[m][p]*(self.cntr[c][m]-beta[m][p]))
        fixed_cntr[c][m] = sum2
    return fixed_x,fixed_cntr

  def binsearch(self,arr,tar,begin,end): #二分探索(ルーレット用)
    if(end-begin == 1):
      if(begin!=end and arr[begin+1]==arr[begin]):
        begin = begin+1
      return begin
    middle = int((end-begin)/2)
    if(arr[begin+middle]==tar):
      if(begin+middle!=end and arr[begin+middle+1]==arr[begin+middle]):
        middle = middle + 1
      return begin+middle
    if(arr[begin+middle] < tar):
      return self.binsearch(arr,tar,begin+middle,end)
    else:
      return self.binsearch(arr,tar,begin,begin+middle)

  def selectPar(self,roulette): #親の選択(ルーレット選択)
    dataNum = len(roulette)
    sum = roulette[dataNum-1]
    rnd = np.random.uniform(0,sum)
    mama = self.binsearch(roulette,rnd,0,dataNum)
    papa = mama
    while(papa == mama):
      rnd = np.random.uniform(0,sum)
      papa = self.binsearch(roulette,rnd,0,dataNum+1)
    return mama,papa

  def evalfit(self,target_pop): #各評価値の算出
    PopSize = target_pop["alpha"].shape[0]
    fit_table = np.zeros(PopSize)
    fit_sum_table = np.zeros(PopSize+1)
    best = sys.float_info.max
    for i in range(PopSize):
      fixed_X_target,fixed_cntr = self.phi(target_pop["alpha"][i],target_pop["beta"][i],target_pop["weight"][i],self.X_train)
      y_pred = self.ts_fuzzy_predict(fixed_X_target,fixed_cntr)
      fit_table[i] = self.Q(self.y_train,y_pred)
      if(fit_table[i] < best):
        best = fit_table[i]
      sum = np.sum(fit_table)
    for i in range(PopSize):
      fit_table[i] = 1/fit_table[i]
      fit_sum_table[i+1] = fit_sum_table[i] + fit_table[i]
    return fit_table,fit_sum_table,best

  def cr2to2(self,papa,mama,point = 2):
    child1 = copy.deepcopy(papa)
    child2 = copy.deepcopy(mama)
    if(point==2):
      rndD = np.random.randint(0,self.D)
      rndNODE = np.random.randint(0,self.Node)
      for i in range(self.D):
        for j in range(self.Node):
          if((i<rndD and j>=rndNODE) or (i>=rndD and j<rndNODE)):
            child1[i][j] = mama[i][j]
            child2[i][j] = papa[i][j]
    if(point==1):
      rndNODE = np.random.randint(0,self.Node)
      for i in range(self.D):
        for j in range(self.Node):
          if(j<=rndNODE):
            child1[i][j] = mama[i][j]
            child2[i][j] = papa[i][j]
    return child1,child2

  def matting(self,point = 2):
    fit_table,roulette,_ = self.evalfit(self.pop)
    for i in range(int(self.PopSize/2)):
      papa,mama = self.selectPar(roulette)
      self.ng_pop["alpha"][2*i],self.ng_pop["alpha"][2*i+1]=self.cr2to2(self.pop["alpha"][papa],self.pop["alpha"][mama],point=point)
      self.ng_pop["beta"][2*i],self.ng_pop["beta"][2*i+1]=self.cr2to2(self.pop["beta"][papa],self.pop["beta"][mama],point=point)
      self.ng_pop["weight"][2*i],self.ng_pop["weight"][2*i+1]=self.cr2to2(self.pop["weight"][papa],self.pop["weight"][mama],point=point)
  
  def mutation(self):
    for i in range(self.PopSize):
      for j in range(self.D):
        for k in range(self.Node):
          rnd = np.random.rand()
          if(rnd<=self.Mut):
            self.pop["alpha"][i][j][k] = np.random.uniform(0.001,self.parMax)
          rnd = np.random.rand()
          if(rnd<=self.Mut):
            self.pop["beta"][i][j][k] = np.random.uniform(-self.parMax,self.parMax)
          rnd = np.random.rand()
          if(rnd<=self.Mut):
            self.pop["weight"][i][j][k] = np.random.uniform(-self.parMax,self.parMax)

  def select_next_gen(self):
    pool = {"alpha":np.concatenate([self.pop["alpha"],self.ng_pop["alpha"]]),
            "beta":np.concatenate([self.pop["beta"],self.ng_pop["beta"]]),
            "weight":np.concatenate([self.pop["weight"],self.ng_pop["weight"]])}
    fit_table,roulette,best_score=self.evalfit(pool)
    best_arg = np.argmax(fit_table)
    for i in range(self.EliteNum):
      max = np.argmax(fit_table)
      fit_table[max] = 0
      self.pop["alpha"][i] = np.copy(pool["alpha"][max])
      self.pop["beta"][i] = np.copy(pool["beta"][max])
      self.pop["weight"][i] = np.copy(pool["weight"][max])
    sum = roulette[2*self.PopSize-1]
    for i in range(self.EliteNum,self.PopSize):
      rnd = np.random.uniform(0,sum)
      next = self.binsearch(roulette,rnd,0,len(roulette))
      self.pop["alpha"][i] = np.copy(pool["alpha"][next])
      self.pop["beta"][i] = np.copy(pool["beta"][next])
      self.pop["weight"][i] = np.copy(pool["weight"][next])
    return best_score

  def learn(self):
    best = 100000000
    score_best = []
    
    for i in range(self.Gen):
        self.matting()
        self.mutation()
        score_best.append(self.select_next_gen())

    return best,score_best