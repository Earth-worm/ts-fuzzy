import numpy as np
class ActiveFunc(): #活性化関数まとめ
  def relu(x):
    return max(x,0)
  
  def sigmoid(x):
    sigmoid_range = 34.538776394910684
    if x <= -sigmoid_range:
      return 1e-15
    if x >= sigmoid_range:
      return 1.0 - 1e-15
    return 1.0 / (1.0 + np.exp(-x))