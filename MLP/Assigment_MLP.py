import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
   def __init__(self, rate , niter):
      self.rate = rate
      self.niter = niter

   def set(self, X, y):
      

      
      self.weight = np.zeros(1 + X.shape[1])

      
      self.errors = []  

      for i in range(self.niter):
         err = 0
         for xi, target in zip(X, y):
            delta_w = self.rate * float(target - self.pred(xi))
            self.weight[1:] += delta_w * xi
            self.weight[0] += delta_w
            err += int(delta_w != 0.0)
         self.errors.append(err)
      return self

   def net_input(self, X):
      
      return np.dot(X, self.weight[1:]) + self.weight[0]

   def pred(self, X):
      
      return np.where(self.net_input(X) >= 0.0, 1, -1)

df = pd.read_csv('Data.csv', header=None)
print("The weights are")
# print(df)
y = df.iloc[0:100, 4].values
y = np.where(y == 1, -1, 1)
x = df.iloc[0:100, 0:4].values
print(x)
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='RED')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='BLUE')
plt.xlabel('XAXIS')
plt.ylabel('YAXIS')
plt.legend(loc='upper left')
plt.show()
print ("---Predicted Values---")
pn = Perceptron(0.1, 5)
pn.set(x, y)
print(pn.weight)
plt.plot(range(1, len(pn.errors) + 1), pn.errors)
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

if pn.pred([5.2,2.7,3.9,1.4]) == -1:

    print("number 1")
else:
    print("number 2")
