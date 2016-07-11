
# coding: utf-8

# In[17]:

from sklearn import datasets
from sklearn import svm
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import itertools as iter

iris = datasets.load_iris()

import matplotlib.pyplot as plt
plt.scatter(iris.data[:, 1], iris.data[:, 2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])

plt.show()


# In[18]:

plt.scatter(iris.data[0:100, 1], iris.data[0:100, 2], c=iris.target[0:100])
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])

plt.show()


# In[19]:


svc = svm.SVC(kernel='linear', C = 1)
X = iris.data[0:100, 1:3]
y = iris.target[0:100]
svc.fit(X, y)


# In[32]:


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    #Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()

plot_estimator(svc, X, y)
plt.show()
print(iris.feature_names)


# In[67]:

def combosfor2flowertypes(low,upper):
    x = slice(low,upper)
    #sep length and sep width
    plot_estimator(svc, iris.data[x,0:2],iris.target[x])
    plt.show()
    #sep lenth and petal length
    plot_estimator(svc, iris.data[x,0:3:2],iris.target[x])
    plt.show()
    #sep lenth and petal width
    plot_estimator(svc, iris.data[x,0:4:2],iris.target[x])
    plt.show()
    
    #sep width and petal lenth
    plot_estimator(svc, iris.data[x,1:3],iris.target[x])
    plt.show()
    #sep width and petal width
    plot_estimator(svc, iris.data[x,1:4:2],iris.target[x])
    plt.show()
    
    #petal length and petal width
    plot_estimator(svc, iris.data[x,2:4],iris.target[x])
    plt.show()


# 1 = setosa 2 = versicolor 3 = virginica
print(iris.target)
print(len(iris.target))
plot_estimator(svc, iris.data[50:150,2:4], iris.target[50:150])
plt.show()


# In[69]:

combosfor2flowertypes(0,100)
combosfor2flowertypes(50,150)
combosfor2flowertypes(0,150)


# In[85]:

z = np.concatenate([iris.target[0:50],iris.target[100:150]])
print(z)
y = np.concatenate([iris.data[0:50],iris.data[100:150]])
print(len(y))
print(y)


# In[86]:

#got it to work for the other two, should factor this in

plot_estimator(svc, y[:,0:2],z)
plt.show()
#sep lenth and petal length
plot_estimator(svc, y[:,0:3:2],z)
plt.show()
#sep lenth and petal width
plot_estimator(svc, y[:,0:4:2],z)
plt.show()
#sep width and petal lenth
plot_estimator(svc, y[:,1:3],z)
plt.show()
#sep width and petal width
plot_estimator(svc, y[:,1:4:2],z)
plt.show()
    
#petal length and petal width
plot_estimator(svc, y[:,2:4],z)
plt.show()


# In[ ]:



