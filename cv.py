
# coding: utf-8

# In[24]:

from sklearn import datasets, cross_validation, svm
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, make_scorer


iris = datasets.load_iris()

iris = datasets.load_iris()
   
X = iris.data

df = pd.DataFrame(X, columns = iris.feature_names)
# 1 = setosa 2 = versicolor 3 = virginica
df['iris type'] = iris.target
df['iris type'] = np.where(df['iris type']==0, iris.target_names[0],np.where(df['iris type']==1, iris.target_names[1], iris.target_names[2]))
df['iris type'] = df['iris type'].astype(object)
print(df.head())

X = df[[0,1,2,3]].values
y = df['iris type'].values


# In[25]:

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.4, random_state=42)


# In[26]:

svc = svm.SVC(kernel='linear')
svc.fit(X_train, y_train)

results = svc.predict(X_test)


# In[27]:

correct_count = 0
incorrect_count = 0
        
print(accuracy_score(y_test, results))


# In[30]:


cv_score_5 = cross_validation.cross_val_score(svc,X,y,cv=5)


# In[33]:

print('cross val score %s' % cv_score)
print('mean %s' % cv_score.mean())
print('variance %s' % cv_score.std()**2)


# In[35]:

#cv_score_5_f1 = cross_validation.cross_val_score(svc,X,y,cv=5, scoring = make_scorer(f1_score, average = None))
#not sure why this doesn't work


cv_score_5_f1 = f1_score(y_test, results, average=None)


# In[36]:

print(cv_score_5_f1)


# In[ ]:




# In[ ]:



