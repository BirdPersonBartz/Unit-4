
# coding: utf-8

# In[2]:

import pandas as pd
import matplotlib.pyplot as plt
from random import randint

colnames = pd.read_csv('/Users/Thomas/Projects/Unit 4/iriscolumnnames.csv')
df = pd.read_csv('/Users/Thomas/Projects/Unit 4/irisdata.csv', names = colnames)

print(df.head())


# In[3]:

plt.scatter(df['sepal length in cm'],df['sepal width in cm'])
plt.show()


# In[4]:

print(len(df))


# In[8]:



rndm_pt = randint(0,len(df))
print(rndm_pt)


# In[73]:

def graphdist(index):
    dist = (((df.ix[index,'sepal length in cm']) - x1)**2 + ((df.ix[rndm_pt,'sepal width in cm']) - y1)**2)**(1/2)
    dist = round(dist, 6)
    return(dist)

df['distance'] = 0


x1 = df.ix[rndm_pt,'sepal length in cm']
y1 = df.ix[rndm_pt,'sepal width in cm']

print(x1)
print(y1)


# In[ ]:




# In[58]:

for index, row in df.iterrows():
    df.ix[index, 'distance'] = graphdist(index)


# In[72]:

df = df.sort_values(['distance'], ascending = 1)


# In[71]:

df_knn = df.head(n=10)


# In[70]:

knn1 = df_knn['iris class'].describe()
print(knn1['top'])


# In[76]:

def knn(k):
    new_df = df.head(n = k)
    flowertype = new_df['iris class'].describe()
    print(flowertype['top'])
    


# In[77]:

knn(10)


# In[78]:

knn(50)


# In[79]:

knn(100)


# In[ ]:



