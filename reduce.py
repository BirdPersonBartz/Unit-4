
# coding: utf-8

# In[499]:

from sklearn import datasets
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sea
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint

#load data

df = pd.DataFrame(X, columns = iris.feature_names)
# 1 = setosa 2 = versicolor 3 = virginica
df['iris type'] = iris.target
df['iris type'] = np.where(df['iris type']==0, iris.target_names[0],np.where(df['iris type']==1, iris.target_names[1], iris.target_names[2]))
df['iris type'] = df['iris type'].astype(object)
print(df.head())


# In[500]:

X = df[[0,1,2,3]].values
y = df['iris type'].values


# In[501]:

#run pca
pca = decomposition.PCA(n_components=4)
pca_data = pca.fit_transform(X)
df_pca = pd.DataFrame(pca_data, columns = iris.feature_names)
df_pca['iris type'] = df['iris type']

#run lda
lda = LDA(solver = 'eigen', n_components = 4)
lda_data = lda.fit_transform(X,y)
df_lda = pd.DataFrame(lda_data, columns = iris.feature_names)
df_lda['iris type'] = df['iris type']


# In[458]:

#plot original points
prplt = sea.pairplot(df, hue ='iris type')
plt.title('Original')
plt.show()

#plot pca transformation
prplt = sea.pairplot(df_pca, hue ='iris type')
plt.title('PCA')
plt.show()

#plot LDA transformation
prplt = sea.pairplot(df_lda, hue ='iris type')
plt.title('LDA')
plt.show()


# In[502]:

#graphing distance function for sepal lenth to sepal with as in K-NN
def graphdist(index,xvar,yvar,df_eq):
    dist = (((df_eq.ix[index,'sepal length (cm)']) - xvar)**2 + ((df_eq.ix[index,'sepal width (cm)']) - yvar)**2)**(1/2)
    dist = round(dist, 5)
    return(dist)
#a distance column
df['distance'] = 0
df_pca['distance'] = 0
df_lda['distance'] = 0



# In[503]:

#picking the random point and setting it
rndm_pt = randint(0,len(df))
x1_df = df.ix[rndm_pt,'sepal length (cm)']
y1_df = df.ix[rndm_pt,'sepal width (cm)']
x1_pca = df_pca.ix[rndm_pt,'sepal length (cm)']
y1_pca = df_pca.ix[rndm_pt,'sepal width (cm)']
x1_lda = df_lda.ix[rndm_pt,'sepal length (cm)']
y1_lda = df_lda.ix[rndm_pt,'sepal width (cm)']

print(df.ix[rndm_pt])
print(df_pca.ix[rndm_pt])
print(df_lda.ix[rndm_pt])


# In[506]:

#applying distance function to all dfs
#def df_distance(xvar,yvar,df_eq,d_type):
#    for index, row in df_eq.iterrows():
#        df_eq.ix[index, 'distance'] = graphdist(index,xvar,yvar,df_eq)
#    df_eq = df_eq.sort_values(['distance'], ascending = 1)
#    return(df_eq)

for index, row in df.iterrows():
    df.ix[index, 'distance'] = graphdist(index,x1_df,y1_df,df)
df = df.sort_values(['distance'], ascending = 1)

for index, row in df_pca.iterrows():
    df_pca.ix[index, 'distance'] = graphdist(index,x1_pca,y1_pca,df_pca)
df_pca = df_pca.sort_values(['distance'], ascending = 1)

for index, row in df_lda.iterrows():
    df_lda.ix[index, 'distance'] = graphdist(index,x1_lda,y1_lda,df_lda)
df_lda = df_lda.sort_values(['distance'], ascending = 1)


# In[510]:

def knn(k,df_eq):
    new_df = df_eq.head(n = k)
    flowertype = new_df['iris type'].describe()
    return(flowertype)

def knntest(i,df_eq):
    correct_count = 0
    false_count = 0
    for i in range(1,i):
        flwr = knn(i,df_eq)
        rndmptflower = df.ix[rndm_pt,'iris type']
        #print(type(flwr))
        #print(flwr['top'])
        #print(type(rndmptflower))
        #print(rndmptflower)
        if flwr['top'] == rndmptflower:
            correct_count += 1
        else:
            false_count += 1
    print(correct_count, false_count)
    print(correct_count/(correct_count+false_count))
            


# In[511]:

knntest(10,df)


# In[512]:

print(knn(10,df))
knntest(20,df)


# In[513]:

print(knn(10,df_lda))
knntest(20, df_lda)


# In[514]:

print(knn(10,df_pca))
knntest(20, df_pca)


# In[ ]:



