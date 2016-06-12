
# coding: utf-8

# In[12]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.naive_bayes import GaussianNB 

df = pd.read_csv('/Users/Thomas/Projects/Unit 4/ideal_weight.csv')


# In[13]:

colnames = df.columns.values


# In[14]:

colnames = [x.strip().replace("'", "") for x in colnames]


# In[15]:

df.columns = colnames


# In[16]:

df['sex'] = [x.strip().replace("'","") for x in df['sex']]
print(df)


# In[17]:

plt.hist(df['ideal'], label = 'ideal')
plt.hist(df['actual'], label = 'actual')
plt.legend()
plt.show()


# In[18]:

plt.hist(df['diff'], label = 'difference')
plt.legend()
plt.show()


# In[19]:

df['sex'].astype('category')
sexcounts = df['sex'].value_counts()
print(sexcounts)


# In[22]:

xvars = df[['actual', 'ideal', 'diff']]
yvar = df['sex']

gnb = GaussianNB()
gnb.fit(xvars,yvar)


# In[31]:

incorrect = 0
correct = 0


for index, row in df.iterrows():
    inputs = row[['actual', 'ideal', 'diff']].values
    expectedvalue = row['sex']
    #print(inputs)
    predictedsex = gnb.predict(inputs)
    if predictedsex == expectedvalue:
        correct += 1
    else:
        incorrect +=1
print(correct)
print(incorrect)
    


# In[33]:

print('Actual weight at 145, with ideal at 160')
print(gnb.predict([145, 160, -15]))
print('Actual weight at 160, with ideal at 145')
print(gnb.predict([160, 145, 15]))


# In[ ]:



