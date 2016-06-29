
# coding: utf-8

# In[4]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea

colnames = pd.read_csv('/Users/Thomas/Projects/Unit 4/iriscolumnnames.csv')
df = pd.read_csv('/Users/Thomas/Projects/Unit 4/irisdata.csv', names = colnames)

print(df.head())
print(df['iris class'].unique())


# In[6]:

sea.pairplot(df, hue = 'iris class')
plt.show()


# In[ ]:



