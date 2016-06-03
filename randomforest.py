
# coding: utf-8

# In[2]:




# In[132]:

import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sklearn.metrics as mets
import seaborn as sea




df = pd.read_csv('/Users/Thomas/Projects/features.txt', header = None, delim_whitespace = True)
df = df.drop(df.columns[[0]], axis=1) 
df.columns = ['fieldname']
df['corrected'] = ""
#print(df)



# In[133]:

#print(len(df))
df.drop_duplicates()
#print(len(df))


# In[134]:

count = 0
for index, row in df.iterrows():
    name = row['fieldname']
    nm = re.sub((r'\W|Body|Mag'),' ', name) 
    df.set_value(index, 'corrected', nm)

df['corrected'] = [x.strip().replace(' ', '_') for x in df['corrected']]
df['corrected'] = [x.strip().replace('mean', 'Mean') for x in df['corrected']]
df['corrected'] = [x.strip().replace('std', 'STD') for x in df['corrected']]
#print(df['corrected'])


# In[92]:

df.drop_duplicates()
#print(len(df))
correctednames = df['corrected'].tolist()


# In[135]:

xtrain = pd.read_csv('/Users/Thomas/projects/Unit 4/UCI HAR Dataset/train/X_train.txt', header=None, delim_whitespace=True, index_col=False)
xtrain.columns = df['corrected']
#print(xtrain)


# In[109]:

ytrain = pd.read_csv('/Users/Thomas/projects/Unit 4/UCI HAR Dataset/train/y_train.txt', names=["activity"], header=None, index_col=False)
ytrain["activity"].astype('category')


# In[106]:

subjects = pd.read_csv('/Users/Thomas/projects/Unit 4/UCI HAR Dataset/train/subject_train.txt', names=["subjects"], header=None, index_col=False)


# In[136]:

data = pd.merge(xtrain, ytrain, right_index = True, left_index = True)
data = pd.merge(data, subjects, right_index = True, left_index = True)
#print(data)


# In[137]:

training_set = data[data['subjects'] > 27]
test_set = data[data['subjects'] <= 6]
cv_set = data[(data["subjects"] >= 21) & (data['subjects'] < 27)] 



# In[138]:

xtraining_set = training_set.drop(['activity','subjects'], axis=1)
ytraining_set = training_set['activity']

xtest_set = test_set.drop(['activity','subjects'], axis=1)
ytest_set = test_set['activity']

xcv_set = cv_set.drop(['activity','subjects'], axis=1)
ycv_set = cv_set['activity']

forest = RandomForestClassifier(n_estimators=500, oob_score=True)
forest.fit(xtraining_set, ytraining_set)
print(forest.oob_score_)





# In[14]:


feat = forest.feature_importances_
featindex = np.argsort(feat)[::-1]
for i in range(10):
    print(1+i, featindex[i], round(feat[featindex[i]],5))

    


# In[15]:

ypredicted_set = forest.predict(xtest_set)

print('Test score')
print(forest.score(xtest_set, ytest_set))

print('Eval score')
print(forest.score(xcv_set, ycv_set))


# In[16]:

confumatrix = mets.confusion_matrix(ytest_set, ypredicted_set)
sea.heatmap(data = confumatrix)
sea.plt.show()



# In[18]:

print('accuracy')
print(round(mets.precision_score(ytest_set, ypredicted_set),5))
print('recall')
print(round(mets.recall_score(ytest_set, ypredicted_set),5))
print('f1')
print(round(mets.f1_score(ytest_set, ypredicted_set),5))


# In[ ]:



