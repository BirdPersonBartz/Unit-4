
# coding: utf-8

# In[2]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.cluster.vq import vq

#import data
df = pd.read_csv('/Users/Thomas/Projects/Unit 4/un.csv')
print(df.head())


# In[3]:

print('Total rows %s' % len(df))
counts = df.count()
print(counts.sort_values(ascending = 0))


# In[25]:

data = df[['lifeMale', 'lifeFemale', 'infantMortality', 'GDPperCapita']]

#dropping NAN since it doesn't work with witten
data = data.dropna()

#setup lists
lifemalelist = []
lifeFemalelist = []
infantMortalitylist = []
GDPperCapitalist = []

#Create points
for index, row in data.iterrows():
    lifemalelist.append([row['lifeMale'], row['GDPperCapita']])
    lifeFemalelist.append([row['lifeFemale'], row['GDPperCapita']])
    infantMortalitylist.append([row['infantMortality'], row['GDPperCapita']])

#whiten/normalize the data
lifemale_plain = lifemalelist
lifemalelist = whiten(lifemalelist)
lifefemale_plain = lifeFemalelist
lifeFemalelist = whiten(lifeFemalelist)
infantmortality_plain = infantMortalitylist
infantMortalitylist = whiten(infantMortalitylist)

#more lists
femalecentslist = []
infantcentslist = []
malecentslist = []
centslistorder = []

#calc the kmean for differnt # of centroids, also append the import kmeans array into a list for each # of centroids
for i in range(10,0, -1):
    kmeanlifeM = kmeans(lifemalelist, i)
    kmeanlifeF = kmeans(lifeFemalelist, i)
    kmeanInfant = kmeans(infantMortalitylist, i)
    malecentslist.append(kmeanlifeM[0])
    femalecentslist.append(kmeanlifeF[0])
    infantcentslist.append(kmeanInfant[0])
    centslistorder.append(i)



# In[5]:


maleKs1to10 = []
femaleKs1to10 = []
infantKs1to10 = []


#get the closest point and its distance then calc the mean squared
for i in range(10):
    mvq = vq(lifemalelist,malecentslist[i])
    fvq = vq(lifeFemalelist,femalecentslist[i])
    ivq = vq(infantMortalitylist,infantcentslist[i])

    maletotal = 0
    femaletotal = 0
    infanttotal = 0
    for i in mvq[1]:
        addM = (i - mvq[1].mean())**2
        maletotal = maletotal + addM
    for i in fvq[1]:
        addF = (i - fvq[1].mean())**2
        femaletotal = femaletotal + addF
    for i in ivq[1]:
        addI = (i - ivq[1].mean())**2
        infanttotal = infanttotal + addI
    #add one for each # of centroids
    maleKs1to10.append(maletotal)
    femaleKs1to10.append(femaletotal)
    infantKs1to10.append(infanttotal)
    
print(maleKs1to10)
print(femaleKs1to10)
print(infantKs1to10)


# In[6]:

#plot it out

rng = range(10,0,-1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(rng, maleKs1to10)
plt.grid(True)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(rng, femaleKs1to10)
plt.grid(True)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(rng, infantKs1to10)
plt.grid(True)
plt.show()



# In[32]:

#clunky way to get this order, should have used numpy arrays


print(centslistorder)
mvq = vq(lifemalelist,malecentslist[7])
fvq = vq(lifeFemalelist,femalecentslist[7])
ivq = vq(infantMortalitylist,infantcentslist[7])


dfmale = pd.DataFrame(lifemale_plain, columns = ['Average Male life Expectancy in years','GDP per cap in USD'])
dfmale['centroid index'] = mvq[0]
dffemale = pd.DataFrame(lifefemale_plain, columns = ['Average Female life Expectancy in years','GDP per cap in USD'])
dffemale['centroid index'] = fvq[0]
#correcting my infant naming
dfinfant = pd.DataFrame(infantmortality_plain, columns = ['Infant Mortality per 100k','GDP per cap in USD'])
dfinfant['centroid index'] = ivq[0]


# In[33]:

sea.pairplot(dfmale, vars=["Average Male life Expectancy in years", "GDP per cap in USD"], hue = 'centroid index')
plt.show()
sea.pairplot(dffemale, vars=["Average Female life Expectancy in years", "GDP per cap in USD"], hue = 'centroid index')
plt.show()
sea.pairplot(dfinfant, vars=["Infant Mortality per 100k", "GDP per cap in USD"], hue = 'centroid index')
plt.show()


# In[ ]:





