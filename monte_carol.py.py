
# coding: utf-8

# In[18]:

'''This script demonstrates simulations of coin flipping'''
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sea

# let's create a fair coin object that can be flipped:

class Coin(object):
    '''this is a simple fair coin, can be pseudorandomly flipped'''
    sides = ('heads', 'tails')
    last_result = None

    def flip(self):
        '''call coin.flip() to flip the coin and record it as the last result'''
        self.last_result = result = random.choice(self.sides)
        return result

# let's create some auxilliary functions to manipulate the coins:

def create_coins(number):
    '''create a list of a number of coin objects'''
    return [Coin() for _ in range(number)]

def flip_coins(coins):
    '''side effect function, modifies object in place, returns None'''
    for coin in coins:
        coin.flip()

def count_heads(flipped_coins):
    return sum(coin.last_result == 'heads' for coin in flipped_coins)

def count_tails(flipped_coins):
    return sum(coin.last_result == 'tails' for coin in flipped_coins)


def main():
    coins = create_coins(1000)
    for i in range(100):
        flip_coins(coins)
        var = count_heads(coins)
        print(var)
        coincounts.append(var)

coincounts = []
        
if __name__ == '__main__':
    main()
    


# In[50]:

sea.distplot(coincounts, bins=50)
plt.show()


# In[51]:

from numpy.random import normal
s = normal(size=(1024*32,))
sea.distplot(s,bins=50)
plt.show()

#created a normal dist


# In[46]:

high_low_refined = []
for i in range(100):
    high_low_data = []
    for _ in range(10):
        high_low_data.append(np.random.normal(loc=0))
    high_low_refined.append(max(high_low_data))
    high_low_refined.append(min(high_low_data))

#running 100 iterations of the max/min for a set of 10 normal vars


# In[48]:

#print(high_low_refined)


# In[52]:

sea.distplot(high_low_refined,bins=50)
plt.show()

#displaying the normal var distribution. Looks like H/T for a coin


# In[ ]:



