# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:00:41 2021

@author: Haroon
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random
 

#%%
#Histogram
#np.random.seed(99)
ages = np.random.normal(28, 5, 30)
plt.hist(ages,bins='auto')
plt.xlabel('Age')
plt.ylabel('Frequency')

#%% Playing with histogram binwidth

fig, ax = plt.subplots(nrows=2, ncols=2)

bin = 4
for  x in range(0,2):
    for y in range(0,2):
        ax[x,y].hist(ages,bins=bin)
        bin = bin + 1

#%%
# Density plot

#np.random.seed(99) 
x_values = np.random.normal(28, 5, 30)
df = pd.DataFrame(x_values, columns = ['age'] ) #Converting array to pandas DataFrame
df.plot(kind = 'density')
        
#%%        
# Bar plot
import collections
 
sample_colors = ['red','green','blue','black']

ball_colors = np.random.choice(sample_colors, 100)

ball_color_count =  collections.Counter(ball_colors)

colors = list (ball_color_count.keys())

values = list (ball_color_count.values())

plt.bar(colors,values,color='blue', width = 0.5)

#%% Box plot for one variable
np.random.seed(99) 
ages = np.random.normal(28, 5, 30)
plt.boxplot(ages)
plt.ylabel('age')
plt.tick_params(axis='x', which='both', bottom = False, labelbottom = False)


#%% box plot for several variables

np.random.seed(9999)

ages = np.random.normal(28, 5, 30)
weights = np.random.normal(60, 5, 30)
df =pd.DataFrame({'ages':ages, 'weights': weights})
boxplot = df.boxplot()

 

#%% QQ plot
# https://towardsdatascience.com/what-in-the-world-are-qq-plots-20d0e41dece1

# Generate some normally distributed random numbers
random_normals = [np.random.normal() for i in range(1000)]

# Create QQ plot
sm.qqplot(np.array(random_normals), line='45')
plt.show()

# Generate some uniformly distributed random variables
random_uniform = [random.random() for i in range(1000)]
# Create QQ plot
sm.qqplot(np.array(random_uniform), line='45')
plt.show()


#%% Scatter plot

height = np.random.normal(5,1,100)
weight = np.random.uniform(120,150,100)

df = pd.DataFrame({'height': height, 'weight': weight})
df.plot.scatter(x='height',y='weight')


#%% scatter plot matrix

A = np.random.normal(0,1,50)
B = np.random.normal(10,2,50)
C = np.random.uniform(30,15,50)
D = np.random.uniform(4,1,50)

df = pd.DataFrame({'A': A, 'B': B,'C': C, 'D': D})
pd.plotting.scatter_matrix(df, diagonal='kde')

 