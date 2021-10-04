# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 01:26:46 2021

@author: Benedict
"""

 
import pandas as pd # data structure and the operations
import numpy as np # arrays and the operations
import matplotlib.pyplot as plt

import sklearn.linear_model as skl_lm # for regression

from sklearn.metrics import mean_squared_error, r2_score # for metrics

import matplotlib.pyplot as plt # plotting library

import seaborn as sns # another plotting library

#from matplotlib.dates import  DateFormatter

file_path = 'C:/Users/Benedict/OneDrive/Spring 2021/IOT CSC 492/Python work/ESP32_data.csv'

#read in the DateTime Col as dateformat
data = pd.read_csv(file_path, usecols=[0,3,4], parse_dates=['DateTime'], dayfirst=True,)

data.head() # see top elements

data.info() # get dataset  details


#over here basically make a column called times, representing times for the day
data['times'] = data.DateTime.dt.time.astype(str)
data['times'] = pd.to_datetime(data['times'])
data.head()
#TEMPERATURE V TIME
fig = plt.figure(figsize=(15,10))


import matplotlib.dates as mdates

#note .values gives an array of values
#Day 1
x = data.times[:61]
y = data.Temperature[:61]
plt.plot(x, y, label="DAY 1")

#Day2
x = data.times[61:61+61]
y = data.Temperature[61:61+61]
plt.plot(x, y, label="DAY 2")

#Day 3
x = data.times[122:122+61]
y = data.Temperature[122:122+61]
plt.plot(x, y, label="DAY 3")

#Day 4
x = data.times[183:183+61]
y = data.Temperature[183:183+61]
plt.plot(x, y, label="DAY 4")

#Day 5
x = data.times[244:244+61]
y = data.Temperature[244:244+61]
plt.plot(x, y, label="DAY 5")

plt.legend()
plt.xlabel("Time")
plt.ylabel("Temperature")

#Change the date to display only the time
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(myFmt)



#Humidity V Time

fig = plt.figure(figsize=(15,10))

#Day 1
x1 = data.times[:61]
y1 = data.Humidity[:61]
plt.plot(x1, y1, label="DAY 1")

#Day2
x1 = data.times[61:61+61]
y1 = data.Humidity[61:61+61]
plt.plot(x1, y1, label="DAY 2")

#Day 3
x1 = data.times[122:122+61]
y1 = data.Humidity[122:122+61]
plt.plot(x1, y1, label="DAY 3")

#Day 3
x1 = data.times[183:183+61]
y1 = data.Humidity[183:183+61]
plt.plot(x1, y1, label="DAY 4")

#Day 3
x1 = data.times[244:244+61]
y1 = data.Humidity[244:244+61]
plt.plot(x1, y1, label="DAY 5")

plt.legend()
plt.xlabel("Time")
plt.ylabel("Humidity")

#change the date to display only the time
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(myFmt)

#%% box plot for Temperature on different days 


temp1 = data.Temperature[:61]
temp2 = data.Temperature[61:61+61]
temp3 = data.Temperature[122:122+61]
temp4 = data.Temperature[183:183+61]
temp5 = data.Temperature[244:244+61]
df =pd.DataFrame({'Day 1':temp1, 'Day 2': temp2, 'Day 3':temp3, 'Day 4': temp4,'Day 5':temp5})

boxplot = df.boxplot()
plt.title("Temperature per Day")

#%% box plot for Humidity on different days 


hum1 = data.Humidity[:61]
hum2 = data.Humidity[61:61+61]
hum3 = data.Humidity[122:122+61]
hum4 = data.Humidity[183:183+61]
hum5 = data.Humidity[244:244+61]
df =pd.DataFrame({'Day 1':hum1, 'Day 2': hum2, 'Day 3':hum3, 'Day 4': hum4,'Day 5':hum5})

boxplot = df.boxplot()
plt.title("Humidity per Day")

#%% scatter plot matrix
# Grid plot all lines

temp1 = data.Temperature[:61]
hum1 = data.Humidity[:61]
temp2 = data.Temperature[61:61+61]
hum2 = data.Humidity[61:61+61]
temp3 = data.Temperature[122:122+61]
hum3 = data.Humidity[122:122+61]


df = pd.DataFrame({'Temp 1': temp1, 'Hum 1': hum1,'Temp 2': temp2, 'Hum 2': hum2, 'Temp 3': temp3, 'Hum 3': hum3})
#pd.plotting.scatter_matrix(df, diagonal='kde')

sns.pairplot(df, diag_kind = "kde")



#%%

#Component #3 Data Analysis
#create an array that sotres the times
times = data['DateTime'].dt.time
print(times)
minutes = []
#hour * 60 + minute + second/60
for i in times:
    hour = i.hour
    minute = i.minute
    second = i.second
    minutes.append((hour * 60) + minute + (second/60))

data['minutes'] = minutes

#data.head()

#%% UNIVARIATE REGRESSION
# Regression coefficients (Ordinary Least Squares) with Scikit Learn library

#------------------MODEL 1---------------------------------------
# Create a linear regression model between temperature and minutes.
# Call this model M1.
M1 = skl_lm.LinearRegression()

X = data.minutes.values.reshape(-1,1) # need a 2d array hence we reshape it
y = data.Temperature

M1.fit(X,y)   # y = b0 + b1 * x1

print('intercept: ',M1.intercept_) 
print('coefficient: ',M1.coef_)
#equation in terms of determined coefficients
#Temperature = 16.975 + 0.006 * minutes + epsilon

#% Predict temperature with the created model
temp_pred1 = M1.predict(X) 

r2_score(y, temp_pred1) # Coefficient of determination

#Mean Squared Error of M1:  3.2356426675499463
mean_squared_error(y,temp_pred1)
print('Mean Squared Error of M1: ',mean_squared_error(y,temp_pred1)) # 52.46563680960855


resid = y - temp_pred1
# sum of squares
sse_M1 = sum(resid**2) 

res_df1 = pd.DataFrame({'actual':y, 'predicted':temp_pred1})

res_df1.index = range(1,len(y)+1)

res_df1.plot()


#%%
#----------------------MODEL 2--------------------------------------
# Create a linear regression model between Humidity and minutes. Call this model M2.
M2 = skl_lm.LinearRegression()

X = data.minutes.values.reshape(-1,1) # need a 2d array hence we reshape it
y = data.Humidity

M2.fit(X,y)   # y = b0 + b1 * x1

print('intercept: ',M2.intercept_) 
print('coefficient: ',M2.coef_)
#equation in terms of determined coefficients
#Humidity = 14.571 + 0.003 * minutes + epsilon

#% Predict temperature with the created model
hum_pred = M2.predict(X) 
r2_score(y, hum_pred) 

#Mean Squared Error of M2
mean_squared_error(y,hum_pred)
#Mean Squared Error tells how accurate is the calculated coefficient with respect to the actual value
#Mean Squared Error of M2:  48.279
print('Mean Squared Error of M2: ',mean_squared_error(y,hum_pred)) 



resid = y - hum_pred
# sum of squares
sse_M2 = sum(resid**2) 

res_df2 = pd.DataFrame({'actual':y, 'predicted':hum_pred})

res_df2.index = range(1,len(y)+1)

res_df2.plot()

#%%
#---------M3-----------------------
M3 = skl_lm.LinearRegression()

X = data[['Humidity', 'minutes']] # converts dataframe to matrix

y = data.Temperature

M3.fit(X,y) # y = b0 + b1 * x1 (Humidity) + b2 * x2 (minutes) + epsion

print('coefficient: ',M3.coef_)
print('intercept: ',M3.intercept_)
#equation in terms of determined coefficients
#Temperature = 24.638 + -0.525 * Humidity +0.0085*minutes + epsilon


multi_pred = M3.predict(X) 

#r2_score(y, hum_pred) 
#Mean Squared Error tells how accurate is the calculated coefficient with respect to the actual value
#Mean Squared Error of M3:  1.18276
print('Mean Squared Error of M3: ',mean_squared_error(y,multi_pred) ) 

resid = y - multi_pred
# sum of squares
sse_M3 = sum(resid**2) 

res_df3 = pd.DataFrame({'actual':y, 'predicted':multi_pred})

res_df3.index = range(1,len(y)+1)

res_df3.plot()



#%%
#  COMPARING DIFFERENT MODELS

#AIC= 2k - 2ln(sse)
AIC_M1 = 2*1 - 2*np.log(sse_M1)

AIC_M2 = 2*1 - 2*np.log(sse_M2)

AIC_M3 = 2*2 - 2*np.log(sse_M3)

#print('AIC of model 1, model 2 and model 3 are', AIC_M1, AIC_M2, AIC_M3)

print('AIC of model 1, model 2, and model 3 are', np.round(AIC_M1,2), np.round(AIC_M2,2), np.round(AIC_M3,2))
#AIC of model 1, model 2, and model 3 are -11.79 -7.78 -5.78


