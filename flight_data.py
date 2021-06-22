#!/usr/bin/env python
# coding: utf-8

# # Flight & Weather Data With Dask

# In[1]:


from dask import delayed
import dask.dataframe as dd

import pandas as pd
import numpy as np


# ### Explore one flight and one weather dataset

# In[48]:


df = pd.read_csv('datasets/flightdelays/flightdelays-2016-1.csv')


# In[49]:


df.head()


# In[50]:


df.describe()


# Can see there is a lot of data, as this is one of 10 csv files for the project

# In[51]:


df.info()


# In[52]:


df.DEP_DELAY.hist(bins=50)


# In[53]:


df.UNIQUE_CARRIER.nunique()


# In[54]:


df.ORIGIN.nunique()


# store same dataframe as dask dataframe for later use

# In[56]:


df_dask = dd.read_csv('datasets/flightdelays/flightdelays-2016-1.csv')


# In[9]:


df_weather = pd.read_csv('datasets/weatherdata/ATL.csv')


# In[10]:


df_weather.head()


# In[11]:


df_weather.describe()


# In[12]:


df_weather.info()


# In[13]:


df_weather['Mean TemperatureF'].plot()


# Store as dask dataframe

# In[57]:


df_weather_dask = dd.read_csv('datasets/weatherdata/ATL.csv')


# ### Creating delayed functions for reading & cleaning of data

# Replace zeros in 'WEATHER DELAY' column with nan to make counting delays easier

# In[14]:


@delayed
def read_flights(filename):
    df = pd.read_csv(filename, parse_dates=['FL_DATE'])
    df['WEATHER_DELAY'] = df['WEATHER_DELAY'].replace(0, np.nan)
    return df


# In[15]:


flight_filenames = ['flightdelays-2016-1.csv', 
                    'flightdelays-2016-2.csv', 
                    'flightdelays-2016-3.csv', 
                    'flightdelays-2016-4.csv', 
                    'flightdelays-2016-5.csv']


# In[16]:


dataframes = []
# Read all data
for filename in flight_filenames:
    dataframes.append(read_flights('datasets/flightdelays/' + filename))

flight_delays = dd.from_delayed(dataframes)

print(flight_delays['WEATHER_DELAY'].mean().compute())


# Second delayed function to read weather data

# In[17]:


@delayed
def read_weather(filename):
    df = pd.read_csv(filename, parse_dates=['Date'])
    df['PrecipitationIn'] = pd.to_numeric(df['PrecipitationIn'], errors='coerce')
    # Create 'Airport' column
    df['Airport'] = filename.split('.')[0]
    return df


# In[18]:


weather_filenames = ['ATL.csv', 'DEN.csv', 'DFW.csv', 'MCO.csv', 'ORD.csv']


# In[19]:


weather_dfs = []


# In[27]:


for filename in weather_filenames:
    weather_dfs.append(read_weather('datasets/weatherdata/' + 'ATL.csv'))

weather = dd.from_delayed(weather_dfs)

print(weather.nlargest(1, 'Max TemperatureF').compute())


# In[28]:


is_snowy = weather['Events'].str.contains('Snow').fillna(False)


# In[29]:


def percent_delayed(df):
    return (df['WEATHER_DELAY'].count() / len(df)) * 100


# In[33]:


import time


# In[42]:


type(df)


# Merge datafranes into weather delay dataframe

# In[58]:


weather_delay = df_dask.merge(df_weather_dask, left_on="FL_DATE", right_on="Date")


# In[59]:


weather_delay.head()


# Compare reading from disk rather than persistig to memory

# In[60]:


t_start = time.time()
print(percent_delayed(weather_delay).compute())
t_end = time.time()
print((t_end-t_start)*1000)


# In[61]:


persisted_weather_delays = weather_delay.persist()


# In[62]:


t_start = time.time()
print(percent_delayed(persisted_weather_delays).compute())
t_end = time.time()
print((t_end-t_start)*1000)


# so persisting into memory results in taking only 3.7% of original time

# ### Finding source of weather delays

# In[64]:


# Group persisted_weather_delays by 'Events': by_event
by_event = persisted_weather_delays.groupby('Events')


# In[65]:


# Count 'by_event['WEATHER_DELAY'] column & divide by total number of delayed flights
pct_delayed = by_event['WEATHER_DELAY'].count() / persisted_weather_delays['WEATHER_DELAY'].count() * 100

# Compute & print five largest values of pct_delayed
print(pct_delayed.nlargest(5).compute())


# In[66]:


# Calculate mean of by_event['WEATHER_DELAY'] column & return the 5 largest entries: avg_delay_time
avg_delay_time = by_event['WEATHER_DELAY'].mean().nlargest(5)

# Compute & print avg_delay_time
print(avg_delay_time.compute())


# In[ ]:




