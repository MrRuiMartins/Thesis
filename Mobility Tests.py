
# coding: utf-8

# In[ ]:







# In[25]:

get_ipython().magic('matplotlib inline')
from sensible_raw.loaders import loader
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas
months=["september_2013",
       "october_2013",
       "november_2013",
       "december_2013",
       "january_2014",
       "february_2014",
       "march_2014",
       "april_2014",
       "may_2014",
       "june_2014",
       "july_2014",
       "august_2014"]


dataframe = pandas.DataFrame({})
for m in months:
    tmp_dict = {}
    columns, data = loader.load_data("location", m);
    for column, array in zip(columns, data):
        tmp_dict[column] = array
    tmp_dataframe = pandas.DataFrame(tmp_dict)
    dataframe = dataframe.append(tmp_dataframe)


# In[26]:

def IsPointWhithinDTUPandas(df):
    dtullcrnrlon = 12.49
    dtullcrnrlat = 55.7741
    dtuurcrnrlon = 12.542
    dtuurcrnrlat = 55.792
    return df['lon_red'] > dtullcrnrlon and df['lon_red'] < dtuurcrnrlon and df['lat_red'] > dtullcrnrlat and df['lat_red'] < dtuurcrnrlat


# In[3]:

def RunUniquenessCheck(sampleSize, my_dataframe, includeDTU):
    
    timeResolution = ['utc','utcHalfHourly', 'utcHourly', 'utcBiHourly', 'utc3Hours']
    spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red'], ['lat', 'lon']];

    # choose n random points from one user:
    userdf = my_dataframe.sample(1)
    
    if includeDTU:
        sample = my_dataframe.loc[my_dataframe['user'] == userdf['user'].values[0]].sample(sampleSize)
    else:
        sample = my_dataframe.loc[(my_dataframe['user'] == userdf['user'].values[0]) & (dataframe['IsInDtu'] == False)].sample(sampleSize)

    usersWithTrace1 = []
    usersWithTrace2 = []
    usersWithTrace3 = []
    usersWithTrace4 = []
    usersWithTrace5 = []
    usersWithTrace6 = []
    usersWithTrace7 = []
    usersWithTrace8 = []
    usersWithTrace9 = []
    usersWithTrace10 = []
    usersWithTrace11 = []
    usersWithTrace12 = []
    usersWithTrace13 = []
    usersWithTrace14 = []
    usersWithTrace15 = []
    usersWithTrace16 = []
    usersWithTrace17 = []
    usersWithTrace18 = []
    usersWithTrace19 = []
    usersWithTrace20 = []
    
    for u in set(my_dataframe['user']):
        udf = my_dataframe.loc[my_dataframe['user'] == u]
        containsTrace1 = True
        containsTrace2 = True
        containsTrace3 = True
        containsTrace4 = True
        containsTrace5 = True
        containsTrace6 = True
        containsTrace7 = True
        containsTrace8 = True
        containsTrace9 = True
        containsTrace10 = True
        containsTrace11 = True
        containsTrace12 = True
        containsTrace13 = True
        containsTrace14 = True
        containsTrace15 = True
        containsTrace16 = True
        containsTrace17 = True
        containsTrace18 = True
        containsTrace19 = True
        containsTrace20 = True
        #print("Performing trace check for user: " + str(u))
        for idx, s in sample.iterrows():
            containsTrace1 = ((udf[spatialResolution[0][0]] == s[spatialResolution[0][0]]) & (udf[spatialResolution[0][1]] == s[spatialResolution[0][1]]) & (udf[timeResolution[0]] == s[timeResolution[0]])).any()
            containsTrace2 = ((udf[spatialResolution[1][0]] == s[spatialResolution[1][0]]) & (udf[spatialResolution[1][1]] == s[spatialResolution[1][1]]) & (udf[timeResolution[0]] == s[timeResolution[0]])).any()
            containsTrace3 = ((udf[spatialResolution[2][0]] == s[spatialResolution[2][0]]) & (udf[spatialResolution[2][1]] == s[spatialResolution[2][1]]) & (udf[timeResolution[0]] == s[timeResolution[0]])).any()
            containsTrace4 = ((udf[spatialResolution[3][0]] == s[spatialResolution[3][0]]) & (udf[spatialResolution[3][1]] == s[spatialResolution[3][1]]) & (udf[timeResolution[0]] == s[timeResolution[0]])).any()
            containsTrace5 = ((udf[spatialResolution[0][0]] == s[spatialResolution[0][0]]) & (udf[spatialResolution[0][1]] == s[spatialResolution[0][1]]) & (udf[timeResolution[1]] == s[timeResolution[1]])).any()
            containsTrace6 = ((udf[spatialResolution[1][0]] == s[spatialResolution[1][0]]) & (udf[spatialResolution[1][1]] == s[spatialResolution[1][1]]) & (udf[timeResolution[1]] == s[timeResolution[1]])).any()
            containsTrace7 = ((udf[spatialResolution[2][0]] == s[spatialResolution[2][0]]) & (udf[spatialResolution[2][1]] == s[spatialResolution[2][1]]) & (udf[timeResolution[1]] == s[timeResolution[1]])).any()
            containsTrace8 = ((udf[spatialResolution[3][0]] == s[spatialResolution[3][0]]) & (udf[spatialResolution[3][1]] == s[spatialResolution[3][1]]) & (udf[timeResolution[1]] == s[timeResolution[1]])).any()
            containsTrace9 = ((udf[spatialResolution[0][0]] == s[spatialResolution[0][0]]) & (udf[spatialResolution[0][1]] == s[spatialResolution[0][1]]) & (udf[timeResolution[2]] == s[timeResolution[2]])).any()
            containsTrace10 = ((udf[spatialResolution[1][0]] == s[spatialResolution[1][0]]) & (udf[spatialResolution[1][1]] == s[spatialResolution[1][1]]) & (udf[timeResolution[2]] == s[timeResolution[2]])).any()
            containsTrace11 = ((udf[spatialResolution[2][0]] == s[spatialResolution[2][0]]) & (udf[spatialResolution[2][1]] == s[spatialResolution[2][1]]) & (udf[timeResolution[2]] == s[timeResolution[2]])).any()
            containsTrace12 = ((udf[spatialResolution[3][0]] == s[spatialResolution[3][0]]) & (udf[spatialResolution[3][1]] == s[spatialResolution[3][1]]) & (udf[timeResolution[2]] == s[timeResolution[2]])).any()
            containsTrace13 = ((udf[spatialResolution[0][0]] == s[spatialResolution[0][0]]) & (udf[spatialResolution[0][1]] == s[spatialResolution[0][1]]) & (udf[timeResolution[3]] == s[timeResolution[3]])).any()
            containsTrace14 = ((udf[spatialResolution[1][0]] == s[spatialResolution[1][0]]) & (udf[spatialResolution[1][1]] == s[spatialResolution[1][1]]) & (udf[timeResolution[3]] == s[timeResolution[3]])).any()
            containsTrace15 = ((udf[spatialResolution[2][0]] == s[spatialResolution[2][0]]) & (udf[spatialResolution[2][1]] == s[spatialResolution[2][1]]) & (udf[timeResolution[3]] == s[timeResolution[3]])).any()
            containsTrace16 = ((udf[spatialResolution[3][0]] == s[spatialResolution[3][0]]) & (udf[spatialResolution[3][1]] == s[spatialResolution[3][1]]) & (udf[timeResolution[3]] == s[timeResolution[3]])).any()
            containsTrace17 = ((udf[spatialResolution[0][0]] == s[spatialResolution[0][0]]) & (udf[spatialResolution[0][1]] == s[spatialResolution[0][1]]) & (udf[timeResolution[4]] == s[timeResolution[4]])).any()
            containsTrace18 = ((udf[spatialResolution[1][0]] == s[spatialResolution[1][0]]) & (udf[spatialResolution[1][1]] == s[spatialResolution[1][1]]) & (udf[timeResolution[4]] == s[timeResolution[4]])).any()
            containsTrace19 = ((udf[spatialResolution[2][0]] == s[spatialResolution[2][0]]) & (udf[spatialResolution[2][1]] == s[spatialResolution[2][1]]) & (udf[timeResolution[4]] == s[timeResolution[4]])).any()
            containsTrace20 = ((udf[spatialResolution[3][0]] == s[spatialResolution[3][0]]) & (udf[spatialResolution[3][1]] == s[spatialResolution[3][1]]) & (udf[timeResolution[4]] == s[timeResolution[4]])).any()

        if containsTrace1:
            usersWithTrace1.append(u)
        if containsTrace2:
            usersWithTrace2.append(u)
        if containsTrace3:
            usersWithTrace3.append(u)
        if containsTrace4:
            usersWithTrace4.append(u)
        if containsTrace5:
            usersWithTrace5.append(u)
        if containsTrace6:
            usersWithTrace6.append(u)
        if containsTrace7:
            usersWithTrace7.append(u)
        if containsTrace8:
            usersWithTrace8.append(u)
        if containsTrace9:
            usersWithTrace9.append(u)
        if containsTrace10:
            usersWithTrace10.append(u)
        if containsTrace11:
            usersWithTrace11.append(u)
        if containsTrace12:
            usersWithTrace12.append(u)
        if containsTrace13:
            usersWithTrace13.append(u)
        if containsTrace14:
            usersWithTrace14.append(u)
        if containsTrace15:
            usersWithTrace15.append(u)
        if containsTrace16:
            usersWithTrace16.append(u)
        if containsTrace17:
            usersWithTrace17.append(u)
        if containsTrace18:
            usersWithTrace18.append(u)
        if containsTrace19:
            usersWithTrace19.append(u)
        if containsTrace20:
            usersWithTrace20.append(u)
        
        
    return [{'utc_red4':usersWithTrace1},{'utc_red3':usersWithTrace2},{'utc_red2':usersWithTrace3}, {'utc_red9':usersWithTrace4},
            {'utcHalfHourly_red4':usersWithTrace5},{'utcHalfHourly_red3':usersWithTrace6},{'utcHalfHourly_red2':usersWithTrace7},{'utcHalfHourly_red9':usersWithTrace8},
            {'utcHourly_red4':usersWithTrace9},{'utcHourly_red3':usersWithTrace10},{'utcHourly_red2':usersWithTrace11},{'utcHourly_red9':usersWithTrace12},
            {'utcBiHourly_red4':usersWithTrace13}, {'utcBiHourly_red3':usersWithTrace14}, {'utcBiHourly_red2':usersWithTrace15}, {'utcBiHourly_red9':usersWithTrace16},
            {'utc3Hours_red4':usersWithTrace17},{'utc3Hours_red3':usersWithTrace18},{'utc3Hours_red2':usersWithTrace19},{'utc3Hours_red9':usersWithTrace20}];


# In[4]:

dataframe = dataframe.set_index('user')


# In[5]:

dataframe['lat_red'] = dataframe['lat'].map(lambda x: x.round(2))
dataframe['lon_red'] = dataframe['lon'].map(lambda x: x.round(2))


# In[5]:

dataframe['lat_red3'] = dataframe['lat'].map(lambda x: x.round(3))
dataframe['lon_red3'] = dataframe['lon'].map(lambda x: x.round(3))

dataframe['lat_red4'] = dataframe['lat'].map(lambda x: x.round(4))
dataframe['lon_red4'] = dataframe['lon'].map(lambda x: x.round(4))


# In[3]:

import datetime

dataframe['utc'] = dataframe['timestamp'].map(lambda x: datetime.datetime.utcfromtimestamp(x/1000.0))

dataframe['timestampBiHourly'] = dataframe['timestamp'].map(lambda x: (round((x/1000.0)/7200))*7200)
dataframe['utcBiHourly'] = dataframe['timestampBiHourly'].map(lambda x: datetime.datetime.utcfromtimestamp(x))

dataframe['timestampHalfHourly'] = dataframe['timestamp'].map(lambda x: (round((x/1000.0)/1800))*1800)
dataframe['utcHalfHourly'] = dataframe['timestampHalfHourly'].map(lambda x: datetime.datetime.utcfromtimestamp(x))


# In[10]:

import pickle

infile = open("./MyDataframe6.pkl", "rb" )
dataframe = pickle.load(infile)


# In[ ]:




# In[4]:

import datetime 

dataframe['timestampHourly'] = dataframe['timestamp'].map(lambda x: (round((x/1000.0)/3600))*3600)
dataframe['utcHourly'] = dataframe['timestampHourly'].map(lambda x: datetime.datetime.utcfromtimestamp(x))


# In[5]:

dataframe['timestamp3Hours'] = dataframe['timestamp'].map(lambda x: (round((x/1000.0)/10800))*10800)
dataframe['utc3Hours'] = dataframe['timestamp3Hours'].map(lambda x: datetime.datetime.utcfromtimestamp(x))


# In[6]:

dataframe['IsInDtu'] = (dataframe['lon'].between(12.486500, 12.544232)) & (dataframe['lat'].between(55.763772, 55.803049))


# In[7]:

dataframeWithoutDtu = dataframe.loc[dataframe['IsInDtu'] == False]


# In[9]:

import pickle

#dataframe.to_pickle("./MyDataframe6.pkl")
dataframeWithoutDtu.to_pickle("./MyDataframeWithoutDtu6.pkl")


# In[109]:



dataframeWithoutDtu


# In[ ]:

import pickle

experimentSize = 1000;
NumberOfdataPoints = [3, 4, 5, 6, 7]
includeDTU = [False, True]

uniqueUsersPerExperiment = []
for i in np.arange(147,experimentSize):
    for sampleSize in NumberOfdataPoints:
        #print("Running experiment " + str(i+1) + " of "+ str(experimentSize) + ". Running data point: " + str(sampleSize));
        pickle.dump(RunUniquenessCheck(sampleSize, dataframe, includeDTU[0]), open("Exp_" + str(i+1)+"_DataPoints_"+str(sampleSize)+"_DTU_"+str(includeDTU[0]), "wb"))
        pickle.dump(RunUniquenessCheck(sampleSize, dataframe, includeDTU[1]), open("Exp_" + str(i+1)+"_DataPoints_"+str(sampleSize)+"_DTU_"+str(includeDTU[1]), "wb"))


# In[ ]:

s = pandas.Series(uniqueUsersPerExperiment)
vc = s.value_counts()
vc = vc.sort_index()
vc.plot(kind ='bar')


# In[49]:

infile = open("Exp_1_DataPoints_3_DTU_True", "rb" )
myList = pickle.load(infile)
myList


# In[30]:

numberOfExperiments = 143


# In[63]:

import pickle

percentage1 = []
percentage2 =[]

for dp in np.arange(3, 8):
    utcHourly_red3 = []
    for i in np.arange(1, numberOfExperiments):
        infile = open("Exp_"+str(i)+"_DataPoints_"+str(dp)
                      +"_DTU_True", "rb" )
        myList = pickle.load(infile)
        utcHourly_red3.append(len(myList[9]['utcHourly_red3']))
    
    print("Information on: " + str(dp) + " data points:");
    print(str((utcHourly_red3.count(1)/(len(utcHourly_red3)*1.0)*100)));
    percentage1.append((utcHourly_red3.count(1)/(len(utcHourly_red3)*1.0)*100))
    percentage2.append((utcHourly_red3.count(2)/(len(utcHourly_red3)*1.0)*100))

    
    

percentage12 = [x + y for x, y in zip(percentage1, percentage2)]



#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
menMeans = tuple(percentage1)
#menStd = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r') #,yerr=menStd)

womenMeans = tuple(percentage12)
#womenStd = (3, 5, 2, 3, 3)
rects2 = ax.bar(ind + width, womenMeans, width, color='y',) #,yerr=womenStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Uniqueness = 1 or 2, utcHourly, red3')
ax.set_xticks(ind + width)
ax.set_xticklabels(('3dp', '4dp', '5dp', '6dp', '7dp'))
ax.set_ylim((55, 100))
ax.legend((rects1[0], rects2[0]), ('# traces =1', '# traces =2'), borderaxespad=0., loc=9)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.savefig('plot_UTCHourly_red3.png')
plt.show()


# In[32]:

percentage1 = []
percentage2 =[]

for dp in np.arange(3, 8):
    utcHourly_red3 = []
    for i in np.arange(1, numberOfExperiments):
        infile = open("Exp_"+str(i)+"_DataPoints_"+str(dp)
                      +"_DTU_True", "rb" )
        myList = pickle.load(infile)
        utcHourly_red3.append(len(myList[5]['utcHalfHourly_red3']))
    
    percentage1.append((utcHourly_red3.count(1)/(len(utcHourly_red3)*1.0)*100))
    percentage2.append((utcHourly_red3.count(2)/(len(utcHourly_red3)*1.0)*100))




percentage12 = [x + y for x, y in zip(percentage1, percentage2)]



#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
menMeans = tuple(percentage1)
#menStd = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r') #,yerr=menStd)

womenMeans = tuple(percentage12)
#womenStd = (3, 5, 2, 3, 3)
rects2 = ax.bar(ind + width, womenMeans, width, color='y',) #,yerr=womenStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Uniqueness = 1 or 2, utcHalfHourly, red3')
ax.set_xticks(ind + width)
ax.set_xticklabels(('3dp', '4dp', '5dp', '6dp', '7dp'))
ax.set_ylim((55, 100))
ax.legend((rects1[0], rects2[0]), ('# traces = 1', '# traces = 2'), borderaxespad=0., loc=9)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.savefig('plot_utcHalfHourly_red3.png')
plt.show()


# In[33]:

percentage1 = []
percentage2 =[]

for dp in np.arange(3, 8):
    utcHourly_red3 = []
    for i in np.arange(1, numberOfExperiments):
        infile = open("Exp_"+str(i)+"_DataPoints_"+str(dp)
                      +"_DTU_True", "rb" )
        myList = pickle.load(infile)
        utcHourly_red3.append(len(myList[1]['utc_red3']))
    
    percentage1.append((utcHourly_red3.count(1)/(len(utcHourly_red3)*1.0)*100))
    percentage2.append((utcHourly_red3.count(2)/(len(utcHourly_red3)*1.0)*100))




percentage12 = [x + y for x, y in zip(percentage1, percentage2)]



#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
menMeans = tuple(percentage1)
#menStd = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r') #,yerr=menStd)

womenMeans = tuple(percentage12)
#womenStd = (3, 5, 2, 3, 3)
rects2 = ax.bar(ind + width, womenMeans, width, color='y',) #,yerr=womenStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Uniqueness = 1 or 2, utc ,red3')
ax.set_xticks(ind + width)
ax.set_xticklabels(('3dp', '4dp', '5dp', '6dp', '7dp'))
ax.set_ylim((55, 100))
ax.legend((rects1[0], rects2[0]), ('# traces = 1', '# traces = 2'), borderaxespad=0., loc=9)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.savefig('plot_utc_red3.png')
plt.show()


# In[34]:

percentage1 = []
percentage2 =[]

for dp in np.arange(3, 8):
    utcHourly_red3 = []
    for i in np.arange(1, numberOfExperiments):
        infile = open("Exp_"+str(i)+"_DataPoints_"+str(dp)
                      +"_DTU_True", "rb" )
        myList = pickle.load(infile)
        utcHourly_red3.append(len(myList[0]['utc_red4']))
    
    percentage1.append((utcHourly_red3.count(1)/(len(utcHourly_red3)*1.0)*100))
    percentage2.append((utcHourly_red3.count(2)/(len(utcHourly_red3)*1.0)*100))




percentage12 = [x + y for x, y in zip(percentage1, percentage2)]



#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
menMeans = tuple(percentage1)
#menStd = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r') #,yerr=menStd)

womenMeans = tuple(percentage12)
#womenStd = (3, 5, 2, 3, 3)
rects2 = ax.bar(ind + width, womenMeans, width, color='y',) #,yerr=womenStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Uniqueness = 1 or 2, utc, red4')
ax.set_xticks(ind + width)
ax.set_xticklabels(('3dp', '4dp', '5dp', '6dp', '7dp'))
ax.set_ylim((55, 100))
ax.legend((rects1[0], rects2[0]), ('# traces =1', '# traces =2'), borderaxespad=0., loc=3)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.savefig('plot_UTC_red4.png')
plt.show()


# In[35]:

percentage1 = []
percentage2 =[]

for dp in np.arange(3, 8):
    utcHourly_red3 = []
    for i in np.arange(1, numberOfExperiments):
        infile = open("Exp_"+str(i)+"_DataPoints_"+str(dp)
                      +"_DTU_True", "rb" )
        myList = pickle.load(infile)
        utcHourly_red3.append(len(myList[4]['utcHalfHourly_red4']))
    
    percentage1.append((utcHourly_red3.count(1)/(len(utcHourly_red3)*1.0)*100))
    percentage2.append((utcHourly_red3.count(2)/(len(utcHourly_red3)*1.0)*100))




percentage12 = [x + y for x, y in zip(percentage1, percentage2)]



#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
menMeans = tuple(percentage1)
#menStd = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r') #,yerr=menStd)

womenMeans = tuple(percentage12)
#womenStd = (3, 5, 2, 3, 3)
rects2 = ax.bar(ind + width, womenMeans, width, color='y',) #,yerr=womenStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Uniqueness = 1 or 2, utcHalfHourly, red4')
ax.set_xticks(ind + width)
ax.set_xticklabels(('3dp', '4dp', '5dp', '6dp', '7dp'))
ax.set_ylim((55, 100))
ax.legend((rects1[0], rects2[0]), ('# traces =1', '# traces =2'), borderaxespad=0., loc=3)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.savefig('plot_utcHalfHourly_red4.png')
plt.show()


# In[36]:

percentage1 = []
percentage2 =[]

for dp in np.arange(3, 8):
    utcHourly_red3 = []
    for i in np.arange(1, numberOfExperiments):
        infile = open("Exp_"+str(i)+"_DataPoints_"+str(dp)
                      +"_DTU_True", "rb" )
        myList = pickle.load(infile)
        utcHourly_red3.append(len(myList[8]['utcHourly_red4']))
    
    percentage1.append((utcHourly_red3.count(1)/(len(utcHourly_red3)*1.0)*100))
    percentage2.append((utcHourly_red3.count(2)/(len(utcHourly_red3)*1.0)*100))




percentage12 = [x + y for x, y in zip(percentage1, percentage2)]



#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
menMeans = tuple(percentage1)
#menStd = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r') #,yerr=menStd)

womenMeans = tuple(percentage12)
#womenStd = (3, 5, 2, 3, 3)
rects2 = ax.bar(ind + width, womenMeans, width, color='y',) #,yerr=womenStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Uniqueness = 1 or 2, utcHourly, red4')
ax.set_xticks(ind + width)
ax.set_xticklabels(('3dp', '4dp', '5dp', '6dp', '7dp'))
ax.set_ylim((55, 100))
ax.legend((rects1[0], rects2[0]), ('# traces =1', '# traces =2'), borderaxespad=0., loc=3)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.savefig('plot_utcHourly_red4.png')
plt.show()


# 

# #### 4 Decimal points = 70 sq meters area.
# #### 3 Decimal points = 6941 m² | 0.01 km² |
# #### 2 Decimal points = 694,061 m² | 0.69 km² |

# In[359]:

utcHourly_red3 = []

infile = open("RuiExp_788_DataPoints_2_DTU_True", "rb" )
myList = pickle.load(infile)
print(myList)
#utcHourly_red2.append(len(myList[10]['utcHourly_red2']))


# In[ ]:




# In[377]:

percentagesDict = {}

for dp in [2, 13]:
    utcHourly_red3 = []
    for i in np.arange(0, 100):
        for j in np.arange(1, 11):
            infile = open("RuiExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                          +"_DTU_True", "rb" )
            myList = pickle.load(infile)
            utcHourly_red3.append(len(myList))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_red3.count(1)/(len(utcHourly_red3)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_red3.count(2)/(len(utcHourly_red3)*1.0)*100)




import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in [2, 13]:
    hundredIters = []
    for i in np.arange(0, 100):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in [2, 13]:
    hundredIters2 = []
    for i in np.arange(0, 100):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))


# In[379]:

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 2
menMeans = tuple(percentage1)
menStd = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r',yerr=menStd)

womenMeans = tuple(percentage2)
womenStd = tuple(std2)
rects2 = ax.bar(ind + width, womenMeans, width, color='y',yerr=womenStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Uniqueness = 1 or 2, utcHalfHourly, red3')
ax.set_xticks(ind + width)
ax.set_xticklabels(('2dp', '13dp'))
ax.set_ylim((30, 100))
ax.legend((rects1[0], rects2[0]), ('# traces = 1', '# traces = 2'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.savefig('plot_utcHourly_red2.png')
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[2]:

def AuxCalc(userInt, sample, dfToUse, timeRes, spaceRes):
    firstIter = True
    usersToCheck = {userInt};
    for idx, s in sample.iterrows():
        if(firstIter):
            firstIter = False
            series = ((s[spaceRes[0]] == dfToUse[spaceRes[0]]) & (s[spaceRes[1]] == dfToUse[spaceRes[1]]) & (s[timeRes] == dfToUse[timeRes]))
            usersToCheck = set(series[series == True].keys())
        else:
            newUsersToCheck = usersToCheck.copy()
            for i in usersToCheck:
                udf = dfToUse.loc[(i)]
                if not ((s[spaceRes[0]] == udf[spaceRes[0]]) & (s[spaceRes[1]] == udf[spaceRes[1]]) & (s[timeRes] == udf[timeRes])).any():
                    newUsersToCheck.remove(i)
            usersToCheck = newUsersToCheck
            if(len(usersToCheck) == 1):
                return usersToCheck;
    return usersToCheck;


# In[3]:

#hopeful one
def RunUniquenessCheck(sampleSize, myDataframe, myDataframeWithoutDtu,includeDtu):

    timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
    spatialResolution = [['lat_red1', 'lon_red1']];


    dfToUse = None;
    if includeDtu:
        dfToUse = myDataframe
    else:
        dfToUse = myDataframeWithoutDtu
        
    sample = None;
    userInt = None;
    while(True):
        # choose n random points from one user:
        userInt = np.random.randint(1, 846)    
        try:
            udf = dfToUse.loc[(userInt)]
        except KeyError:
            continue
        # make sure user has enough points.
        if(type(udf) is pandas.DataFrame and len(udf.index) >= sampleSize):
            sample = udf.sample(sampleSize)
            break;
    response = []
    for time in timeResolution:
        for space in spatialResolution:
            res = AuxCalc(userInt, sample, dfToUse, time, space)
            response.append(res)
    
    return response


# In[ ]:




# In[4]:

# hopeful one
def RunChecks(i):
    NumberOfdataPoints = [2, 3, 4, 5, 6, 7]
    includeDTU = [False, True]

    uniqueUsersPerExperiment = []
    for sampleSize in NumberOfdataPoints:
        print("Running experiment " + str(i+1) + ". Running data point: " + str(sampleSize));
        pickle.dump(RunUniquenessCheck(sampleSize, dataframe, dataframeWithoutDtu, includeDTU[0]), open("./BigExp/latred1_" + str(i+1)+"_DataPoints_"+str(sampleSize)+"_DTU_"+str(includeDTU[0]), "wb"))
        pickle.dump(RunUniquenessCheck(sampleSize, dataframe, dataframeWithoutDtu, includeDTU[1]), open("./BigExp/latred1_" + str(i+1)+"_DataPoints_"+str(sampleSize)+"_DTU_"+str(includeDTU[1]), "wb"))


# In[ ]:




# In[1]:

import pickle


infile = open("./MyDataframe.pkl", "rb" )
dataframe = pickle.load(infile)


# In[ ]:

dataframe['timestampDaily'] = dataframe['timestamp'].map(lambda x: (round((x/1000.0)/86400))*86400)


dataframe['lat_red1'] = dataframe['lat'].map(lambda x: x.round(1))
dataframe['lon_red1'] = dataframe['lon'].map(lambda x: x.round(1))


# In[282]:

import pickle

infile = open("./MyDataframeWithoutDtu.pkl", "rb" )
dataframeWithoutDtu = pickle.load(infile)


# In[9]:

dataframeWithoutDtu['timestampDaily'] = dataframeWithoutDtu['timestamp'].map(lambda x: (round((x/1000.0)/86400))*86400)


# In[10]:

dataframeWithoutDtu['lat_red1'] = dataframeWithoutDtu['lat'].map(lambda x: x.round(1))
dataframeWithoutDtu['lon_red1'] = dataframeWithoutDtu['lon'].map(lambda x: x.round(1))


# In[ ]:

def AuxCalc(userInt, sample, dfToUse, timeRes, spaceRes):
    firstIter = True
    usersToCheck = {userInt};
    for idx, s in sample.iterrows():
        if(firstIter):
            firstIter = False
            series = ((s[spaceRes[0]] == dfToUse[spaceRes[0]]) & (s[spaceRes[1]] == dfToUse[spaceRes[1]]) & (s[timeRes] == dfToUse[timeRes]))
            usersToCheck = set(series[series == True].keys())
        else:
            newUsersToCheck = usersToCheck.copy()
            for i in usersToCheck:
                udf = dfToUse.loc[(i)]
                if not ((s[spaceRes[0]] == udf[spaceRes[0]]) & (s[spaceRes[1]] == udf[spaceRes[1]]) & (s[timeRes] == udf[timeRes])).any():
                    newUsersToCheck.remove(i)
            usersToCheck = newUsersToCheck
            if(len(usersToCheck) == 1):
                return usersToCheck;
    return usersToCheck;


# In[2]:

def decreaseCounter(experimentType):
    if experimentType[0] > 0:
        experimentType[0] -=1
        return experimentType
    if experimentType[1] > 0:
        experimentType[0] -=1
        return experimentType
    if experimentType[2] > 0:
        experimentType[0] -=1
        return experimentType


# In[3]:

def DynamicAuxCalc(userInt, sample, dfToUse, experimentType):
    
    timeResolution = ['utc3Hours', 'timestampDaily']
    spatialResolution = [['lat_red', 'lon_red'], ['lat_red1', 'lon_red1']];
    
    firstIter = True
    usersToCheck = {userInt};
    timeRes = None;
    spaceRes = None;
    #for idx, s in sample.iterrows():
    # experimentType is : [1, 1, 0]
    
    for idx, s in sample.iterrows():
        if sum(experimentType) > 0:
            if(firstIter):
                firstIter = False
                if experimentType[0] > 0:
                    timeRes = timeResolution[0]
                    spaceRes = spatialResolution[0]
                elif experimentType[1] > 0:
                    timeRes = timeResolution[1]
                    spaceRes = spatialResolution[0]
                elif experimentType[2] > 0:
                    timeRes = timeResolution[1]
                    spaceRes = spatialResolution[1]
                series = ((s[spaceRes[0]] == dfToUse[spaceRes[0]]) & (s[spaceRes[1]] == dfToUse[spaceRes[1]]) & (s[timeRes] == dfToUse[timeRes]))
                usersToCheck = set(series[series == True].keys())
                experimentType = decreaseCounter(experimentType)
                continue;
            else:
                newUsersToCheck = usersToCheck.copy()
                for i in usersToCheck:
                    udf = dfToUse.loc[(i)]

                    if experimentType[0] > 0:
                        timeRes = timeResolution[0]
                        spaceRes = spatialResolution[0]
                    elif experimentType[1] > 0:
                        timeRes = timeResolution[1]
                        spaceRes = spatialResolution[0]
                    elif experimentType[2] > 0:
                        timeRes = timeResolution[1]
                        spaceRes = spatialResolution[1]
                    if not ((s[spaceRes[0]] == udf[spaceRes[0]]) & (s[spaceRes[1]] == udf[spaceRes[1]]) & (s[timeRes] == udf[timeRes])).any():
                        newUsersToCheck.remove(i)
                usersToCheck = newUsersToCheck
                if(len(usersToCheck) == 1):
                    return usersToCheck;
            experimentType = decreaseCounter(experimentType)
    return usersToCheck;


# In[4]:

def RunDynamicUniquenessCheck(experimentType, myDataframe, myDataframeWithoutDtu,includeDtu):

    sampleSize = sum(experimentType)
    
    dfToUse = myDataframe
        
    sample = None;
    userInt = None;
    while(True):
        # choose n random points from one user:
        userInt = np.random.randint(1, 846)    
        try:
            udf = dfToUse.loc[(userInt)]
        except KeyError:
            continue
        # make sure user has enough points.
        if(type(udf) is pandas.DataFrame and len(udf.index) >= sampleSize):
            sample = udf.sample(sampleSize)
            break;
    response = []
    
    res = DynamicAuxCalc(userInt, sample, dfToUse, experimentType)
    response.append(res)
    
    return response


# In[5]:

# hopeful one
def RunDynamicChecks(i):
    #_ 3 hours, lat_red2
    #_  daily, lat_red2
    #_ daily, lat_red1
        
    NumberOfdataPoints = [[1, 1, 0]]
    includeDTU = [True]
    dataframeWithoutDtu = dataframe
    uniqueUsersPerExperiment = []
    for sampleSize in NumberOfdataPoints:
        dps = str(sampleSize[0])+ str(sampleSize[1])+ str(sampleSize[2])
        pickle.dump(RunDynamicUniquenessCheck(sampleSize, dataframe, dataframeWithoutDtu, includeDTU[0]), open("./DynamicExp/DynamicExp" + str(i+1)+"_DataPoints_"+dps+"_DTU_"+str(includeDTU[0]), "wb"))


# In[ ]:




# In[ ]:

import pickle

import pandas
infile = open("./dynamic.pkl", "rb" )
dataframe = pickle.load(infile)


# In[ ]:

import pickle
from sensible_raw.loaders import loader
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas
from multiprocessing import Pool
import numpy as np
import os.path

p = Pool(processes=3)
expNum = np.arange(0, 500)
result = p.map(RunDynamicChecks, expNum)


# In[ ]:




# In[ ]:




# Label Testing

# In[71]:

# hopeful one
def RunShuffleLabelChecks(i):
    NumberOfdataPoints = [3, 6, 9, 12, 15, 18]
    for sampleSize in NumberOfdataPoints:
        pickle.dump(RunShuffleLabelUniquenessCheck(sampleSize, dataframe), open("./ShuffleLabelExperiment/ShuffleLabel_Home_" + str(i+1)+"_DataPoints_"+str(sampleSize)+"_DTU_True", "wb"))


# In[72]:

def RunShuffleLabelUniquenessCheck(sampleSize, myDataframe):

    timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']

    dfToUse = myDataframe;
        
    sample = None;
    userInt = None;
    while(True):
        # choose n random points from one user:
        userInt = np.random.randint(1, 846)    
        try:
            udf = dfToUse.loc[(userInt)]
        except KeyError:
            continue
        # make sure user has enough points.
        if(type(udf) is pandas.DataFrame and len(udf.index) >= sampleSize):
            sample = udf.sample(sampleSize)
            break;
    response = []
    for time in timeResolution:
        res = AuxShuffleLabelCalc(userInt, sample, dfToUse, time)
        response.append(res)
    
    return response


# In[73]:

def AuxShuffleLabelCalc(userInt, sample, dfToUse, timeRes):
    firstIter = True
    usersToCheck = {userInt};
    for idx, s in sample.iterrows():
        if(firstIter):
            firstIter = False
            series = ((s['AtHomeShuffle'] == dfToUse['AtHomeShuffle'])& (s[timeRes] == dfToUse[timeRes]))
            usersToCheck = set(series[series == True].keys())
        else:
            newUsersToCheck = usersToCheck.copy()
            for i in usersToCheck:
                udf = dfToUse.loc[(i)]
                if not ((s['AtHomeShuffle'] == udf['AtHomeShuffle'])& (s[timeRes] == udf[timeRes])).any():
                    newUsersToCheck.remove(i)
            usersToCheck = newUsersToCheck
            if(len(usersToCheck) == 1):
                return usersToCheck;
    return usersToCheck;


# In[4]:

def AtWork(r):
    try:
        return (abs(float(r.lat_red3) - float(works[r.user][0])) <= 0.01) and (abs(float(r.lon_red3) - float(works[r.user][1])) <= 0.01)
    except KeyError:
        return False

        

    
def AtHome(r):
    try:
        return (abs(float(r.lat_red3) - float(homes[r.user][0])) <= 0.01) and (abs(float(r.lon_red3) - float(homes[r.user][1])) <= 0.01)
    except KeyError:
        return False


# In[283]:

import pickle

import pandas
infile = open("./MyDataframe4.pkl", "rb" )
dataframe = pickle.load(infile)


# In[6]:

infile = open("./homes", "rb" )
homes = pickle.load(infile)

infile = open("./works", "rb" )
works = pickle.load(infile)



# In[296]:

test = dataframe[['utc3Hours', 'timestampDaily', 'lat_red', 'lon_red', 'lat_red1', 'lon_red1']]
test = dataframe[['timestamp', 'lat', 'lon', 'utc3Hours', 'lat_red', 'lon_red']]


# In[298]:

test['timestampDaily'] = test['timestamp'].map(lambda x: (round((x/1000.0)/86400))*86400)
test['lat_red1'] = test['lat'].map(lambda x: x.round(1))
test['lon_red1'] = test['lon'].map(lambda x: x.round(1))


# In[301]:

test = test[['utc3Hours', 'timestampDaily', 'lat_red', 'lon_red', 'lat_red1', 'lon_red1']]


# In[302]:

test


# In[7]:

dataframe['user'] = dataframe.index


# In[ ]:




# In[10]:

dataframe['AtWork'] = dataframe.apply(AtWork, axis=1)


# In[ ]:




# In[ ]:




# In[8]:

dataframe['AtHome'] = dataframe.apply(AtHome, axis=1)


# In[ ]:




# In[303]:

test.to_pickle("./dynamic.pkl")


# In[13]:

print("Hello")


# In[ ]:




# In[ ]:




# In[68]:

import pickle
from sensible_raw.loaders import loader
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas
from multiprocessing import Pool
import numpy as np
import os.path

p = Pool(processes=2)
expNum = np.arange(0, 100)
result = p.map(RunLabelChecks, expNum)


# In[57]:

infile = open("./BigExp/Label_9_DataPoints_10_DTU_True", "rb" )
myList = pickle.load(infile)
myList


# In[6]:

dataframe


# In[34]:

tmpDF = dataframe[['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours', 'AtHome']]


# In[61]:

import numpy as np
labelShuffleDF = pandas.DataFrame()
for i in np.arange(0, 846):
    try:
        udf = tmpDF.loc[(i)]
    except KeyError:
        continue
    shuffle = np.copy(udf[['AtHome']].values)
    np.random.shuffle(shuffle)
    udf['AtHomeShuffle'] = shuffle
    labelShuffleDF = labelShuffleDF.append(udf)


# In[51]:

labelShuffleDF


# In[62]:

labelShuffleDF.to_pickle("./labelShuffle.pkl")


# In[79]:

import pickle

import pandas
infile = open("./labelShuffle.pkl", "rb" )
dataframe = pickle.load(infile)


# In[80]:

dataframe


# In[74]:

import pickle
from sensible_raw.loaders import loader
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas
from multiprocessing import Pool
import numpy as np
import os.path

p = Pool(processes=2)
expNum = np.arange(0, 500)
result = p.map(RunShuffleLabelChecks, expNum)


# In[64]:

dataframe


# In[75]:

# hopeful one
def RunLabelChecks(i):
    NumberOfdataPoints = [3, 6, 9, 12, 15, 18]
    for sampleSize in NumberOfdataPoints:
        pickle.dump(RunLabelUniquenessCheck(sampleSize, dataframe), open("./LabelExperiment/Label_Home_" + str(i+1)+"_DataPoints_"+str(sampleSize)+"_DTU_True", "wb"))


# In[76]:

def RunLabelUniquenessCheck(sampleSize, myDataframe):

    timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']

    dfToUse = myDataframe;
        
    sample = None;
    userInt = None;
    while(True):
        # choose n random points from one user:
        userInt = np.random.randint(1, 846)    
        try:
            udf = dfToUse.loc[(userInt)]
        except KeyError:
            continue
        # make sure user has enough points.
        if(type(udf) is pandas.DataFrame and len(udf.index) >= sampleSize):
            sample = udf.sample(sampleSize)
            break;
    response = []
    for time in timeResolution:
        res = AuxLabelCalc(userInt, sample, dfToUse, time)
        response.append(res)
    
    return response


# In[77]:

def AuxLabelCalc(userInt, sample, dfToUse, timeRes):
    firstIter = True
    usersToCheck = {userInt};
    for idx, s in sample.iterrows():
        if(firstIter):
            firstIter = False
            series = ((s['AtHome'] == dfToUse['AtHome'])& (s[timeRes] == dfToUse[timeRes]))
            usersToCheck = set(series[series == True].keys())
        else:
            newUsersToCheck = usersToCheck.copy()
            for i in usersToCheck:
                udf = dfToUse.loc[(i)]
                if not ((s['AtHome'] == udf['AtHome'])& (s[timeRes] == udf[timeRes])).any():
                    newUsersToCheck.remove(i)
            usersToCheck = newUsersToCheck
            if(len(usersToCheck) == 1):
                return usersToCheck;
    return usersToCheck;


# In[ ]:

import pickle
from sensible_raw.loaders import loader
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas
from multiprocessing import Pool
import numpy as np
import os.path

p = Pool(processes=2)
expNum = np.arange(0, 500)
result = p.map(RunLabelChecks, expNum)


# In[ ]:




# In[ ]:




# In[273]:

import pickle
import pandas

infile = open("./labelShuffle.pkl", "rb" )
dataframe = pickle.load(infile)


# In[ ]:




# In[ ]:




# In[52]:




# In[274]:

udf = dataframe.reset_index()[['user', 'utcHalfHourly', 'AtHomeShuffle']]
udf = udf.drop_duplicates(subset = ['user', 'utcHalfHourly'])


# In[ ]:




# In[275]:

udf = udf.pivot(index = 'user', columns='utcHalfHourly', values='AtHomeShuffle')


# In[223]:

import pickle
udf.to_pickle("./matrixShuffle.pkl")


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[224]:

import pickle
import numpy

infile = open("./matrixShuffle.pkl", "rb" )
dataframe = pickle.load(infile)

percentage_home = []

for i in np.arange(0, 845):
    tmpDF = dataframe.loc[i].values
    cleanUdf = [x for x in tmpDF if str(x) != 'nan']
    percentage_home.append(cleanUdf.count(True)/len(cleanUdf)*100)


# In[225]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
hist, bins = np.histogram(percentage_home)
plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), facecolor='y', alpha=0.75)

plt.xlabel('Percentage of time spent at home')
plt.ylabel('Percentage of users')
plt.title('Histogram of time spent home')
#plt.axis([40, 160, 0, 1])
plt.grid(True)
plt.show()


# In[104]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
hist, bins = np.histogram(percentage_home)
plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), facecolor='y', alpha=0.75)

plt.xlabel('Percentage of time spent at home')
plt.ylabel('Percentage of users')
plt.title('Histogram of time spent home')
#plt.axis([40, 160, 0, 1])
plt.grid(True)
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# # Matrix Label

# In[227]:

import pickle
import numpy

infile = open("./matrixShuffle.pkl", "rb" )
dataframe = pickle.load(infile)


# In[276]:

dataframe = udf


# In[277]:

dataframe


# In[ ]:




# In[ ]:




# In[ ]:




# In[209]:




# In[278]:

def AuxLabelCalcMatrix(userInt, sample, dfToUse):
    users = np.arange(0, 846)
    for idx, s in sample.iteritems():
        users = dataframe.loc[users].loc[ dataframe[idx] == s].index
    return users;


# In[279]:

def RunLabelUniquenessCheckMatrix(sampleSize, myDataframe):
    
    dfToUse = myDataframe;
    sample = None;
    userInt = None;
    
    while(True):
        # choose n random points from one user:
        userInt = np.random.randint(1, 846)    
        try:
            udf = dfToUse.loc[(userInt)]
        except KeyError:
            continue
        # make sure user has enough points.
        if(udf[(udf==True) | (udf==False)].size >= sampleSize):
            sample = udf[(udf==True) | (udf==False)].sample(sampleSize)
            break;
    response = []
    res = AuxLabelCalcMatrix(userInt, sample, dfToUse)
    response.append(res)
    
    return response


# In[280]:

def RunLabelChecksMatrix(i):
    NumberOfdataPoints = [3, 6, 9, 12, 15, 18]
    for sampleSize in NumberOfdataPoints:
        pickle.dump(RunLabelUniquenessCheckMatrix(sampleSize, dataframe), open("./LabelShuffleExperimentMatrix/30minutes/LabelShuffle_Home_" + str(i+1)+"_DataPoints_"+str(sampleSize)+"_DTU_True", "wb"))


# In[ ]:

import pickle
from sensible_raw.loaders import loader
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas
from multiprocessing import Pool
import numpy as np
import os.path

p = Pool(processes=2)
expNum = np.arange(0, 500)
result = p.map(RunLabelChecksMatrix, expNum)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



