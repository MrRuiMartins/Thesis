
# coding: utf-8

# In[ ]:




# Having done the uniqueness analysis, we will now plot the results in a graph.
# 
# Visualizing the data is often a useful way to better understand the data.
# 
# We will start by plotting the 

# In[16]:

def convertidxToName(idx):
    if idx == 0:
        return "TimeResolution: 15 minutes. SpaceResolution: 4 gps decimal digits"
    if idx == 1:
        return "TimeResolution: 15 minutes. SpaceResolution: 3 gps decimal digits"
    if idx == 2:
        return "TimeResolution: 15 minutes. SpaceResolution: 2 gps decimal digits"
    if idx == 3:
        return "TimeResolution: 30 minutes. SpaceResolution: 4 gps decimal digits"
    if idx == 4:
        return "TimeResolution: 30 minutes. SpaceResolution: 3 gps decimal digits"
    if idx == 5:
        return "TimeResolution: 30 minutes. SpaceResolution: 2 gps decimal digits"
    if idx == 6:
        return "TimeResolution: 1 Hour. SpaceResolution: 4 gps decimal digits"
    if idx == 7:
        return "TimeResolution: 1 Hour. SpaceResolution: 3 gps decimal digits"
    if idx == 8:
        return "TimeResolution: 1 Hour. SpaceResolution: 2 gps decimal digits"
    if idx == 9:
        return "TimeResolution: 2 Hour. SpaceResolution: 4 gps decimal digits"
    if idx == 10:
        return "TimeResolution: 2 Hour. SpaceResolution: 3 gps decimal digits"
    if idx == 11:
        return "TimeResolution: 2 Hour. SpaceResolution: 2 gps decimal digits"
    if idx == 12:
        return "TimeResolution: 3 Hour. SpaceResolution: 4 gps decimal digits"
    if idx == 13:
        return "TimeResolution: 3 Hour. SpaceResolution: 3 gps decimal digits"
    if idx == 14:
        return "TimeResolution: 3 Hour. SpaceResolution: 2 gps decimal digits"
    


# In[ ]:

# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];


# In[17]:

for idx in np.arange(0, 15):
    import pickle
    percentagesDict = {}

    errorCount = 0
    for dp in np.arange(2, 8):
        results = []
        for i in np.arange(0, 100):
            for j in np.arange(1, 11):
                try:
                    infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                              +"_DTU_False", "rb" )
                except FileNotFoundError:
                    errorCount +=1
                    continue;
                myList = pickle.load(infile)
                # 8 is utc Hourly, latitude and longitude to 2 decimal digits (0.69  km²)
                results.append(len(myList[idx]))
            if len(results) > 0:
                percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (results.count(1)/(len(results)*1.0)*100)
                percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(results.count(2)/(len(results)*1.0)*100)


    import numpy
    percentage1 = []
    std1= []
    percentageAverages =[]
    for j in np.arange(2, 8):
        hundredIters = []
        for i in np.arange(0, 100):
            try:
                hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
            except KeyError:
                continue;
        percentage1.append(sum(hundredIters) / float(len(hundredIters)))
        std1.append(numpy.std(hundredIters))

    percentage2 = []
    std2= []
    percentageAverages2 =[]
    for j in np.arange(2, 8):
        hundredIters2 = []
        for i in np.arange(0, 100):
            try:
                hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
            except KeyError:
                continue;
        percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
        std2.append(numpy.std(hundredIters2))

    #!/usr/bin/env python
    # a bar plot with errorbars
    get_ipython().magic('matplotlib inline')
    import numpy as np
    import matplotlib.pyplot as plt

    N = 6
    trace1 = tuple(percentage1)
    trace1Std = tuple(std1)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

    trace2 = tuple(percentage2)
    trace2Std = tuple(std2)
    rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Percentages')
    ax.set_title(convertidxToName(idx)+'. Excluding Dtu')
    ax.set_xticks(ind + width)
    ax.set_xlabel('Number of Spatio-temporal points')
    ax.set_xticklabels(('2','3','4','5','6', '7'))
    ax.set_ylim((60, 100))
    ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    #autolabel(rects1)
    #autolabel(rects2)

    plt.savefig('./results/DTU_False/'+convertidxToName(idx)+'.png')
    plt.show()


# In[18]:

for idx in np.arange(0, 15):
    import pickle
    percentagesDict = {}

    errorCount = 0
    for dp in np.arange(2, 8):
        results = []
        for i in np.arange(0, 100):
            for j in np.arange(1, 11):
                try:
                    infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                              +"_DTU_True", "rb" )
                except FileNotFoundError:
                    errorCount +=1
                    continue;
                myList = pickle.load(infile)
                # 8 is utc Hourly, latitude and longitude to 2 decimal digits (0.69  km²)
                results.append(len(myList[idx]))
            if len(results) > 0:
                percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (results.count(1)/(len(results)*1.0)*100)
                percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(results.count(2)/(len(results)*1.0)*100)


    import numpy
    percentage1 = []
    std1= []
    percentageAverages =[]
    for j in np.arange(2, 8):
        hundredIters = []
        for i in np.arange(0, 100):
            try:
                hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
            except KeyError:
                continue;
        percentage1.append(sum(hundredIters) / float(len(hundredIters)))
        std1.append(numpy.std(hundredIters))

    percentage2 = []
    std2= []
    percentageAverages2 =[]
    for j in np.arange(2, 8):
        hundredIters2 = []
        for i in np.arange(0, 100):
            try:
                hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
            except KeyError:
                continue;
        percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
        std2.append(numpy.std(hundredIters2))

    #!/usr/bin/env python
    # a bar plot with errorbars
    get_ipython().magic('matplotlib inline')
    import numpy as np
    import matplotlib.pyplot as plt

    N = 6
    trace1 = tuple(percentage1)
    trace1Std = tuple(std1)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

    trace2 = tuple(percentage2)
    trace2Std = tuple(std2)
    rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Percentages')
    ax.set_title(convertidxToName(idx)+'. Including Dtu')
    ax.set_xticks(ind + width)
    ax.set_xlabel('Number of Spatio-temporal points')
    ax.set_xticklabels(('2','3','4','5','6', '7'))
    ax.set_ylim((60, 100))
    ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    #autolabel(rects1)
    #autolabel(rects2)

    plt.savefig('./results/DTU_True/'+convertidxToName(idx)+'.png')
    plt.show()


# In[105]:

# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];


# In[5]:

# DTU Points False

# UTC Hourly, lat_red
# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];
import pickle
percentagesDict = {}

for dp in np.arange(3, 8):
    utcHourly_latred = []
    for i in np.arange(0, 40):
        for j in np.arange(1, 11):
            infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                          +"_DTU_False", "rb" )
            myList = pickle.load(infile)
            # 8 is utc Hourly, latitude and longitude to 2 decimal digits (0.69  km²)
            utcHourly_latred.append(len(myList[8]))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_latred.count(1)/(len(utcHourly_latred)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_latred.count(2)/(len(utcHourly_latred)*1.0)*100)
        



import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in np.arange(3, 8):
    hundredIters = []
    for i in np.arange(0, 40):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in np.arange(3, 8):
    hundredIters2 = []
    for i in np.arange(0, 40):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
trace1 = tuple(percentage1)
trace1Std = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

trace2 = tuple(percentage2)
trace2Std = tuple(std2)
rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Resolution: 1 hour, 2 gps decimal digits (0.69 km^2)')
ax.set_xticks(ind + width)
ax.set_xlabel('Number of Spatio-temporal points')
ax.set_xticklabels(('3','4','5','6', '7'))
ax.set_ylim((60, 100))
ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.savefig('./results/DTU_False/BigExp_1Hours_latlonred.png')
plt.show()

# UTC , lat_red
# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];
percentagesDict = {}

for dp in np.arange(3, 8):
    utcHourly_latred = []
    for i in np.arange(0, 40):
        for j in np.arange(1, 11):
            infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                          +"_DTU_False", "rb" )
            myList = pickle.load(infile)
            
            # 2 is utc 15 minutes, latitude and longitude to 2 decimal digits (0.69  km²)
            idx = 2
            utcHourly_latred.append(len(myList[idx]))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_latred.count(1)/(len(utcHourly_latred)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_latred.count(2)/(len(utcHourly_latred)*1.0)*100)
        



import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in np.arange(3, 8):
    hundredIters = []
    for i in np.arange(0, 40):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in np.arange(3, 8):
    hundredIters2 = []
    for i in np.arange(0, 40):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
trace1 = tuple(percentage1)
trace1Std = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

trace2 = tuple(percentage2)
trace2Std = tuple(std2)
rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Resolution: 15 minutes, 2 gps decimal digits (0.69 km^2)')
ax.set_xticks(ind + width)
ax.set_xlabel('Number of Spatio-temporal points')
ax.set_xticklabels(('3','4','5','6', '7'))
ax.set_ylim((60, 100))
ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.savefig('./results/DTU_False/BigExp_15min_latlonred.png')
plt.show()

# UTC 3 hours , lat_red
# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];
percentagesDict = {}

for dp in np.arange(3, 8):
    utcHourly_latred = []
    for i in np.arange(0, 40):
        for j in np.arange(1, 11):
            if(i*10+j > 198):
                continue;
            infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                          +"_DTU_False", "rb" )
            myList = pickle.load(infile)
            
            # 14 is utc 3 hours, latitude and longitude to 2 decimal digits (0.69  km²)
            idx = 14
            utcHourly_latred.append(len(myList[idx]))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_latred.count(1)/(len(utcHourly_latred)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_latred.count(2)/(len(utcHourly_latred)*1.0)*100)
        



import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in np.arange(3, 8):
    hundredIters = []
    for i in np.arange(0, 40):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in np.arange(3, 8):
    hundredIters2 = []
    for i in np.arange(0, 40):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
trace1 = tuple(percentage1)
trace1Std = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

trace2 = tuple(percentage2)
trace2Std = tuple(std2)
rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Resolution: 3 hours, 2 gps decimal digits (0.69 km^2)')
ax.set_xticks(ind + width)
ax.set_xlabel('Number of Spatio-temporal points')
ax.set_xticklabels(('3','4','5','6', '7'))
ax.set_ylim((60, 100))
ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.savefig('./results/DTU_False/BigExp_3Hours_latlonred.png')
plt.show()

# UTC 15 minutes , lat_red4
# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];
percentagesDict = {}

for dp in np.arange(3, 8):
    utcHourly_latred = []
    for i in np.arange(0, 40):
        for j in np.arange(1, 11):
            if(i*10+j > 198):
                continue;
            infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                          +"_DTU_False", "rb" )
            myList = pickle.load(infile)
            
            # 0 is utc 15 minutes, latitude and longitude to 4 decimal digits (70 sq meters)
            idx = 0
            utcHourly_latred.append(len(myList[idx]))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_latred.count(1)/(len(utcHourly_latred)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_latred.count(2)/(len(utcHourly_latred)*1.0)*100)
        



import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in np.arange(3, 8):
    hundredIters = []
    for i in np.arange(0, 40):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in np.arange(3, 8):
    hundredIters2 = []
    for i in np.arange(0, 40):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
trace1 = tuple(percentage1)
trace1Std = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

trace2 = tuple(percentage2)
trace2Std = tuple(std2)
rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Resolution: 15 mintues, 4 gps decimal digits (70 sq meters)')
ax.set_xticks(ind + width)
ax.set_xlabel('Number of Spatio-temporal points')
ax.set_xticklabels(('3','4','5','6', '7'))
ax.set_ylim((60, 100))
ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.savefig('./results/DTU_False/BigExp_15min_latlonred4.png')
plt.show()

# UTC Hourly , lat_red4
# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];
percentagesDict = {}

for dp in np.arange(3, 8):
    utcHourly_latred = []
    for i in np.arange(0, 40):
        for j in np.arange(1, 11):
            if(i*10+j > 198):
                continue;
            infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                          +"_DTU_False", "rb" )
            myList = pickle.load(infile)
            
            # 6 is utc 1Hour, latitude and longitude to 4 decimal digits (70 sq meters)
            idx = 6
            utcHourly_latred.append(len(myList[idx]))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_latred.count(1)/(len(utcHourly_latred)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_latred.count(2)/(len(utcHourly_latred)*1.0)*100)
        



import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in np.arange(3, 8):
    hundredIters = []
    for i in np.arange(0, 40):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in np.arange(3, 8):
    hundredIters2 = []
    for i in np.arange(0, 40):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
trace1 = tuple(percentage1)
trace1Std = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

trace2 = tuple(percentage2)
trace2Std = tuple(std2)
rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Resolution: 1 Hour, 4 gps decimal digits (70 sq meters)')
ax.set_xticks(ind + width)
ax.set_xlabel('Number of Spatio-temporal points')
ax.set_xticklabels(('3','4','5','6', '7'))
ax.set_ylim((60, 100))
ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.savefig('./results/DTU_False/BigExp_1Hours_latlonred4.png')
plt.show()

# UTC 3 Hours , lat_red4
# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];
percentagesDict = {}

for dp in np.arange(3, 8):
    utcHourly_latred = []
    for i in np.arange(0, 40):
        for j in np.arange(1, 11):
            if(i*10+j > 198):
                continue;
            infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                          +"_DTU_False", "rb" )
            myList = pickle.load(infile)
            
            # 12 is utc 3Hours, latitude and longitude to 4 decimal digits (70 sq meters)
            idx = 12
            utcHourly_latred.append(len(myList[idx]))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_latred.count(1)/(len(utcHourly_latred)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_latred.count(2)/(len(utcHourly_latred)*1.0)*100)
        



import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in np.arange(3, 8):
    hundredIters = []
    for i in np.arange(0, 40):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in np.arange(3, 8):
    hundredIters2 = []
    for i in np.arange(0, 40):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
trace1 = tuple(percentage1)
trace1Std = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

trace2 = tuple(percentage2)
trace2Std = tuple(std2)
rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Resolution: 3 Hour, 4 gps decimal digits (70 sq meters)')
ax.set_xticks(ind + width)
ax.set_xlabel('Number of Spatio-temporal points')
ax.set_xticklabels(('3','4','5','6', '7'))
ax.set_ylim((60, 100))
ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.savefig('./results/DTU_False/BigExp_3Hours_latlonred4.png')
plt.show()


# In[ ]:










# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[6]:

# DTU Points True

# UTC Hourly, lat_red
# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];
percentagesDict = {}

for dp in np.arange(3, 8):
    utcHourly_latred = []
    for i in np.arange(0, 40):
        for j in np.arange(1, 11):
            if(i*10+j > 198):
                continue;
            infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                          +"_DTU_True", "rb" )
            myList = pickle.load(infile)
            # 8 is utc Hourly, latitude and longitude to 2 decimal digits (0.69  km²)
            utcHourly_latred.append(len(myList[8]))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_latred.count(1)/(len(utcHourly_latred)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_latred.count(2)/(len(utcHourly_latred)*1.0)*100)
        



import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in np.arange(3, 8):
    hundredIters = []
    for i in np.arange(0, 40):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in np.arange(3, 8):
    hundredIters2 = []
    for i in np.arange(0, 40):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
trace1 = tuple(percentage1)
trace1Std = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

trace2 = tuple(percentage2)
trace2Std = tuple(std2)
rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Resolution: 1 hour, 2 gps decimal digits (0.69 km^2)')
ax.set_xticks(ind + width)
ax.set_xlabel('Number of Spatio-temporal points')
ax.set_xticklabels(('3','4','5','6', '7'))
ax.set_ylim((60, 100))
ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.savefig('./results/DTU_True/BigExp_1Hours_latlonred.png')
plt.show()

# UTC , lat_red
# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];
percentagesDict = {}

for dp in np.arange(3, 8):
    utcHourly_latred = []
    for i in np.arange(0, 40):
        for j in np.arange(1, 11):
            if(i*10+j > 198):
                continue;
            infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                          +"_DTU_True", "rb" )
            myList = pickle.load(infile)
            
            # 2 is utc 15 minutes, latitude and longitude to 2 decimal digits (0.69  km²)
            idx = 2
            utcHourly_latred.append(len(myList[idx]))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_latred.count(1)/(len(utcHourly_latred)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_latred.count(2)/(len(utcHourly_latred)*1.0)*100)
        



import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in np.arange(3, 8):
    hundredIters = []
    for i in np.arange(0, 40):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in np.arange(3, 8):
    hundredIters2 = []
    for i in np.arange(0, 40):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
trace1 = tuple(percentage1)
trace1Std = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

trace2 = tuple(percentage2)
trace2Std = tuple(std2)
rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Resolution: 15 minutes, 2 gps decimal digits (0.69 km^2)')
ax.set_xticks(ind + width)
ax.set_xlabel('Number of Spatio-temporal points')
ax.set_xticklabels(('3','4','5','6', '7'))
ax.set_ylim((60, 100))
ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.savefig('./results/DTU_True/BigExp_15min_latlonred.png')
plt.show()

# UTC 3 hours , lat_red
# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];
percentagesDict = {}

for dp in np.arange(3, 8):
    utcHourly_latred = []
    for i in np.arange(0, 40):
        for j in np.arange(1, 11):
            if(i*10+j > 198):
                continue;
            infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                          +"_DTU_True", "rb" )
            myList = pickle.load(infile)
            
            # 14 is utc 3 hours, latitude and longitude to 2 decimal digits (0.69  km²)
            idx = 14
            utcHourly_latred.append(len(myList[idx]))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_latred.count(1)/(len(utcHourly_latred)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_latred.count(2)/(len(utcHourly_latred)*1.0)*100)
        



import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in np.arange(3, 8):
    hundredIters = []
    for i in np.arange(0, 40):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in np.arange(3, 8):
    hundredIters2 = []
    for i in np.arange(0, 40):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
trace1 = tuple(percentage1)
trace1Std = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

trace2 = tuple(percentage2)
trace2Std = tuple(std2)
rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Resolution: 3 hours, 2 gps decimal digits (0.69 km^2)')
ax.set_xticks(ind + width)
ax.set_xlabel('Number of Spatio-temporal points')
ax.set_xticklabels(('3','4','5','6', '7'))
ax.set_ylim((60, 100))
ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.savefig('./results/DTU_True/BigExp_3Hours_latlonred.png')
plt.show()

# UTC 15 minutes , lat_red4
# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];
percentagesDict = {}

for dp in np.arange(3, 8):
    utcHourly_latred = []
    for i in np.arange(0, 40):
        for j in np.arange(1, 11):
            if(i*10+j > 198):
                continue;
            infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                          +"_DTU_True", "rb" )
            myList = pickle.load(infile)
            
            # 0 is utc 15 minutes, latitude and longitude to 4 decimal digits (70 sq meters)
            idx = 0
            utcHourly_latred.append(len(myList[idx]))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_latred.count(1)/(len(utcHourly_latred)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_latred.count(2)/(len(utcHourly_latred)*1.0)*100)
        



import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in np.arange(3, 8):
    hundredIters = []
    for i in np.arange(0, 40):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in np.arange(3, 8):
    hundredIters2 = []
    for i in np.arange(0, 40):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
trace1 = tuple(percentage1)
trace1Std = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

trace2 = tuple(percentage2)
trace2Std = tuple(std2)
rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Resolution: 15 mintues, 4 gps decimal digits (70 sq meters)')
ax.set_xticks(ind + width)
ax.set_xlabel('Number of Spatio-temporal points')
ax.set_xticklabels(('3','4','5','6', '7'))
ax.set_ylim((60, 100))
ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.savefig('./results/DTU_True/BigExp_15min_latlonred4.png')
plt.show()

# UTC Hourly , lat_red4
# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];
percentagesDict = {}

for dp in np.arange(3, 8):
    utcHourly_latred = []
    for i in np.arange(0, 40):
        for j in np.arange(1, 11):
            if(i*10+j > 198):
                continue;
            infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                          +"_DTU_True", "rb" )
            myList = pickle.load(infile)
            
            # 6 is utc 1Hour, latitude and longitude to 4 decimal digits (70 sq meters)
            idx = 6
            utcHourly_latred.append(len(myList[idx]))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_latred.count(1)/(len(utcHourly_latred)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_latred.count(2)/(len(utcHourly_latred)*1.0)*100)
        



import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in np.arange(3, 8):
    hundredIters = []
    for i in np.arange(0, 40):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in np.arange(3, 8):
    hundredIters2 = []
    for i in np.arange(0, 40):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
trace1 = tuple(percentage1)
trace1Std = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

trace2 = tuple(percentage2)
trace2Std = tuple(std2)
rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Resolution: 1 Hour, 4 gps decimal digits (70 sq meters)')
ax.set_xticks(ind + width)
ax.set_xlabel('Number of Spatio-temporal points')
ax.set_xticklabels(('3','4','5','6', '7'))
ax.set_ylim((60, 100))
ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.savefig('./results/DTU_True/BigExp_1Hours_latlonred4.png')
plt.show()

# UTC 3 Hours , lat_red4
# timeResolution = ['utc','utcHalfHourly', 'utcHourly','utcBiHourly', 'utc3Hours']
# spatialResolution = [['lat_red4', 'lon_red4'],['lat_red3', 'lon_red3'], ['lat_red', 'lon_red']];
percentagesDict = {}

for dp in np.arange(3, 8):
    utcHourly_latred = []
    for i in np.arange(0, 40):
        for j in np.arange(1, 11):
            if(i*10+j > 198):
                continue;
            infile = open("./results/BigExp/BigExp_"+str(i*10+j)+"_DataPoints_"+str(dp)
                          +"_DTU_True", "rb" )
            myList = pickle.load(infile)
            
            # 12 is utc 3Hours, latitude and longitude to 4 decimal digits (70 sq meters)
            idx = 12
            utcHourly_latred.append(len(myList[idx]))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_latred.count(1)/(len(utcHourly_latred)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_latred.count(2)/(len(utcHourly_latred)*1.0)*100)
        



import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in np.arange(3, 8):
    hundredIters = []
    for i in np.arange(0, 40):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in np.arange(3, 8):
    hundredIters2 = []
    for i in np.arange(0, 40):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 5
trace1 = tuple(percentage1)
trace1Std = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

trace2 = tuple(percentage2)
trace2Std = tuple(std2)
rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Resolution: 3 Hour, 4 gps decimal digits (70 sq meters)')
ax.set_xticks(ind + width)
ax.set_xlabel('Number of Spatio-temporal points')
ax.set_xticklabels(('3','4','5','6', '7'))
ax.set_ylim((60, 100))
ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

#autolabel(rects1)
#autolabel(rects2)

plt.savefig('./results/DTU_True/BigExp_3Hours_latlonred4.png')
plt.show()


# In[ ]:




# In[ ]:




# In[35]:

dataframe


# ## Probability Density plot

# In[1]:

from sensible_raw.loaders import loader
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas
months=["november_2013"]


dataframe = pandas.DataFrame({})
for m in months:
    tmp_dict = {}
    columns, data = loader.load_data("location", m);
    for column, array in zip(columns, data):
        tmp_dict[column] = array
    tmp_dataframe = pandas.DataFrame(tmp_dict)
    dataframe = dataframe.append(tmp_dataframe)


# In[2]:

dict={}
for i in set(dataframe['user']):
    dict[i]=len(dataframe.loc[dataframe['user'] == i])


# In[3]:

l = list(dict.values())

valuesDict = {}

for item in l:
    if item in valuesDict:
        valuesDict[item] +=1
    else:
        valuesDict[item] = 1


# In[ ]:




# In[ ]:




# In[ ]:




# In[4]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
hist, bins = np.histogram(list(valuesDict.keys()))
plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), facecolor='y', alpha=0.75)

plt.xlabel('Number of readings')
plt.ylabel('Percentage of users')
plt.title('Histogram of gps readings per user in one month')
#plt.axis([40, 160, 0, 1])
plt.grid(True)
plt.show()


# Chloropleth of all points

# In[ ]:




# In[5]:

# First load the dataframe:

import pickle

infile = open("./MyDataframe.pkl", "rb" )
dataframe = pickle.load(infile)


# In[6]:


def myFunc(df):
    lat = df[0]
    lon = df[1]
    if str(lat)+'*'+str(lon) in chloroplethLatRed:
        chloroplethLatRed[str(lat)+'*'+str(lon)] +=1
    else:
        chloroplethLatRed[str(lat)+'*'+str(lon)] =1


# In[7]:

chloroplethLatRed = {}
dataframe[['lat_red', 'lon_red']].apply(myFunc, axis=1)


# In[ ]:




# In[25]:

get_ipython().magic('matplotlib inline')
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

 
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
    
def draw_screen_poly( lats, lons, m, color):
    if color=='black':
        print('Black: ')
        print(lats)
        print(lons)
    x, y = m( lons, lats )
    xy = list(zip(x,y))
    poly = Polygon( xy, facecolor=color, alpha=0.6 )
    plt.gca().add_patch(poly)

def get_color(count):
    if count >100000:
        return 'black'
    if count>10000:
        return 'white'
    if count > 1000:
        return 'white'
    if count > 100:
        return 'white'
    if count > 10:
        return 'white'
    return (0.992, 1, 0.6)
    
plt.figure(figsize=(22, 12))

my_map = Basemap(llcrnrlon=7.5,llcrnrlat=54.8,urcrnrlon=13.5,urcrnrlat=57.5,
             resolution='h', projection='tmerc', lat_0 = 39.5, lon_0 = 1)
 

my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color=(0.992, 1, 0.6))
#my_map.drawmapboundary(fill_color='#31baff')
my_map.drawmapscale(12.48, 55.58, 1, 45.5, 10)

red_patch = mpatches.Patch(color='black', label='> 100.000')
blue_patch1 = mpatches.Patch(color='#1b2c2d', label='> 10.000')
blue_patch2 = mpatches.Patch(color='#083c40', label='> 1.000')
blue_patch3 = mpatches.Patch(color='#146266', label='> 100')
blue_patch4 = mpatches.Patch(color='#5bb8bd', label='> 10')

plt.legend(handles=[red_patch, blue_patch1, blue_patch2, blue_patch3, blue_patch4])

import math
for key, item in chloroplethLatRed.items():
    
    coors = key.split('*')
    
    latred1 = math.floor(float(coors[0]) * 10.0) / 10.0
    lonred1 = math.floor(float(coors[1]) * 10.0) / 10.0
    if latred1 > 1 and lonred1 > 1 and latred1 < 70 and lonred1 < 70:
        lats = [latred1, latred1+0.1, latred1+0.1, latred1]
        lons = [lonred1, lonred1, lonred1+0.1, lonred1+0.1]
        if len(lats) == 4 and len(lons) == 4:
            color = get_color(item)
            draw_screen_poly(lats, lons, my_map, color)


#my_map.readshapefile('./Copenhagen-shp/shape/roads', 'comarques')
my_map.drawmeridians(np.arange(7, 14, .1), labels=[False,False,False,True])
my_map.drawparallels(np.arange(54, 58, .1), labels=[False,True,False,False])

plt.title("Chloropleth map with spatial resolution of 0.7 km^2")
plt.show()


# In[ ]:




# In[ ]:




# In[32]:

get_ipython().magic('matplotlib inline')
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

 
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
    
def draw_screen_poly( lats, lons, m, color):
    x, y = m( lons, lats )
    xy = list(zip(x,y))
    poly = Polygon( xy, facecolor=color, alpha=0.6 )
    plt.gca().add_patch(poly)

def get_color(count):
    if count >1000000:
        return 'red'
    if count>100000:
        return 'orange'
    if count > 10000:
        return 'yellow'
    if count > 1000:
        return 'white'
    if count > 100:
        return 'black'
    return (0.992, 1, 0.6)
    
plt.figure(figsize=(22, 12))

my_map = Basemap(llcrnrlon=12.28,llcrnrlat=55.58,urcrnrlon=12.67,urcrnrlat=55.80,
             resolution='h', projection='tmerc', lat_0 = 39.5, lon_0 = 1)
 

my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color=(0.992, 1, 0.6))
my_map.drawmapboundary(fill_color='#31baff')
my_map.drawmapscale(12.48, 55.58, 1, 45.5, 10)

red_patch = mpatches.Patch(color='red', label='> 1.000.000')
blue_patch1 = mpatches.Patch(color='orange', label='> 100.000')
blue_patch2 = mpatches.Patch(color='yellow', label='> 10.000')
blue_patch3 = mpatches.Patch(color='white', label='> 1.000')
blue_patch4 = mpatches.Patch(color='black', label='> 100')

plt.legend(handles=[red_patch, blue_patch1, blue_patch2, blue_patch3, blue_patch4])

import math
for key, item in chloroplethLatRed.items():
    
    coors = key.split('*')
    
    latred1 = math.floor(float(coors[0]) * 100.0) / 100.0
    lonred1 = math.floor(float(coors[1]) * 100.0) / 100.0
    if latred1 > 1 and lonred1 > 1 and latred1 < 70 and lonred1 < 70:
        lats = [latred1, latred1+0.01, latred1+0.01, latred1]
        lons = [lonred1, lonred1, lonred1+0.01, lonred1+0.01]
        if len(lats) == 4 and len(lons) == 4:
            color = get_color(item)
            draw_screen_poly(lats, lons, my_map, color)


my_map.readshapefile('./shapefiles/OSM/Copenhagen-shp/shape/roads', 'comarques')
my_map.drawmeridians(np.arange(7, 14, .01), labels=[False,False,False,True])
my_map.drawparallels(np.arange(54, 58, .01), labels=[False,True,False,False])

plt.title("Chloropleth map with spatial resolution of 10.000 sq. meters")
plt.show()


# In[ ]:

FF0000 <-- red
FF1100
FF2200
FF3300
FF4400
FF5500
FF6600
FF7700
FF8800
FF9900
FFAA00
FFBB00
FFCC00
FFDD00
FFEE00
FFFF00 <-- yellow
EEFF00
DDFF00
CCFF00
BBFF00
AAFF00
99FF00
88FF00
77FF00
66FF00
55FF00
44FF00
33FF00
22FF00
11FF00
00FF00


# In[ ]:

my_map = Basemap(llcrnrlon=12.502,llcrnrlat=55.778,urcrnrlon=12.535,urcrnrlat=55.795,
             resolution='h', projection='tmerc', lat_0 = 39.5, lon_0 = 1)


# In[17]:

get_ipython().magic('matplotlib inline')
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

 
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
    
def draw_screen_poly( lats, lons, m, color):
    x, y = m( lons, lats )
    xy = list(zip(x,y))
    poly = Polygon( xy, facecolor=color, alpha=1 )
    plt.gca().add_patch(poly)

def get_color(count):
    if count >1000000:
        return '#FF0000' #red
    if count>500000:
        return '#FF5500' #
    if count > 100000:
        return '#FFAA00' #
    if count > 50000:
        return '#FFFF00' #yellow
    if count > 10000:
        return '#AAFF00'
    if count > 1000:
        return '#55FF00'
    return (0.992, 1, 0.6)
    
plt.figure(figsize=(22, 12))

my_map = Basemap(llcrnrlon=12.472,llcrnrlat=55.758,urcrnrlon=12.555,urcrnrlat=55.805,
             resolution='h', projection='tmerc', lat_0 = 39.5, lon_0 = 1)
 

my_map.drawcoastlines()
my_map.drawcountries()
#my_map.fillcontinents(color=(0.992, 1, 0.6))
#my_map.drawmapboundary(fill_color='#31baff')
#my_map.drawmapscale(12.48, 55.58, 1, 45.5, 10)

red_patch = mpatches.Patch(color='#FF0000', label='> 1.000.000')
blue_patch1 = mpatches.Patch(color='#FF5500', label='> 500.000')
blue_patch2 = mpatches.Patch(color='#FFAA00', label='> 100.000')
blue_patch3 = mpatches.Patch(color='#FFFF00', label='> 50.000')
blue_patch4 = mpatches.Patch(color='#AAFF00', label='> 10.000')
blue_patch5 = mpatches.Patch(color='#55FF00', label='> 1.000')

plt.legend(handles=[red_patch, blue_patch1, blue_patch2, blue_patch3, blue_patch4, blue_patch5])

import math
for key, item in chloroplethLatRed.items():
    
    coors = key.split('*')
    
    latred1 = math.floor(float(coors[0]) * 100.0) / 100.0
    lonred1 = math.floor(float(coors[1]) * 100.0) / 100.0
    if latred1 > 1 and lonred1 > 1 and latred1 < 70 and lonred1 < 70:
        lats = [latred1, latred1+0.01, latred1+0.01, latred1]
        lons = [lonred1, lonred1, lonred1+0.01, lonred1+0.01]
        if len(lats) == 4 and len(lons) == 4:
            color = get_color(item)
            draw_screen_poly(lats, lons, my_map, color)


my_map.readshapefile('./shapefiles/OSM/Copenhagen-shp/shape/roads', 'comarques')
my_map.drawmeridians(np.arange(7, 14, .01), labels=[False,False,False,True])
my_map.drawparallels(np.arange(54, 58, .01), labels=[False,True,False,False])

plt.title("Chloropleth map with spatial resolution of 10.000 sq. meters")
plt.show()


# In[ ]:




# # Plot a user's points on a map.

# In[16]:

# First get the data:

from sensible_raw.loaders import loader
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas
months=["november_2013"]


dataframe = pandas.DataFrame({})
for m in months:
    tmp_dict = {}
    columns, data = loader.load_data("location", m);
    for column, array in zip(columns, data):
        tmp_dict[column] = array
    tmp_dataframe = pandas.DataFrame(tmp_dict)
    dataframe = dataframe.append(tmp_dataframe)


# In[ ]:




# In[28]:

from random import randint

user = 95 #rand int

udf = dataframe.loc[dataframe['user'] == user]

if(len(udf) < 1000):
    print("Not enough points!")


# In[72]:

get_ipython().magic('matplotlib inline')
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
    
    
plt.figure(figsize=(22, 12))

my_map = Basemap(llcrnrlon=12.28,llcrnrlat=55.58,urcrnrlon=12.67,urcrnrlat=55.80,
             resolution='h', projection='tmerc', lat_0 = 39.5, lon_0 = 1)
 
def plotPoint(lat, lon, timestamp, m):
    x,y = m(float(lon), float(lat))

    hour = datetime.datetime.fromtimestamp(int(timestamp/1000.0)).hour
    color = 'black'
    
    if(hour >= 0 and hour <= 6):
        color = 'red'
    if(hour > 6 and hour <= 12):
        color = 'yellow'
    if(hour > 12 and hour <= 18):
        color = 'green'
    if(hour > 18 and hour <= 23):
        color = 'blue'

    if color=='black':
        print(hour)
        print(timestamp)
        print('-----------------------')
    m.plot(x, y, color=color, marker='o', markersize=10)
    return
    
my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color=(0.992, 1, 0.6))
my_map.drawmapboundary(fill_color='#31baff')
my_map.drawmapscale(12.48, 55.58, 1, 45.5, 10)

red_patch = mpatches.Patch(color='red', label='00h00-06h00')
yellow_patch = mpatches.Patch(color='yellow', label='06h00-12h00')
green_patch = mpatches.Patch(color='green', label='12h00-18h00')
blue_patch = mpatches.Patch(color='blue', label='18h00-24h00')

plt.legend(handles=[red_patch, yellow_patch, green_patch, blue_patch])

for lat, lon, timestamp in udf[['lat', 'lon', 'timestamp']].values:
    plotPoint(lat, lon, timestamp, my_map);

my_map.readshapefile('./shapefiles/OSM/Copenhagen-shp/shape/roads', 'comarques')
my_map.drawmeridians(np.arange(7, 14, .01), labels=[False,False,False,True])
my_map.drawparallels(np.arange(54, 58, .01), labels=[False,True,False,False])

plt.show()


# In[ ]:




# In[ ]:

#timeResolution = ['utc3Hours', 'timestampDaily']
#spatialResolution = [['lat_red', 'lon_red'], ['lat_red1', 'lon_red1']];    


# In[12]:

import pickle

utcHourly_latred = []

infile = open("./DynamicExp/DynamicExp32_DataPoints_001_DTU_True", "rb")
myList = pickle.load(infile)

utcHourly_latred.append(len(myList[0]))


# In[ ]:




# In[13]:

import numpy as np
import pickle
percentagesDict = {}

#for dp in ['001', '010', '100', '002', '020', '200', '011', '101', '110', '111', '003', '030', '300', '022', '202', '220', '222', '330']:
for dp in ['001', '100', '010', '002', '020', '200', '011', '101', '110', '111', '003', '030', '300']:
    utcHourly_latred = []
    for i in np.arange(0, 100):
        for j in np.arange(0, 5):
            infile = open("./DynamicExp/DynamicExp"+str(i*5+j+1)+"_DataPoints_"+str(dp)
                          +"_DTU_True", "rb" )
            myList = pickle.load(infile)
            utcHourly_latred.append(len(myList[0]))

        percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (utcHourly_latred.count(1)/(len(utcHourly_latred)*1.0)*100)
        percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(utcHourly_latred.count(2)/(len(utcHourly_latred)*1.0)*100)


import numpy
percentage1 = []
std1= []
percentageAverages =[]
for j in ['001', '100', '010', '002', '020', '200', '011', '101', '110', '111', '003', '030', '300']:
    hundredIters = []
    for i in np.arange(0, 100):
        hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage1.append(sum(hundredIters) / float(len(hundredIters)))
    std1.append(numpy.std(hundredIters))
    
percentage2 = []
std2= []
percentageAverages2 =[]
for j in ['001', '100', '010', '002', '020', '200', '011', '101', '110', '111', '003', '030', '300']:
    hundredIters2 = []
    for i in np.arange(0, 100):
        hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
    percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
    std2.append(numpy.std(hundredIters2))

#!/usr/bin/env python
# a bar plot with errorbars
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

N = 13
trace1 = tuple(percentage1)
trace1Std = tuple(std1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots(figsize=(30, 10))

rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

trace2 = tuple(percentage2)
trace2Std = tuple(std2)
rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentages')
ax.set_title('Types of data points.')
ax.set_xticks(ind + width)
ax.set_xlabel('Number of Spatio-temporal points (X-Y-Z)')


ax.set_xticklabels(('0-0-1', '0-1-0', '1-0-0', '0-0-2', '0-2-0', '2-0-0', '0-1-1', '1-0-1', '1-1-0', '1-1-1', '0-0-3', '0-3-0', '3-0-0'))
ax.set_ylim((0, 100))

black_dot, = plt.plot(0, color='black', marker='o', markersize=2)

ax.legend((rects1[0], rects2[0], black_dot, black_dot, black_dot), ('# Traces = 1', '# Traces <= 2', 'X: 3 hours, 2 gps decimal', 'Y: Daily, 2 gps decimal', 'Z: Daily, 1 gps decimal'), borderaxespad=0., loc=4)


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

print(rects1[0])

plt.savefig('./results/DynamicExp/NewDynamic.png')
plt.show()


# In[ ]:




# In[ ]:




# In[2]:

import pickle

infile = open("./MyDataframe2.pkl", "rb" )
dataframe = pickle.load(infile)


# In[21]:

udf = dataframe.loc[113]
if(len(udf) < 1000):
    print("Not enough points!")


# In[7]:

def calculateHome(udf):
    lat_red3Counts = {}
    for lat_red3, lon_red3, timestamp in udf[['lat_red3', 'lon_red3', 'timestamp']].values:
        hour = datetime.datetime.fromtimestamp(int(timestamp/1000.0)).hour
        if (hour >= 0 and hour <= 8) or (hour >= 22):
            if str(lat_red3) + '*' + str(lon_red3) in lat_red3Counts:
                lat_red3Counts[str(lat_red3) + '*' + str(lon_red3)] +=1
            else:
                lat_red3Counts[str(lat_red3) + '*' + str(lon_red3)] = 1
    import operator
    sorted_mostCommonLocations = sorted(lat_red3Counts.items(), key=operator.itemgetter(1))
    if not sorted_mostCommonLocations:
        return []
    coors = sorted_mostCommonLocations[-1:][0][0].split('*')
    
    return coors


# In[12]:

import datetime
import numpy as np
infile = open("./homes", "rb" )
homes = pickle.load(infile)

for idx in np.arange(730, 1000):
    udf = dataframe.loc[idx]
    home = calculateHome(udf)
    if home:
        homes[idx] = home
    else:
        print("Could  not add user : " + str(idx))




# In[14]:

import pickle
pickle.dump(homes, open("homes", "wb"))


# In[74]:

def calculateWork(udf, home):
    
    lat_red3Counts = {}
    for lat_red3, lon_red3, timestamp in udf[['lat_red3', 'lon_red3', 'timestamp']].values:
        hour = datetime.datetime.fromtimestamp(int(timestamp/1000.0)).hour
        if (hour >= 9 and hour <= 17):
            if str(lat_red3) + '*' + str(lon_red3) in lat_red3Counts:
                lat_red3Counts[str(lat_red3) + '*' + str(lon_red3)] +=1
            else:
                lat_red3Counts[str(lat_red3) + '*' + str(lon_red3)] = 1
    
    import operator
    i=1
    while i < 100:
        sorted_mostCommonLocations = sorted(lat_red3Counts.items(), key=operator.itemgetter(1))
        coors = sorted_mostCommonLocations[-i:][0][0].split('*')
        print(coors)
        print(abs(float(coors[0]) - float(home[0])) > 0.01) or (abs(float(coors[1]) - float(home[1])) > 0.01)
        if (abs(float(coors[0]) - float(home[0])) > 0.01) or (abs(float(coors[1]) - float(home[1])) > 0.01):
            break;
        i+=1
    return coors


# In[ ]:




# In[23]:

print(sorted_mostCommonLocations[-1:][0][0].split('*'))
print(sorted_mostCommonLocations[-2:][0][0].split('*'))
print(sorted_mostCommonLocations[-3:][0][0].split('*'))


# In[34]:

infile = open("./homes", "rb" )
homes = pickle.load(infile)
print(homes[113])
print(coors)
coors[0] = '55.750'


# In[59]:

print("Hello")


# In[76]:

import datetime
works = {}
for idx in np.arange(0, 846):
    udf = dataframe.loc[idx]
    try:
        work = calculateWork(udf, homes[idx])
    except KeyError:
        work = calculateWork(udf, ['0', '0'])
    works[idx] = work


# In[77]:


import pickle
#pickle.dump(homes, open("homes", "wb"))
pickle.dump(works, open("works", "wb"))


# In[64]:

u32df = dataframe.loc[32]


# In[66]:

calculateHome(u32df)


# In[75]:

calculateWork(u32df, ['55.66', '12.339'])


# In[24]:

get_ipython().magic('matplotlib inline')
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import datetime

plt.figure(figsize=(22, 12))

#dtu
#my_map = Basemap(llcrnrlon=12.472,llcrnrlat=55.758,urcrnrlon=12.555,urcrnrlat=55.805,
#             resolution='h', projection='tmerc', lat_0 = 39.5, lon_0 = 1)

#cph
my_map = Basemap(llcrnrlon=12.28,llcrnrlat=55.58,urcrnrlon=12.67,urcrnrlat=55.80,
             resolution='h', projection='tmerc', lat_0 = 39.5, lon_0 = 1)
 
def plotPoint(lat, lon, timestamp, m):
    x,y = m(float(lon), float(lat))

    hour = datetime.datetime.fromtimestamp(int(timestamp/1000.0)).hour
    color = 'black'
    
    if(hour >= 0 and hour <= 8) or (hour >= 22):
        color = 'red'
        m.plot(x, y, color=color, marker='o', markersize=5)
    if(hour > 6 and hour <= 12):
        color = 'purple'
        m.plot(x, y, color=color, marker='o', markersize=5)
    if(hour > 12 and hour <= 18):
        color = 'green'
        m.plot(x, y, color=color, marker='o', markersize=5)
    if(hour > 18 and hour <= 23):
        color = 'blue'
        m.plot(x, y, color=color, marker='o', markersize=5)

    return
    
my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color=(0.992, 1, 0.6))
my_map.drawmapboundary(fill_color='#31baff')
#my_map.drawmapscale(12.48, 55.58, 1, 45.5, 10)

red_patch = mpatches.Patch(color='red', label='22h00-08h00')
yellow_patch = mpatches.Patch(color='purple', label='06h00-12h00')
green_patch = mpatches.Patch(color='green', label='12h00-18h00')
blue_patch = mpatches.Patch(color='blue', label='18h00-24h00')

plt.legend(handles=[red_patch, yellow_patch, green_patch, blue_patch])

for lat, lon, timestamp in udf[['lat', 'lon', 'timestamp']].values:
    plotPoint(lat, lon, timestamp, my_map);

my_map.readshapefile('./shapefiles/OSM/Copenhagen-shp/shape/roads', 'comarques')
my_map.drawmeridians(np.arange(7, 14, .01), labels=[True,False,True,True])
my_map.drawparallels(np.arange(54, 58, .01), labels=[False,True,False,False])

plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[15]:

import pickle
import numpy as np

indexes = np.arange(0, 5)
label = ['15 minutes', '30 minutes', '1 hour', '2 hours', '3 hours']
for timeResolution, timeResolutionLabel in zip(indexes, label):
    percentagesDict = {}

    for dp in ['3', '6', '9', '12', '15', '18']:
        entries = []
        for i in np.arange(0, 100):
            for j in np.arange(0, 5):
                try: 
                    infile = open("./BigExp/Label_Home_"+str(i*5+j+1)+"_DataPoints_"+str(dp)
                              +"_DTU_True", "rb" )
                except FileNotFoundError:
                    continue;
                    
                myList = pickle.load(infile)
                entries.append(len(myList[timeResolution]))

            if(len(entries) == 0):
                print("check number: " + str(i*5+j+1)+ " . dp: "+str(dp))
            percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (entries.count(1)/(len(entries)*1.0)*100)
            percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(entries.count(2)/(len(entries)*1.0)*100)


    import numpy
    percentage1 = []
    std1= []
    percentageAverages =[]
    for j in ['3', '6', '9', '12', '15', '18']:
        hundredIters = []
        for i in np.arange(0, 100):
            hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
        percentage1.append(sum(hundredIters) / float(len(hundredIters)))
        std1.append(numpy.std(hundredIters))

    percentage2 = []
    std2= []
    percentageAverages2 =[]
    for j in ['3', '6', '9', '12', '15', '18']:
        hundredIters2 = []
        for i in np.arange(0, 100):
            hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
        percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
        std2.append(numpy.std(hundredIters2))

    #!/usr/bin/env python
    # a bar plot with errorbars
    get_ipython().magic('matplotlib inline')
    import numpy as np
    import matplotlib.pyplot as plt

    N = 6
    trace1 = tuple(percentage1)
    trace1Std = tuple(std1)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots(figsize=(30, 10))

    rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

    trace2 = tuple(percentage2)
    trace2Std = tuple(std2)
    rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Percentages')
    ax.set_title('Spatial Resolution: Home, Not Home. Time Resolution: '+ timeResolutionLabel+'.')
    ax.set_xticks(ind + width)
    ax.set_xlabel('Number of Spatio-temporal points')


    ax.set_xticklabels(('3', '6', '9', '12', '15', '18'))
    ax.set_ylim((0, 100))

    black_dot, = plt.plot(0, color='black', marker='o', markersize=2)

    ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    #autolabel(rects1)
    #autolabel(rects2)

    print(rects1[0])

    plt.savefig('./results/LabelExp/label_Home' + timeResolutionLabel+ '.png')
    plt.show()


# In[ ]:




# In[ ]:

udf = dataframe.loc[123]


# In[20]:

udf.loc[udf['timestamp'] == 1386269100000]


# In[19]:

udf


# In[48]:

import pandas


# In[52]:

df = pandas.DataFrame(1378380600000, 12341, columns=['timestamp', 'home'])


# In[59]:

#### import pickle

infile = open("./MyDataframe4.pkl", "rb" )
dataframe = pickle.load(infile)


# In[60]:

udf = dataframe.loc[123]
df = udf.loc[udf['timestamp'] == 1386269100000]

df1 = df[['timestamp','AtHome']]


# In[61]:

df1


# In[62]:

df2 = df[['timestamp','lat', 'lon']]


# In[63]:


df2


# In[10]:




# In[3]:

import pickle
import numpy as np

indexes = np.arange(0, 5)
label = ['30 minutes']
for timeResolution, timeResolutionLabel in zip(indexes, label):
    percentagesDict = {}

    for dp in ['3', '6', '9', '12', '15', '18']:
        entries = []
        for i in np.arange(0, 100):
            for j in np.arange(0, 5):
                try: 
                    infile = open("./LabelShuffleExperimentMatrix/30minutes/LabelShuffle_Home_"+str(i*5+j+1)+"_DataPoints_"+str(dp)
                              +"_DTU_True", "rb" )
                except FileNotFoundError:
                    continue;
                    
                myList = pickle.load(infile)
                entries.append(len(myList[timeResolution]))

            if(len(entries) == 0):
                print("check number: " + str(i*5+j+1)+ " . dp: "+str(dp))
            percentagesDict['percentage_1_dp_' +str(dp) + "_iter_" + str(i)]= (entries.count(1)/(len(entries)*1.0)*100)
            percentagesDict['percentage_2_dp_' +str(dp) + "_iter_" + str(i)] =(entries.count(2)/(len(entries)*1.0)*100)


    import numpy
    percentage1 = []
    std1= []
    percentageAverages =[]
    for j in ['3', '6', '9', '12', '15', '18']:
        hundredIters = []
        for i in np.arange(0, 100):
            hundredIters.append(percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
        percentage1.append(sum(hundredIters) / float(len(hundredIters)))
        std1.append(numpy.std(hundredIters))

    percentage2 = []
    std2= []
    percentageAverages2 =[]
    for j in ['3', '6', '9', '12', '15', '18']:
        hundredIters2 = []
        for i in np.arange(0, 100):
            hundredIters2.append(percentagesDict['percentage_2_dp_'+str(j)+'_iter_' + str(i)] + percentagesDict['percentage_1_dp_'+str(j)+'_iter_' + str(i)])
        percentage2.append(sum(hundredIters2) / float(len(hundredIters2)))
        std2.append(numpy.std(hundredIters2))

    #!/usr/bin/env python
    # a bar plot with errorbars
    get_ipython().magic('matplotlib inline')
    import numpy as np
    import matplotlib.pyplot as plt

    N = 6
    trace1 = tuple(percentage1)
    trace1Std = tuple(std1)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots(figsize=(30, 10))

    rects1 = ax.bar(ind, trace1, width, color='r',yerr=trace1Std)

    trace2 = tuple(percentage2)
    trace2Std = tuple(std2)
    rects2 = ax.bar(ind + width, trace2, width, color='y',yerr=trace2Std)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Percentages')
    ax.set_title('Spatial Resolution: Home, Not Home. Time Resolution: '+ timeResolutionLabel+'.')
    ax.set_xticks(ind + width)
    ax.set_xlabel('Number of Spatio-temporal points')


    ax.set_xticklabels(('3', '6', '9', '12', '15', '18'))
    ax.set_ylim((0, 100))

    black_dot, = plt.plot(0, color='black', marker='o', markersize=2)

    ax.legend((rects1[0], rects2[0]), ('# Traces = 1', '# Traces <= 2'), borderaxespad=0., loc=4)


    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    #autolabel(rects1)
    #autolabel(rects2)

    print(rects1[0])

    plt.savefig('./results/ShuffleMatrixLabel/LabelShuffle_Home' + timeResolutionLabel+ '.png')
    plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[1]:

import pickle
import numpy

infile = open("./matrix.pkl", "rb" )
dataframe = pickle.load(infile)


# In[94]:

a = dataframe.reset_index().drop('user', 1).values


# In[31]:

b = dataframe.columns.values


# In[21]:

numpy.concatenate((a, b), axis=0)


# In[36]:

a.shape


# In[95]:

numpy.random.shuffle(a)


# In[37]:

a = np.random.shuffle(a)


# In[96]:

a


# In[97]:

a.shape


# In[104]:

c = numpy.insert(a, 0, b, axis=0)


# In[45]:

c.shape


# In[126]:

a


# In[115]:

import datetime
datetime.datetime.utcfromtimestamp(1.40952780e+09)


# In[58]:

1377993600000000000/1e3


# In[106]:

c[0] = numpy.divide(c[0],1000000000)


# In[107]:

c.shape


# In[121]:

c.tofile('./homelabels')


# In[118]:

bla = numpy.fromfile('./homelabels.npy')


# In[120]:

bla.shape


# In[84]:

numpy.savez('./homelabels', c)


# In[99]:

arr = numpy.arange(9).reshape((3, 3))


# In[131]:

a


# In[101]:

import numpy as np
np.random.shuffle(arr)


# In[108]:

purty


# In[132]:

numpy.savetxt('./myFile', a)


# In[134]:

e = numpy.loadtxt('./myFile')


# In[135]:

e.shape


# In[130]:

e


# In[125]:

import datetime
datetime.datetime.utcfromtimestamp(1.40952780e+09)


# In[ ]:



