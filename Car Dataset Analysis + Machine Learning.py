#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:32:19 2020

@author: aleksanderbielinski
"""

import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline


df1=pd.read_csv(r'/Users/aleksanderbielinski/Desktop/CarPrice1.csv')

#Displaying columns names

for col in df1:
    print(col)
    
#Determining data types

df1.dtypes

#Viewing the object values

obj_df1=df1.select_dtypes(include=['object']).copy()

for col in obj_df1:
    print(col)

#Transfering cylinder number we can simpy use find and replace approach as chategorical values are just numbers in this case

#determining the categories for cylinders

obj_df1['cylindernumber'].value_counts()

#changing the values

new_nums={'cylindernumber': {"four": 4, "six": 6, "five": 5, "eight": 8, "two": 2, "three": 3, "twelve": 12}}

#swapping the columns for the the newly created dictionary

df1.replace(new_nums, inplace=True)

df1['cylindernumber']




#getting heatmap 

#heatmap1=sn.heatmap(df1.corr(),center=0,cmap='BrBG', annot=True)

#heatmap1=sn.heatmap(df1.corr(),center=0,cmap='BrBG',label='Correlation')
fig, ax = plt.subplots(figsize=(12,12))
sn.heatmap(df1.corr(),center=0,cmap='BrBG',ax=ax, annot=True, mask=mask)

ax.set_title('Numerical Values Correlation Matrix')
plt.show()


#Determining which make is the most expensive

makeprice=df1.groupby(['CarMake']).price.agg(['mean'])

#bar plot of car prices

makeprice.plot.bar(color='darkgreen', figsize=(12,8))
plt.xlabel('Car Make', size='xx-large')
plt.ylabel('Price in $',size='xx-large')
plt.title("Average Car Prices", size='xx-large')
plt.show()

#scatterplot of horsepower vs price with enginesize colorbar
plt.figure(figsize=(9,7))
plt.scatter(df1['horsepower'], df1['price'], c=df1['enginesize'], marker = 'o', alpha=0.7)
plt.colorbar(label='Engine Size (litre x 100)')
plt.title('Scatterplot of Horsepower vs Price in relation to the Engine Size', size='x-large')
plt.xlabel('Horsepower', size='x-large')
plt.ylabel('Price in $', size='x-large')

#fule consumption in the city

mpgcity=df1.groupby(['CarMake']).citympg.agg(['mean'])

mpghighway=df1.groupby(['CarMake']).highwaympg.agg(['mean'])

#plotting two charts

plt.subplot(211)
plt.plot(mpgcity,lw=1.5,label='Average City MPG')
plt.xlabel(False)
plt.plot(mpgcity,'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('City MPG')
#plt.xticks(rotation=90)
plt.title('City MPG vs Highway MPG across different car makes')
plt.subplot(212)
plt.plot(mpghighway, 'g', lw=1.5, label='Average Highway MPG')
plt.plot(mpghighway,'ro')
plt.xticks(rotation=90)
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('Highway MPG')

#ask how to delete x label from the first plot 

# in one chart
plt.figure(figsize=(10,7))
plt.plot(mpgcity,lw=1.5,label='Average City MPG')
plt.plot(mpghighway, 'purple', lw=1.5, label='Average Highway MPG')
plt.plot(mpgcity, 'go')
plt.plot(mpghighway, 'ro')
plt.grid(True)
plt.xticks(rotation=90)
plt.legend(loc=0) # specifies where to place the legend3
plt.axis('tight')
plt.xlabel('Car Make', size='x-large')
plt.ylabel('MPG', size='x-large')
plt.title('MPG City vs Highway', size='x-large')


#checking whihc fueltype can give us bet mileage 


#fuelsystem 

fuelsyscity=df1.groupby(['fuelsystem']).citympg.agg(['mean'])

fuelsyshighway=df1.groupby(['fuelsystem']).highwaympg.agg(['mean'])

#creating a new column front surface 

for col in df2:
    print(col)
    
#creating an separate dataframe 

df2=df1

#checking for ranges

#for width

maxValueswidth= df2['carwidth'].max(axis=0)

#for height

maxValuesheight= df2['carheight'].max(axis=0)

#adding a column to a dataframe

def frontsurface(a,b):
    return a*b

Approx_Car_Surface=[]

carwidth=df2['carwidth']
carheight=df2['carheight']

for i in range (len(carwidth)):
    Approx_Car_Surface.append(frontsurface(carwidth[i], carheight[i]))
    
#adding new column to the dataframe 2

df2['Approx_Front_Surface']=Approx_Car_Surface

#creating a MPG average across all makes

def meanmpg (c,h):
    return (c+h)/2

city=df2['citympg']
highway=df2['highwaympg']

Average_MPG=[]
for j in range (len(highway)):
    Average_MPG.append(meanmpg(city[j], highway[j]))
    
df2['Average_MPG']=Average_MPG

#Creating a Sccaterplot of Front Surface vs Average MPG in relation to kerb weight
plt.figure(figsize=(10,8))
plt.scatter(df2['Approx_Front_Surface'], df2['Average_MPG'], c=df2['curbweight'], marker = 'o',alpha=0.7)
plt.colorbar(label='Total Car Weight in lbs ')
plt.title('Sccaterplot of Front Surface vs Average MPG in relation to kerb weight', size='x-large')
plt.xlabel('Approximate Front Surface in inches^2', size='x-large' )
plt.ylabel('Average MPG', size='x-large')

#Creating a Scatterplot of Front Surface vs Car Price in relation to kerb weight
plt.figure(figsize=(10,8))
plt.scatter(df2['Approx_Front_Surface'], df2['price'], c=df2['curbweight'], marker = 'o',alpha=0.7)
plt.colorbar(label='Total Car Weight in lbs ')
plt.title('Scatterplot of Front Surface vs Car Price in relation to kerb weight',size='x-large')
plt.xlabel('Approximate Front Surface in inches^2', size='x-large')
plt.ylabel('Car Price in $', size='x-large')


#Determining whihc fuel type and fuelsystem gives best MPG and Price

#engine type and mpg

enginempg=df2.groupby(['enginetype']).Average_MPG.agg(['mean'])

enginecounts=df2.groupby(['enginetype']).agg(['count'])

plt.plot(enginempg)

#engine type and price

engineprice=df2.groupby(['enginetype']).price.agg(['mean'])

plt.plot(engineprice)


#Fueltype and mpg

fuelsys=df2.groupby(['fuelsystem']).Average_MPG.agg(['mean'])

fuelscount=df2.groupby(['fuelsystem']).agg(['count'])
plt.plot(fuelsys)


#Fuelsys and price

fuelsysp=df2.groupby(['fuelsystem']).price.agg(['mean'])

plt.plot(fuelsysp)


#arrays for the engine price

m1=(18116.416667, 31400.500000, 14627.583333, 11574.048426, 13738.600000, 25098.384615, 13020.000000)
label=np.array(['dohc','dohcv','l','ohc','ohcf','ohcv','rotor'])


color1=['saddlebrown','sandybrown','peru','peachpuff','olivedrab','darkgreen','seagreen']
fig, ax1 = plt.subplots() #create subplots
ax1.bar(label, m1, 0.5, lw=1.5, label='Engine type price',color=color1)
ax1.legend(loc=2)
ax1.axis('tight')
plt.xlabel('Engine Type')
plt.grid(True)
ax1.set_ylabel('Average Price')
plt.title('Engine Type Price and MPG')
ax2 = ax1.twinx()#create 2nd subplot where x-axis is shared4
ax2.plot(enginempg, 'g', lw=1.5, label='Engine type MPG')
ax2.plot(enginempg, 'ro')
ax2.legend(loc=0)
ax2.set_ylabel('Average MPG')


#ALternative

#installing plotly 

conda install -c plotly plotly

import plotly.graph_objects as go

#Data for the fuel system

#MPG

fuelsys

#Price

fuelsysp

#Fuelsystem Prices

f1=[7555.545455, 7478.151515, 12145.000000, 15838.150000, 12964.000000, 17754.602840, 10990.444444, 11048.000000]

#Fuelsystem average MPG
f2=[34.000000, 33.083333, 20.000000, 32.525000, 21.500000, 23.420213, 24.388889, 26.500000]

#Labels for the fuel system

labelf=['1bb1','2bbl','4bbl','idi','mfi','mpfi','spdi','spfi']

#Plotting bar chart (SEE COMENTARY!!!)

fig = go.Figure(
    data=[
        go.Bar(name='Fuel System Average Price', x=labelf, y=f1, yaxis='y', offsetgroup=1),
        go.Bar(name='Fuel System Average MPG',x=labelf, y=f2, yaxis='y2', offsetgroup=2)
        ],
        layout={
            'yaxis':{'title': 'Price'},
            'yaxis2':{'title': 'MPG', 'overlaying': 'y', 'side': 'right'}
            }
        )

fig.update_layout(barmode='group')
fig.show()


#Data for engine type

#MPG

enginempg

#Price

engineprice

#engine type prices

e1=[18116.416667, 31400.500000, 14627.583333, 11574.048426, 13738.600000, 25098.384615, 13020.000000]

#Engine Type Average MPG

e2=[22.416667, 22.500000, 26.666667, 29.685811, 27.033333, 19.000000, 19.875000]

#labels for engine type 

labele=['dohc','dohcv','l','ohc','ohcf','ohcv','rotor']

#Plotting a bar chart (SEE COMENTARY!!!)
fig = go.Figure(
    data=[
        go.Bar(name='Engine Type Average Price', x=labele, y=e1, yaxis='y', offsetgroup=1),
        go.Bar(name='Engine Type Average MPG',x=labele, y=e2, yaxis='y2', offsetgroup=2)
        ],
        layout={
            'yaxis':{'title': 'Price'},
            'yaxis2':{'title': 'MPG', 'overlaying': 'y', 'side': 'right'}
            }
        )

fig.update_layout(barmode='group')
fig.show()


#Type of Fuel vs Average MPG and Price

#Average MPG for each fuel type

typef=df2.groupby(["fueltype"]).Average_MPG.agg(['mean'])

#Average Price for each fuel type

typep=df2.groupby(["fueltype"]).price.agg(['mean'])

#displaying the values for fueltype

#for the Price

typep

#For the MPG

typef

#fuel type prices

f1=[15838.1500, 12999.7982]

#fuel type mpg

f2=[32.525000, 27.494595]

#labels for the fuel type

labelf=['diesel','gas']

#Plotting a bar chart (SEE COMENTARY!!!)

fig = go.Figure(
    data=[
        go.Bar(name='Fuel Type Average Price', x=labelf, y=f1, yaxis='y', offsetgroup=1),
        go.Bar(name='Fuel Type Average MPG',x=labelf, y=f2, yaxis='y2', offsetgroup=2)
        ],
        layout={
            'yaxis':{'title': 'Price'},
            'yaxis2':{'title': 'MPG', 'overlaying': 'y', 'side': 'right'}
            }
        )

fig.update_layout(barmode='group')
fig.show()

#COMMENTARY// The above operations had to be performed in the jupyter nootebook as I could not display it in the Spyder IDE


#different way


from io import StringIO


engine2 = StringIO(""" MPG Price
diesel    32.525000     15838.1500
gas       27.494595     12999.7982""")

fueldf2=pd.read_csv(engine2, index_col=0, delimiter= ' ', skipinitialspace=True)

xticks=['diesel', 'gas']

fueldf2.plot(kind='bar', secondary_y='Price', ylim=(0,50), color=('purple','teal'), legend=True, figsize=(10,8)).set_xticklabels(xticks, rotation=0, size='x-large')
plt.title('Fuel Type vs MPG and Average Price', size='xx-large')
ax1,ax2=plt.gcf().get_axes()
ax1.set_ylabel('Average MPG', size='x-large')
ax2.set_ylabel('Average Price in $', size='x-large')
ax1.set_xlabel('Fuel Type', size='x-large')

#Unsupervised 

from sklearn import cluster

from sklearn import metrics


#Price distribution 

df1.hist('price',color='teal', figsize=(10,8));
plt.title('Distribution of Prices', size='x-large');
plt.xlabel('Price in $', size='x-large')



#in case of only three cathegories , remember to reload initial df1 first!!!

def cathegory3 (a):
    if 0<a<=10000:
        return 'low'
    elif 10000<a<=25000:
        return 'mid'
    elif 25000<a<=100000:
        return 'high'

#new dataframe

df3=df1

#

price=df3['price']

plist3=[]

for i in range (len(price)):
    plist3.append(cathegory3(price[i]))
    
df3['Cathegory']=plist

#numerical values

num_df3=df3.select_dtypes(include=['int64','float64']).copy()

#dropping unessecary values

num_df3.drop(['price','car_ID','symboling'],axis=1,inplace=True)

#adding new cathegories

num_df3['Cathegory']=plist3

#defining new X and Y

X1=num_df3.iloc[:, :-1].values

Y1=num_df3.loc[:,'Cathegory'].values

#scale X

scaled_data1=scale(X1)

#Label encoder on Y1

Y3=LabelEncoder().fit_transform(Y1)

#KMeans for 3 cathegories

sil3=[]
com3=[]
hom3=[]
for k in range(2, 13):
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(scaled_data)
    print(k)
    print("silhouette_score = ", metrics.silhouette_score(scaled_data1, kmeans.labels_))
    print("completeness_score = ", metrics.completeness_score(Y3, kmeans.labels_))
    print("homogeneity_score = ", metrics.homogeneity_score(Y3, kmeans.labels_))
    sil3.append(metrics.silhouette_score(scaled_data, kmeans.labels_))
    com3.append(metrics.completeness_score(Y3, kmeans.labels_))
    hom3.append( metrics.homogeneity_score(Y3, kmeans.labels_))
    
    
#new dataframe

scores_KM3=pd.DataFrame({'No. of Clusters': range(2,13), 'silhouette':sil3, 'completeness':com3, 'homogenity':hom3})

#plotting a scores 

plt.plot('No. of Clusters','silhouette',data=scores_KM3, color='teal', linewidth=1.2)
plt.plot('No. of Clusters', 'completeness', data=scores_KM3, color='darkgreen', linewidth=1.2)
plt.plot('No. of Clusters', 'homogenity', data=scores_KM3, color='purple', linewidth=1.2)
plt.legend()
plt.title('Results for different no. of clusters')

#works better with 5 cathegories

#checking for 2 categories

def cathegory2 (a):
    if 0<a<=10000:
        return 'low'
    elif 10000<a<=100000:
        return 'high'

#new dataframe

df4=df1

#

price=df4['price']

plist2=[]

for i in range (len(price)):
    plist2.append(cathegory2(price[i]))
    
df4['Cathegory']=plist2

#numerical values

num_df4=df4.select_dtypes(include=['int64','float64']).copy()

#dropping unessecary values

num_df4.drop(['price','car_ID','symboling'],axis=1,inplace=True)

#adding new cathegories

num_df4['Cathegory']=plist2

#defining new X and Y

X4=num_df3.iloc[:, :-1].values

Y5=num_df3.loc[:,'Cathegory'].values

#scale X

scaled_data4=scale(X4)

#Label encoder on Y1

Y6=LabelEncoder().fit_transform(Y5)

#KMmeans for 2 cathegories

sil2=[]
com2=[]
hom2=[]
for k in range(2, 13):
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(scaled_data)
    print(k)
    print("silhouette_score = ", metrics.silhouette_score(scaled_data4, kmeans.labels_))
    print("completeness_score = ", metrics.completeness_score(Y6, kmeans.labels_))
    print("homogeneity_score = ", metrics.homogeneity_score(Y6, kmeans.labels_))
    sil2.append(metrics.silhouette_score(scaled_data4, kmeans.labels_))
    com2.append(metrics.completeness_score(Y6, kmeans.labels_))
    hom2.append( metrics.homogeneity_score(Y6, kmeans.labels_))
    
#new dataframe

scores_KM2=pd.DataFrame({'No. of Clusters': range(2,13), 'silhouette':sil2, 'completeness':com2, 'homogenity':hom2})

#plotting a scores 

plt.plot('No. of Clusters','silhouette',data=scores_KM2, color='teal', linewidth=1.2)
plt.plot('No. of Clusters', 'completeness', data=scores_KM2, color='darkgreen', linewidth=1.2)
plt.plot('No. of Clusters', 'homogenity', data=scores_KM2, color='purple', linewidth=1.2)
plt.legend()
plt.title('Results for different no. of clusters')



# In the report I have only used 5 categories!!!!!!!


def cathegory5 (a):
    if 0<a<=6000:
        return 'very_low'
    elif 6001<a<=12000:
        return 'low'
    elif 12000<a<=20000:
        return 'mid'
    elif 20000<a<=35000:
        return 'high'
    elif a>35000:
        return 'very_high'
    
price=df1['price']

plist5=[]

for i in range (len(price)):
    plist5.append(cathegory5(price[i]))
    
df1['Cathegory']=plist5

#checking for null values

df1_clean=df1.copy()

df1_clean.isnull().sum()

#no null values

#checking for outliers

sn.boxplot(x=df1['price'])

#nothing suprising we knew that majority of cars are in those groups

#Now fo KM clustering

df1.dtypes

#Viewing the object values

num_df1=df1.select_dtypes(include=['int64','float64']).copy()

num_df1

#adding cathegory to the new dataset

num_df1['Cathegory']=plist5

#removing price car symboling and car id as they not relevant from the new dataset

num_df1.drop(['price','car_ID','symboling'],axis=1,inplace=True)

for col in num_df1:
    print(col)
    
#creating an array of values

X=num_df1.iloc[:, :-1].values

Y=num_df1.loc[:,'Cathegory'].values

#now wee neet to scale the data

from sklearn.preprocessing import scale

scaled_data=scale(X)

scaled_data

#Label Encoder for the Y

from sklearn.preprocessing import LabelEncoder

Y2=LabelEncoder().fit_transform(Y)


#Now for the plotting 

sil5=[]
com5=[]
hom5=[]
for k in range(2, 13):
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(scaled_data)
    print(k)
    print("silhouette_score = ", metrics.silhouette_score(scaled_data, kmeans.labels_))
    print("completeness_score = ", metrics.completeness_score(Y2, kmeans.labels_))
    print("homogeneity_score = ", metrics.homogeneity_score(Y2, kmeans.labels_))
    sil5.append(metrics.silhouette_score(scaled_data, kmeans.labels_))
    com5.append(metrics.completeness_score(Y2, kmeans.labels_))
    hom5.append( metrics.homogeneity_score(Y2, kmeans.labels_))
    
#quick check

#sil
#com

#creating new dataframe

scores_KM5=pd.DataFrame({'No. of Clusters': range(2,13), 'silhouette':sil5, 'completeness':com5, 'homogenity':hom5})

#plotting newly created dataframe
plt.figure(figsize=(10,8))
plt.plot('No. of Clusters','silhouette',data=scores_KM5, color='teal', linewidth=3)
plt.plot('No. of Clusters', 'completeness', data=scores_KM5, color='darkgreen', linewidth=3)
plt.plot('No. of Clusters', 'homogenity', data=scores_KM5, color='purple', linewidth=3)
plt.legend()
plt.title('Results for different no. of clusters', size='xx-large')


#5 cathegories perform the best!!!!!!!!!!!!!!!!!!

#hierarchical clustering

n_digits = len(np.unique(Y))
Y2 = LabelEncoder().fit_transform(Y)
aff = ["euclidean", "manhattan", "cosine"]
link = ["ward", "complete", "average"] 
result = []
for a in aff:
    for l in link:
        if(l=="ward" and a!="euclidean"):
           continue
        else:
            model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage=l, affinity=a)
            model.fit(scaled_data)
            result.append([a,l,metrics.silhouette_score(scaled_data, model.labels_),metrics.completeness_score(Y2, model.labels_),metrics.homogeneity_score(Y2, model.labels_)])
maxI = -1
maxV = 0
for i in range(0,len(result)):
  print(result[i])
  if(result[i][2]>maxV):
    maxV = result[i][2]
    maxI = i
print("Max silhouette_score: ", result[maxI])
maxI = -1
maxV = 0
for i in range(0,len(result)):
  #print(result[i])
  if(result[i][3]>maxV):
    maxV = result[i][3]
    maxI = i
print("Max completeness_score: ", result[maxI])
maxI = -1
maxV = 0
for i in range(0,len(result)):
  #print(result[i])
  if(result[i][4]>maxV):
    maxV = result[i][4]
    maxI = i
print("Max homogeneity_score: ", result[maxI])



result



scores_HC=pd.DataFrame({'Approach': ['euclidean_ward','euclidean_complete','euclidean_average','manhattan_complete',\
                                     'manhattan_average','cosine_complete','cosine_average'], 'Silhoutte_Score':[0.2205934283704637, 0.20464984551355583, 0.23642857149074226, 0.18400762562084655, 0.22949454217500334, 0.12756639971655728, 0.16617575931788495],
                        'Completeness_Score': [0.33440511329872186, 0.3219222283174183, 0.43912957323951346, 0.3923262019785241, 0.37946968074671106, 0.31219222199735974, 0.3261561995668747],
                        'Homogenity_Score': [0.4023271562334016, 0.31247197942435484, 0.1661929537519801, 0.4815570799137614, 0.1276383048644587, 0.3992161457007582, 0.3855450436761425]
                        })


tick=['euclidean_ward','euclidean_complete','euclidean_average','manhattan_complete','manhattan_average','cosine_complete','cosine_average']
scores_HC.plot(kind='bar', color=('purple','cyan','darkgreen'), legend=True, figsize=(15,8),ylim=(0,0.5)).set_xticklabels(tick,rotation=0,fontsize='medium')


#plotting dedogram for the best approach i.e. euclidean_average

import scipy.cluster.hierarchy as sch

dendrogram_e_a= sch.dendrogram(sch.linkage(scaled_data, method='average',metric='euclidean'))

dendrogram_m_a= sch.dendrogram(sch.linkage(scaled_data, method='average',metric='manhattan'))
          
                               
          
                               
#regression  My apologies for a mess in this section, I was struggling a lot with linear regression and thats the best I could organize it. 

#Multiple linear regression

df1_r=df1

#extracting numerical values

numerical_columns=['price','wheelbase','carlength','carwidth','carheight','curbweight','cylindernumber','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg']

#num_df1=df1_r.select_dtypes(include=['int64','float64']).copy()

#dataset with only numerical values with price in the beginning

df_r=df1_r[numerical_columns]

#num_df1.drop(['car_ID','symboling'],inplace=True,axis=1)

#defining which variables we want to get dummies for

dummy_col= ['aspiration','fueltype','fuelsystem','carbody','drivewheel','enginetype']

#getting dummies for those columns

dummy=pd.get_dummies(df1_r[dummy_col])

#adding numerical values to the newly created dummies

df_r=pd.concat([df_r, dummy],axis=1)


#df_r[numerical_columns]=scale.fit_transform(df_r[numerical_columns])

plt.figure(figsize = (20, 20))
sn.heatmap(df_r.corr(), cmap="BrBG", annot=True)
plt.title('Heatmap of all variables including dummies',size='xx-large')
plt.show()

sn.heatmap(a, cmap="BrBG", annot=True)
    
a=df_r.corr()['price'][:]

df_corr = df.corr()

#correlation plot

a.plot(kind='bar', color='darkgreen', figsize=(12,6))
plt.yticks(np.arange(-1,1.1, step=0.2))
plt.axhline(y=0.2, xmin=0, xmax=1, lw=0.5, color='r', linestyle='dashed')
plt.axhline(y=-0.2, xmin=0, xmax=1, lw=0.5, color='r', linestyle='dashed')
plt.axhline(y=0, xmin=0, xmax=1, lw=0.5, color='k')
plt.title('Correlation between Price and chosen Variables')

#df1_reg.drop(['price'],inplace=True,axis=1)

#splitting data into test and train

#standard 70/30 split

from sklearn.model_selection import train_test_split

X_train, X_test= train_test_split(df_r, train_size=0.7, test_size=0.3)


#from sklearn.preprocessing import scale

#from sklearn.preprocessing import StandardScaler

#scale= StandardScaler()

#scale the numerical data, so that we can have it in a form of a dataset

X_train[numerical_columns]=scale.fit_transform(X_train[numerical_columns])


#plt.figure(figsize = (20, 20))
#sn.heatmap(X_train.corr(), cmap="BrBG", annot=True)
#plt.title('Heatmap of all variables including dummies',size='xx-large')
#plt.show()


#making sure everything is correct

#X_train

#geting  the Y variable, using pop so no column 

Y_train=X_train.pop('price')

#modeling the MLR

#based on the second heatmap

X_train_rel=X_train[['wheelbase','carlength','carwidth','carheight','curbweight','cylindernumber','enginesize','boreratio',\
                  'horsepower','citympg','highwaympg','fuelsystem_2bbl','fuelsystem_mpfi','carbody_hardtop', 'carbody_hatchback',\
                      'drivewheel_fwd','drivewheel_rwd','enginetype_ohc','enginetype_ohcv']]

#now for the regression 

import statsmodels.api as sm

#now we need to add a constant

X_train_rel1=sm.add_constant(X_train_rel)

#cearing a model

model1=sm.OLS(Y_train, X_train_rel1).fit()

#showing the parameters

model1.params

print(model1.summary())

#checking standard errors distribution

Y_train_price=model1.predict(X_train_rel1)

fig = plt.figure()
sn.distplot((Y_train-Y_train_price),bins=20,color='purple')
fig.suptitle("Error terms Histogram", fontsize=12)
plt.xlabel('errors', fontsize=10)

#now for testing the model

#we need to scale the X_test just like we scaled X_train

X_test[numerical_columns]=scale.transform(X_test[numerical_columns])

#now getting the test price

Y_test=X_test.pop('price')

X_test=X_test

#aggain adding constant

X_test_rel2=sm.add_constant(X_test)

X_test_rel3=X_test_rel2[X_train_rel1.columns]

#now to make a predictions using model1

Y_Pred=model1.predict(X_test_rel3)

#RMSE score
from sklearn.metrics import r2_score

r2_score(Y_test,Y_Pred)

#RMSE for the test
from sklearn.metrics import mean_squared_error
from math import sqrt

RMSE=sqrt(mean_squared_error(Y_test, Y_Pred))

#RMSE for the train

RMSE_t=sqrt(mean_squared_error(Y_train, Y_train_price))

#MAE for the train

from sklearn.metrics import mean_absolute_error

RAE_t=mean_absolute_error(Y_train, Y_train_price)

#MAE for the test

RAE=mean_absolute_error(Y_test, Y_Pred)







#Gaussian approach

df_g=pd.read_csv(r'/Users/aleksanderbielinski/Desktop/CarPrice1.csv')


obj_df_g=df_g.select_dtypes(include=['object']).copy()


new_nums={'cylindernumber': {"four": 4, "six": 6, "five": 5, "eight": 8, "two": 2, "three": 3, "twelve": 12}}



df_g.replace(new_nums, inplace=True)


num_df_g=df_g.select_dtypes(include=['int64','float64']).copy()


#use categories

def cathegory5 (a):
    if 0<a<=6000:
        return 'very_low'
    elif 6001<a<=12000:
        return 'low'
    elif 12000<a<=20000:
        return 'mid'
    elif 20000<a<=35000:
        return 'high'
    elif a>35000:
        return 'very_high'
    
price=num_df_g['price']

plist5=[]

for i in range (len(price)):
    plist5.append(cathegory5(price[i]))
    
num_df_g['Cathegory']=plist5

#dropping unwanted cathegories

num_df_g.drop(['price','car_ID','symboling'],axis=1,inplace=True)

#sets for model

X_g=num_df_g.values[:, 0:14]

Y_g=num_df_g.loc[:, "Cathegory"].values

#scale X_g
#from sklearn.preprocessing import scale

scaled_X_g=scale(X_g)

#Label encoder on the cathegories
#from sklearn.preprocessing import LabelEncoder

Y_g_enc=LabelEncoder().fit_transform(Y_g)

#splitting the data 
#from sklearn import model_selection

X_train_g, X_test_g, Y_train_g, Y_test_g=model_selection.train_test_split(scaled_X_g, Y_g_enc, test_size=0.30)

#now to the actual Gaussian Naive
#from sklearn.naive_bayes import GaussianNB

print("\n\n Naive Bayes")
print("**************************************")
model = GaussianNB()
model.fit(X_train_g, Y_train_g)
print(model)
predicted = model.predict(X_test_g)
print(metrics.classification_report(Y_test_g, predicted))
print(metrics.confusion_matrix(Y_test_g, predicted))


print("\n\nDecision Tree")
print("**************************************")
from sklearn.tree import DecisionTreeClassifier
#model = DecisionTreeClassifier()
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train_g, Y_train_g)
print(model)
predicted = model.predict(X_test_g)
print(metrics.classification_report(Y_test_g, predicted))
print(metrics.confusion_matrix(Y_test_g, predicted))



#getting the version 

import sys
sys.version

















