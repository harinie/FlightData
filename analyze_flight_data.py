# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:33:33 2015

@author: tintin
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.preprocessing import Imputer, scale
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer as DV
import matplotlib.pyplot as plt

flight_data='/home/tintin/Data/DataIncubator/flight_data/2008_LGA_ORD.csv'
weather_data='/home/tintin/Data/DataIncubator/flight_data/LGA_ORD_2008_weather.csv'

dtypes={'Year':np.int,'Month': pd.Categorical,'DayofMonth': np.int,'DayOfWeek': pd.Categorical,'DepTime':np.float64,
'CRSDepTime':np.int,
'ArrTime':np.float64,'CRSArrTime':np.float64,'UniqueCarrier':object,'FlightNum':np.float64,'TailNum':object,
'ActualElapsedTime':np.float64,
'CRSElapsedTime':np.float64,'AirTime':np.float64,'ArrDelay':np.float64,'DepDelay':np.float64,'Origin':object,
'Dest':object,'Distance':np.float64,
'TaxiIn':np.float64,'TaxiOut':np.float64,'Cancelled':np.float64,'CancellationCode':object,'Diverted':np.float64,
'Carrier Delay':np.float64,
'WeatherDelay':np.float64,'NASDelay':np.float64,'SecurityDelay':np.float64,'LateAircraftDelay':np.float64}

df_data = pd.read_csv(flight_data,header=0,dtype=dtypes, quotechar='"',sep=',',na_values = ['NA', '-', '.', ''])

df_data = df_data.ix[((df_data['Dest']=="ORD") | (df_data['Dest']=="LGA")) & ((df_data['Origin']=="LGA") | (df_data['Origin']=="ORD"))]

df_data["EST"] = [str(df_data.ix[i,"Year"])+'-'+str(df_data.ix[i,"Month"])+'-'+str(df_data.ix[i,"DayofMonth"]) for i in range(df_data.shape[0])]

cols_to_drop=['Year','DayofMonth','DepTime','ArrTime','CRSArrTime','Distance','ActualElapsedTime','FlightNum','TailNum','CRSElapsedTime','AirTime','ArrDelay','TaxiIn','TaxiOut']
df_data.drop(cols_to_drop, axis=1, inplace=True)

df_data["CRSDepTime"] = np.floor(df_data["CRSDepTime"]/100 )
df_data["CRSDepTime"] = df_data["CRSDepTime"].apply(str)

df_weather = pd.read_csv(weather_data,header=0,dtype=dtypes, quotechar='"',sep=',',na_values = ['NA','T', '-', '.', ''])

cols_to_drop=['Max TemperatureF','Min TemperatureF','Max Dew PointF','Min DewpointF','Max Humidity',' Min Humidity',' Max Sea Level PressureIn',' Min Sea Level PressureIn']
df_weather.drop(cols_to_drop, axis=1, inplace=True)

df_data = pd.merge(df_data,df_weather, how='left', left_on=['EST','Origin'],right_on=['EST','Location'],suffixes=('','_Origin'))
df_data = pd.merge(df_data,df_weather, how='left', left_on=['EST','Dest'],right_on=['EST','Location'],suffixes=('','_Dest'))

df_data[' WindDirDegrees'] = np.floor(df_data[" WindDirDegrees"]/30 )
df_data[' WindDirDegrees_Dest'] = np.floor(df_data[" WindDirDegrees_Dest"]/30 )
df_data[" WindDirDegrees"] = df_data[" WindDirDegrees"].apply(str)
df_data[" WindDirDegrees_Dest"] = df_data[" WindDirDegrees_Dest"].apply(str)

df_data.ix[pd.isnull(df_data["DepDelay"]),"DepDelay"]=0

labels = df_data["Cancelled"].values
delay_vals = df_data["DepDelay"].values

#%% one hot encode

categorical_cols = ["Month", "DayOfWeek","CRSDepTime","UniqueCarrier",' Events',' WindDirDegrees',' Events_Dest', 
' WindDirDegrees_Dest']

cat_df = df_data[ categorical_cols ]
cat_dict = cat_df.T.to_dict().values()


vectorizer = DV( sparse = False )
cat_features = vectorizer.fit_transform( cat_dict )
cat_keys = vectorizer.get_feature_names()

#%% combine discrete and cont data
all_keys = ["Mean TemperatureF",' Mean Humidity', 
' Mean Sea Level PressureIn', ' Max VisibilityMiles', ' Mean VisibilityMiles', ' Min VisibilityMiles',
' Max Wind SpeedMPH', ' Mean Wind SpeedMPH', ' Max Gust SpeedMPH','PrecipitationIn', ' CloudCover', 
 'Mean TemperatureF_Dest', 'MeanDew PointF_Dest', ' Mean Humidity_Dest',
' Mean Sea Level PressureIn_Dest', ' Max VisibilityMiles_Dest', ' Mean VisibilityMiles_Dest', 
' Min VisibilityMiles_Dest', ' Max Wind SpeedMPH_Dest', ' Mean Wind SpeedMPH_Dest', 
' Max Gust SpeedMPH_Dest', 'PrecipitationIn_Dest', ' CloudCover_Dest']
 
cont_features = df_data[list(all_keys)].values

all_features=np.concatenate((cont_features, cat_features), axis=1)
all_keys = all_keys + cat_keys
#%% impute data
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
all_features = imp.fit(all_features).transform(all_features)    

scale(all_features, axis=0, with_mean=True, with_std=True, copy=False)

#%% run classification
cv=StratifiedShuffleSplit(labels, n_iter=10, test_size=0.1)

tuned_parameters = [{'C': np.logspace(-5,5,11,base=10)}]
#clf_base = SVC()
clf_base =LogisticRegression(penalty='l1',dual=False)
clf = GridSearchCV(clf_base, tuned_parameters, scoring='roc_auc', cv=cv,refit=True,verbose=True)
clf.fit(all_features,labels)

#%%
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() / 2, params))     

clf.best_estimator_.fit(all_features,labels)

#%% custom fit 

clf_base =LogisticRegression(penalty='l1',dual=False,C=0.1)
clf_base.fit(all_features,labels)

#%%
#weights = clf.best_estimator_.coef_
weights = clf_base.coef_
weights = np.reshape(weights,[weights.shape[1]])
weights_pos = np.where(weights>0.5*max(weights))[0]

plt.subplot(2,1,1)
barlist=plt.barh(range(weights_pos.shape[0]),weights[weights_pos],color='b',alpha=0.3)

high_keys = [all_keys[idx] for idx in weights_pos]
    
plt.yticks(range(weights_pos.shape[0]), high_keys,fontsize=14)

#plt.axes().set_aspect(10)
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.05)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.5)

plt.title('Factors that cause flight cancellation, LGA-ORD',fontsize=18)

weights_neg = np.where(weights<0.5*min(weights))[0]

plt.subplot(2,1,2)
barlist=plt.barh(range(weights_neg.shape[0]),np.abs(weights[weights_neg]),color='b',alpha=0.3)

high_keys_neg = [all_keys[idx] for idx in weights_neg]
    
plt.yticks(range(weights_neg.shape[0]), high_keys_neg,fontsize=14)

#plt.axes().set_aspect(10)
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.05)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.5)

plt.xlabel('Classifier Weight',fontsize=18)
plt.title('Factors that keep flights from getting cancelled, LGA-ORD',fontsize=18)

plt.show()            

#%% run regression
# first remove cancelled flights - 
keep_ids = df_data["Cancelled"] == 0
all_features = all_features[keep_ids.values,:]

delay_vals = delay_vals[keep_ids.values]
clf=None
clf_base=None

tuned_parameters = [{'alpha': np.arange(1000,10000,1000)}]
#clf_base = SVC()
clf_base =Ridge()
clf = GridSearchCV(clf_base, tuned_parameters, scoring='r2', cv=10,refit=True,verbose=True)
clf.fit(all_features,delay_vals)

#%%
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() / 2, params))     

clf.best_estimator_.fit(all_features,delay_vals)
delay_vals_predict=clf.best_estimator_.predict(all_features)

#%%
weights=None
weights = clf.best_estimator_.coef_
#weights = clf_base.coef_
weights = np.reshape(weights,[weights.shape[1]])
weights_pos = np.where(weights>0.5*max(weights))[0]

plt.subplot(2,1,1)
barlist=plt.barh(range(weights_pos.shape[0]),weights[weights_pos],color='b',alpha=0.3)

high_keys = [all_keys[idx] for idx in weights_pos]
    
plt.yticks(range(weights_pos.shape[0]), high_keys, fontsize=14)

#plt.axes().set_aspect(10)
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.1)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.5)
plt.title('Factors that cause flight delays, LGA-ORD',fontsize=18)

weights_neg = np.where(weights<0.5*min(weights))[0]

plt.subplot(2,1,2)
barlist=plt.barh(range(weights_neg.shape[0]),np.abs(weights[weights_neg]),color='b',alpha=0.3)

high_keys_neg = [all_keys[idx] for idx in weights_neg]
    
plt.yticks(range(weights_neg.shape[0]), high_keys_neg, fontsize=14)

#plt.axes().set_aspect(10)
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.1)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.5)

plt.xlabel('Regression Weight',fontsize=18)
plt.title('Factors that keep flights from getting delayed, LGA-ORD',fontsize=18)

plt.show()            