''' 
Author: Pratik Mohan Ramdasi

Date: 03/20/2017

Title: Predict popularity of apartment listings - Kaggle (TwoSigma/RentHop)

Method: XG Boost Classifier

'''

## Import necessary libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize']= 12, 4

## Data Pre-processing
def preProcess_data(data):
	## Transform variables
		# Formatting 'created' variable to get 'year', 'month' and 'day' separately
	data['created'] = pd.to_datetime(data['created'])
	data['created_year'] = data['created'].dt.year
	data['created_month'] = data['created'].dt.month
	data['created_day'] = data['created'].dt.day
		# Replace descriptive variables with size/length of description
	data['num_features'] = data['features'].apply(len)
	data['num_photos'] = data['photos'].apply(len)
	data['len_description'] = data['description'].apply(lambda x: len(x.split(" ")))
		# Remove unnecessary independent variables
	rem_var = ['building_id', 'created','description','display_address','features','listing_id','manager_id','photos','street_address']
	data = data.drop(rem_var, axis=1)
	## Return train and test data
	train_data = data.ix['x']
	test_data = data.ix['y']
	return (train_data, test_data)

## Read train and test json object files
Address_train = '/home/pratikramdasi/Kaggle/twoSigma_rentHop/train.json'
Address_test = '/home/pratikramdasi/Kaggle/twoSigma_rentHop/test.json'
Address_result = '/home/pratikramdasi/Kaggle/twoSigma_rentHop/submission_2.csv'
df_train = pd.read_json(open(Address_train, 'r'))
df_test = pd.read_json(open(Address_test, 'r'))
	# concatenate train and test data for preprocessing
df_train = df_train.dropna(subset=['interest_level'])
df_whole_data = pd.concat([df_train, df_test], keys=['x','y'])

## Preprocess the data 	
(processed_train_data, processed_test_data) = preProcess_data(df_whole_data)

## Map target labels into index values 
targets = processed_train_data['interest_level'].unique()
map_to_idx = {name: n for n, name in enumerate(targets)}
print map_to_idx

## Separate training, validation and testing data for model development
X = processed_train_data.drop(['interest_level'], axis=1)
y = processed_train_data['interest_level'].apply(lambda x: map_to_idx[x])
X_test = processed_test_data.drop(['interest_level'], axis=1)

## Design the model
params = {'objective':'multi:softprob', 'num_class': 3, 'max_depth': 4, 'eval_metric': 'mlogloss',
			'subsample': 0.7, 'colsample_bytree' : 0.7, 'silent': 1 }
num_rounds = 1000
xgtrain = xgb.DMatrix(X, label=y)
clf = xgb.train(params, xgtrain, num_rounds)

## Predict output for test data
xgtest = xgb.DMatrix(X_test)
y_pred = clf.predict(xgtest)

## Write output into csv
res = pd.DataFrame()
res['listing_id'] = df_test['listing_id']
for label in ["high","medium", "low"]:
	res[label] = y_pred[:, map_to_idx[label]]
print res.head()
#res.to_csv(Address_result, sep=',', index=False)



