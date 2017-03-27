''' 
Author: Pratik Mohan Ramdasi

Date: 03/16/2017

Title: Predict popularity of apartment listings - Kaggle (TwoSigma/RentHop)

Method: Random Forest Classifier

'''

## Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

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
Address_result = '/home/pratikramdasi/Kaggle/twoSigma_rentHop/submission.csv'
df_train = pd.read_json(open(Address_train, 'r'))
df_test = pd.read_json(open(Address_test, 'r'))
	# concatenate train and test data for preprocessing
df_train = df_train.dropna(subset=['interest_level'])
df_whole_data = pd.concat([df_train, df_test], keys=['x','y'])

## Preprocess the data 	
(processed_train_data, processed_test_data) = preProcess_data(df_whole_data)

## Separate training, validation and testing data for model development
X = processed_train_data.drop(['interest_level'], axis=1)
y = processed_train_data['interest_level']
X_train, X_val, y_train, y_val = train_test_split(X, y)
X_test = processed_test_data.drop(['interest_level'], axis=1)

## Design the model
rf = RandomForestClassifier(n_estimators=1000, criterion='entropy', min_samples_split=20)

## Fit data into the model and check validation error as multi-class logarithmic loss function
rf.fit(X_train,y_train)
y_pred_val = rf.predict_proba(X_val)
loss = log_loss(y_val, y_pred_val)
print "Logarithmic loss is %.4f" % loss

## Predict output for test data
y_pred = rf.predict_proba(X_test)

## Map target labels into index values 
targets = rf.classes_
map_to_idx = {name: n for n, name in enumerate(targets)}

## Write output into csv
res = pd.DataFrame()
res['listing_id'] = df_test['listing_id']
for label in ["high","medium", "low"]:
	res[label] = y_pred[:, map_to_idx[label]]
print res.head()
res.to_csv(Address_result, sep=',', index=False)



