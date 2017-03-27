''' 
Author: Pratik Mohan Ramdasi

Date: 03/16/2017

Title: Predict popularity of apartment listings - Kaggle (TwoSigma/RentHop)

Model : PCA + SVM

'''

## Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV 
from sklearn.metrics import log_loss
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

## Remove deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
Address_result = '/home/pratikramdasi/Kaggle/twoSigma_rentHop/submission_1.csv'
df_train = pd.read_json(open(Address_train, 'r'))
df_test = pd.read_json(open(Address_test, 'r'))
	# concatenate train and test data for preprocessing
df_train = df_train.dropna(subset=['interest_level'])
df_whole_data = pd.concat([df_train, df_test], keys=['x','y'])

## Preprocess the data 	
(processed_train_data, processed_test_data) = preProcess_data(df_whole_data)

## Separate training, validation and testing data for model development
X = scale(processed_train_data.drop(['interest_level'], axis=1))
y = processed_train_data['interest_level']
X_train, X_val, y_train, y_val = train_test_split(X, y)
X_test = scale(processed_test_data.drop(['interest_level'], axis=1))

## PCA dimension reduction
pca = PCA()
'''
X_train_reduced = pca.fit_transform(X_train)
	# Plot cumulative variance along PCs
cum_var_exp = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(cum_var_exp)
plt.xlabel('Number of Principle Components')
plt.ylabel('Cumulative Variance')
plt.show()
'''
	# Based on the cumulative variance part, 8 PCs are sufficient to represent the entire data variance 
X_train_reduced = pca.fit_transform(X_train)[:,:8]
X_val_reduced = pca.transform(X_val)[:,:8]
X_test_reduced = pca.transform(X_test)[:,:8]
print (X_train_reduced.shape, X_val_reduced.shape, X_test_reduced.shape)

## SVM classifier
'''
Grid_dict = {"C":[1e-1, 1e0, 1e1], "gamma": np.logspace(-2, 1, 3)}
svc_tuned = GridSearchCV(svm.SVC(kernel='linear', gamma = 0.1, tol = 0.05), cv= 5, param_grid = Grid_dict)
svc_tuned.fit(X_train_reduced, y_train)
print svc_tuned.best_params_
#best params - C: 0.1, Gamma = 0.01
'''
svc = svm.SVC(C=0.1, kernel='linear', gamma= 0.01, probability=True)
	# fit training data into SVC model and obtain log loss for validation data
svc.fit(X_train_reduced, y_train)
y_pred_val = svc.predict_proba(X_val_reduced)
loss = log_loss(y_val, y_pred_val)
print "Logarithmic loss is %.4f" % loss

## Predict output for test data
y_pred = svc.predict_proba(X_test_reduced)

## Map target labels into index values 
targets = y.unique()
map_to_idx = {name: n for n, name in enumerate(targets)}

## Write output into csv
res = pd.DataFrame()
res['listing_id'] = df_test['listing_id']
for label in ["high","medium", "low"]:
	res[label] = y_pred[:, map_to_idx[label]]
print res.head()
res.to_csv(Address_result, sep=',', index=False)



