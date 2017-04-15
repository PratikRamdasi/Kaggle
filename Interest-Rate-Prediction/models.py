
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
from data_PreProcessor import PreProcess_data
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV 
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split

def SVR_model(Scaled_input_data, Output_data):
	n = len(Scaled_input_data)	
	# test on different C and gamma values
	Grid_dict = {"C":[1e-1, 1e0, 1e1], "gamma": np.logspace(-2, 1, 3)}
	svr_tuned = GridSearchCV(SVR(kernel='rbf', gamma = 0.1, tol = 0.05), cv= 5, param_grid = Grid_dict, scoring='neg_mean_squared_error')
	MeanMSE_SVR = 1
	# fit training data into tuned SVR model
	svr_tuned.fit(Scaled_input_data, Output_data)
	#t0 = time.time()
	# select best C and gamma values
	SVR_MSE = SVR(kernel='rbf', C=svr_tuned.best_params_['C'], gamma= svr_tuned.best_params_['gamma'], tol=0.01)
	#SVR_time = time.time() - t0
	#print ('Computational time of RBF based SVR for ', n , ' examples is: ', SVR_time/10)
	# compute cross-validation MSE for 10-folds
	MSEs_SVR = cross_validation.cross_val_score(SVR_MSE, Scaled_input_data, Output_data, cv=10, scoring='neg_mean_squared_error')
	MeanMSE_SVR = np.mean(list(MSEs_SVR))
	print ('The average MSE of RBF SVR for ' , n , ' examples is: ' , MeanMSE_SVR)
	return(MeanMSE_SVR, svr_tuned)
	

def SVR_predict(svr_tuned, Input_test_data, Address_test):
	Predicted_SVR = svr_tuned.predict(Input_test_data)
	Predicted_SVR_S = pd.Series(Predicted_SVR)
	Predicted_SVR_S.to_csv(Address_test, sep=',', index=False, header=['Model1_Results'])
	return(Predicted_SVR)
	

def PCA_model(Scaled_input_data, Output_data):
	pca = PCA()
	# reduce scaled data dimensions
	reduced_train_data = pca.fit_transform(Scaled_input_data)
	n = len(reduced_train_data)
	# perform 10-fold cross validation with shuffle
	kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=1)
	# regression model
	regr = LinearRegression()
	mse = []
	# select number of Principle Components (PCs)
		# based on the cumulative variance over number of PCs, 30 principle components are selected which represent 
		# > 93% of the data
	cum_var_exp = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
	plt.plot(cum_var_exp)
	plt.xlabel('Number of Principle Components')
	plt.ylabel('Cumulative Variance')
	plt.show() 	
	# calculate MSE using Cross Validation for 30 PCs
	for i in np.arange(1, 30):
		score = -1 * cross_validation.cross_val_score(regr, reduced_train_data[:,:i] , Output_data.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
		mse.append(score)
	# plot results
	plt.plot(np.array(mse), '-v')
	plt.xlabel('Number of Principle Components in regression')
	plt.ylabel('Cross Validation MSE')
	plt.xlim(xmin=-1)
	plt.show()	
	# calculate mean MSE
	MeanMSE_PCA = np.mean(list(mse))
	print ('The average MSE of PCR for ' , n , ' examples is: ' , MeanMSE_PCA)	
	# fit regression model on training data
	regr.fit(reduced_train_data[:,:30], Output_data)
	return(MeanMSE_PCA, pca, regr)
	
	
def PCA_predict(pca, regr, Input_test_data, Address_test):
	reduced_test_data = pca.transform(Input_test_data)[:,:30]  # use first 30 PCs 
	Predicted_PCA = regr.predict(reduced_test_data) 
	Predicted_PCA_S = pd.Series(Predicted_PCA)
	Predicted_PCA_S.to_csv(Address_test, sep=',', index=False, header=['Model2_Results'])
	return(Predicted_PCA)

def RF_model(Scaled_input_data, Output_data):
	n = len(Scaled_input_data)
	rf = RandomForestRegressor(n_estimators=100)
	rf.fit(Scaled_input_data, Output_data)
	MeanMSE_RF = np.mean(cross_validation.cross_val_score(rf, Scaled_input_data, Output_data, scoring='neg_mean_squared_error', cv=10, n_jobs=1))
	print ('The average MSE of RF for ' , n , ' examples is: ' , MeanMSE_RF)
	return(MeanMSE_RF, rf)
	
def RF_predict(rf, Input_test_data, Address_test): 
	Predicted_RFR = rf.predict(Input_test_data) 
	Predicted_RFR_S = pd.Series(Predicted_RFR)
	Predicted_RFR_S.to_csv(Address_test, sep=',', index=False, header=['Model3_Results'])
	return(Predicted_RFR)	
