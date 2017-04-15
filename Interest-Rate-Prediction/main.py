''' 
Project: Develope prediction models to peredict interest rates.

Author : Pratik Mohan Ramdasi

Date   : 02/ 09/ 2017

Models : Support Vector Regression, Principle Component Regression, Random Forest Regressor

''' 

import numpy as np
import pandas as pd
import datetime
import time 

from data_PreProcessor import PreProcess_data
from data_PreProcessor import DTFormatOpt
from models import SVR_model
from models import SVR_predict
from models import PCA_model
from models import PCA_predict
from models import RF_model
from models import RF_predict
from sklearn.preprocessing import scale


# Read input data
Address_train = '/home/pratikramdasi/comp_inter/statefarm/work_assignment/Data for Cleaning & Modeling.csv'
Address_test = '/home/pratikramdasi/comp_inter/statefarm/work_assignment/Holdout for Testing.csv'
Address_test_SVR = '/home/pratikramdasi/comp_inter/statefarm/work_assignment/Results_m1.csv'
Address_test_PCA = '/home/pratikramdasi/comp_inter/statefarm/work_assignment/Results_m2.csv'
Address_test_RFR = '/home/pratikramdasi/comp_inter/statefarm/work_assignment/Results_m3.csv'
Address_combined = '/home/pratikramdasi/comp_inter/statefarm/work_assignment/Results.csv'

IntRate_train_data = pd.read_csv(Address_train, low_memory=False)
IntRate_train_data = IntRate_train_data.drop(IntRate_train_data.index[399999])
IntRate_train_data = IntRate_train_data.dropna(subset=['X1'])
IntRate_test_data = pd.read_csv(Address_test, low_memory=False)
IntRate_whole_data = pd.concat([IntRate_train_data, IntRate_test_data], keys=['x','y'])

(Processed_train_data1, Processed_test_data) = PreProcess_data(IntRate_whole_data)

# considering first 3000 samples for training
Processed_train_data = Processed_train_data1.sample(n = 3000, replace = False)

# Separate the input variables and output variables, scale the input data
Input_train_data = Processed_train_data.iloc[:, 1:]
Scaled_train_data = scale(Input_train_data)
Output_data = Processed_train_data['X1']
Input_test_data = Processed_test_data.iloc[:, 1:]
Scaled_test_data = scale(Input_test_data)

######################################### First Model: Support Vector Regression #################################################

# Building the model
(MeanMSE_SVR, svr_tuned) = SVR_model(Scaled_train_data, Output_data)

# Predict test set using built model
SVR_results = SVR_predict(svr_tuned, Scaled_test_data, Address_test_SVR)

######################################### Second Model: PCA + Linear Regression #################################################

# Build the model
(MeanMSE_PCA, pca, regr) = PCA_model(Scaled_train_data, Output_data)

# Predict values for testing
PCA_results = PCA_predict(pca, regr, Scaled_test_data, Address_test_PCA)

######################################### Third Model: Random Forest Regression #############################################

# Build the model
(MeanMSE_RF, rf) = RF_model(Scaled_train_data, Output_data)

# Predict values for testing
RF_results = RF_predict(rf, Scaled_test_data, Address_test_RFR)

################################################################################################################################

# concatenate results into one csv file
df1 = pd.read_csv(Address_test_SVR, header=None)
df2 = pd.read_csv(Address_test_PCA, header=None)
df3 = pd.read_csv(Address_test_RFR, header=None)
dflist = [df1, df2, df3]
concatDf = pd.concat(dflist, axis=1)  # append column wise
concatDf.to_csv(Address_combined, header=False, index=None)




	
