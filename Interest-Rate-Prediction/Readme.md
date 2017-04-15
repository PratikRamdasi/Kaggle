
Objective
==========
Develop Regression models for borrowers with 32 variables to predict their interest rate in Python.


Cleaning and Preparing data
============================
1. Delete unnecessary variables
Following variables are removed from the dataset as they do not contribute towards prediction: 
X2 - A unique ID for the loan 
X3 - unique ID assigned for the borrower 
X8 - load grade 
X16 - Reason for loan provided by borrower 
X18 - Loan title 
X19 - First 3 numbers of zip code 
X20 - State of the borrower

2. Convert the coulmns with dollar and percentage sign to floats.

3. Handle missing values
Missing values were replaced with median if they're continuous and by mode if they're catagorical. Median is used because it is more robust and it is less expensive.

4. Handle catagorical variables by converting them to binary dummy variables. 

5. Transform some of the variables as follows:
-> Variable X5 was subtracted from variable X4 as they are highly correlated with each other.
-> Variable X15 was catagorized into 4 quarters and converted into binary dummy variable. This variable represent time 	of the year the loans were issued. Interest rate will be different during different quarters in the year.	
-> Variable X23 was subtracted from most recent credit line which was opened among all the borrowers. New variable 	  represents relative age of each borrower's credit line.
-> Variable X11 was converted to floats. 
* Borrowers with work exp < 1 year and missing value for 'employer or job title' is replaced with 0 years.  
* Borrowers with work exp < 1 year and existing value for 'employer or job title' is replaced with 1 year.
* Borrowers with work exp > 10 year is replaced by on average 12 years.


Building Models
================
After preparing the dataset, three models are developed to predict the interest rates. First model is Support Vector Regression (SVR), Second model is Linear Regression with Principle Component Analysis(PCR) and Third model is Random Forest Regression (RFR).

SVR Model
---------- 
-> For tuning the parameters in the model, random permutation cross vaildation was used in which in each iteration 0.01% of the dataset was used for training and 0.05% of the dataset was used for validation. 
-> The process repeats 10 times and find parameters which denote the lowest MSE. The best parameters are selected among C = [0.1, 1, 10] and gamma = [0.01, 0.1, 1, 10].
-> Method takes advantage of kernel trick (RBF kernel) in which original space is transformed to a non-linear space. 

PCA + Linear Regression model (Principle Components Regression)
----------------------------------------------------------------
-> In PCR, principle components analysis(PCA) is used to decompose the independent variables into an orthogonal basis (principle components or PCs) and select a subset of those components as the variables to predict the output.
-> Selected first 30 PCs with highest variance, that is the components with the largest singular values because the subspace defined by these PCs capture most of the variation ( > 93% ) in the data and thus represents a smaller space that we believe captures most of the qualities of the data.
-> Effect of PCA on MSE is studied by performing 10-fold cross-validation. Based on finding smallest cross- vaildation	   error, number of PCs are selected to be 30.
-> Dimensionally reduced data is then fitted into Linear Regression model for prediction.

Random Forest Regression (RFR)
----------------------------------
-> Useful for regression analysis on huge dataset with large number of features (Predictors). It undertakes dimensionality reduction and it is helpful when dealing with missing values and outliers.
-> Bagging model consists of 100 decision trees and final prediction of the value is done by taking average of each decision tree prediction. 

Model Comparison and Prediction  Results
=========================================
SVR model has a penalty term (L1 norm) which helps with generality of the model. Loss function for SVR is epsilon-insensitive loss. On the other hand,as a result of the way PCR is implemented, the final model is more difficult to interpret because it does not perform any kind of variable selection or even directly produce coefficient estimates. Rnadom Forest model seems to be best choice for this project. Bootstrap sampling feature in RF regressor helps optimizing and generalizing the results. 
In terms of model complexity, the models are time consuming for the large dataset and therefore, smaller portion of the dataset (3000 samples) is considered for training and validation.
In terms of MSE, using 10-fold cross validation on a sample dataset with 3000 samples, RF outperforms SVR and PCR. MSE for RFR is 2.54 while MSE for SVR is 3.7096 and MSE for PCR with Linear Regression is 6.8824. 


Finally, the test set were predicted using the built models and the results were written in the .CSV file.
