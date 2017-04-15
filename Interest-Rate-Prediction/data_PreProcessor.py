
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import datetime


### function to format date, time
def DTFormatOpt(s, flist):
	for f in flist:
		try:
			return datetime.strptime(s,f)
		except ValueError:
			pass

### preprocess the input data
def PreProcess_data(IntRate_whole_data):
	
	# Delete unnecessary columns - X2, X3, X8, X16, X18, X19, X20
	IntRate_dropped = IntRate_whole_data.drop(['X2', 'X3', 'X8', 'X16', 'X18', 'X19', 'X20'], axis = 1)
	
	# Replace $ and % sign into float values - X4, X5, X6 and X1, X30
	IntRate_dropped[['X4', 'X5', 'X6']] = IntRate_dropped[['X4', 'X5', 'X6']].replace('[\$,]', '', regex=True).astype(float)
	IntRate_dropped[['X1', 'X30']] = IntRate_dropped[['X1', 'X30']].replace('[,\%]', '', regex=True).astype(float)
	
	# Convert loan subgrades to dummy variables(ranks)
	IntRate_dropped['X9'] = IntRate_dropped['X9'].rank(method='dense')
	
	# Process missing values by median if continuous and by mode if categorical
	IntRate_dropped['X25'].fillna(0, inplace = True)
	IntRate_dropped['X26'].fillna(0, inplace = True)
	IntRate_dropped.fillna(IntRate_dropped.median()['X4':], inplace = True)
	Ctg_Ind_Miss = ['X7', 'X11', 'X12', 'X14', 'X17', 'X23', 'X32']
	IntRate_dropped[Ctg_Ind_Miss] = IntRate_dropped[Ctg_Ind_Miss].apply(lambda x: x.fillna(x.value_counts().index[0]))
	
	# Handle catagorical features by converting them to binary dummy variables
	Ctg_Ind = ['X7', 'X12', 'X14', 'X17', 'X32']
	IntRate_dummies = pd.get_dummies(IntRate_dropped[Ctg_Ind], drop_first = True)
	IntRate_NoMiss =  IntRate_dropped.join(IntRate_dummies)
	IntRate_NoMiss =  IntRate_NoMiss.drop(Ctg_Ind, axis = 1)
	
	# Transform some variables 
		# Variable X5 subtracted from X4
	IntRate_NoMiss['X5'] = IntRate_NoMiss['X4'] - IntRate_NoMiss['X5']
			
		# X15 categorized to 4 quarters and converted to binary dummy variables
	flist_1 = ['%b-%d','%d-%b']
	IntRate_NoMiss['X15'] = IntRate_NoMiss['X15'].apply(lambda x: DTFormatOpt(str(x), flist_1))
	IntRate_NoMiss['X15'] = IntRate_NoMiss['X15'].dt.quarter
	IssueDate_dummies = pd.get_dummies(IntRate_NoMiss['X15'], drop_first = True)
	IntRate_NoMiss =  IntRate_NoMiss.join(IssueDate_dummies)
	IntRate_NoMiss =  IntRate_NoMiss.drop(['X15'], axis = 1)
		
		# Variable X23 subtracted from most recent credit line which was opened among all borrowers to denote the relative 
		# duration of borrowers having credit lines 
	flist = ['%b-%y', '%d-%b']
	IntRate_NoMiss['X23'] = IntRate_NoMiss['X23'].apply(lambda x: DTFormatOpt(str(x), flist))
	IntRate_NoMiss['X23'] = IntRate_NoMiss['X23'].map(lambda dt: dt.replace(year=2001) if dt.year==1900 else dt.replace(year=dt.year))
	IntRate_NoMiss['X23'] = IntRate_NoMiss['X23'].map(lambda dt: dt.replace(year=dt.year-100) if dt.year>2020 else dt.replace(year=dt.year))
	Most_recent_date = IntRate_NoMiss['X23'].max()
	Days_creditLine = Most_recent_date - IntRate_NoMiss['X23']
	IntRate_NoMiss['X23'] = Days_creditLine.dt.days.astype(float)
	
	
		# Variable X11 is converted into floats:
			# Variable X11 for customers with work exp  < 1 year and missing 'employer of job title' is replaced by 0
			# Variable X11 for customers with work exp  < 1 year and not missing 'employer of job title' is replaced by 1
			# Variable X11 for customers with work exp 10+ years is replaced by 12 as average
	IntRate_NoMiss['X11'] = IntRate_NoMiss['X11'].replace('[,years]' or '[,year]', '', regex=True).replace( '10\+','12', regex=True)
	IntRate_NoMiss.ix[(IntRate_NoMiss['X10'].isnull())&(IntRate_NoMiss['X11'].str.contains('< 1').astype('bool')),'X11'] = '0'
	IntRate_NoMiss.ix[(IntRate_NoMiss['X10'].notnull())&(IntRate_NoMiss['X11'].str.contains('< 1').astype('bool')),'X11'] = '1'
	IntRate_NoMiss['X11'] = IntRate_NoMiss['X11'].convert_objects(convert_numeric=True)
	IntRate_Final =  IntRate_NoMiss.drop(['X10'], axis=1)
	IntRate_Final.fillna(IntRate_Final.median()['X4':], inplace = True)
	IntRate_Train_Final	= IntRate_Final.ix['x']
	IntRate_Test_Final	= IntRate_Final.ix['y']
	
	return (IntRate_Train_Final, IntRate_Test_Final)
	
	
