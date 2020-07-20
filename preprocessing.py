import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats as st
from scipy.stats import norm, skew
from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.preprocessing import StandardScaler ,LabelEncoder

if __name__ == "__main__":
	train  = pd.read_csv("train.csv")
	test = pd.read_csv("test.csv")
	train_ID = train['Id']
	test_ID = test['Id']

	train.drop("Id", axis = 1, inplace = True)
	test.drop("Id", axis = 1, inplace = True)


	train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

	## Log transform the target ,because its distribution is skewed
	train["SalePrice"] = np.log1p(train["SalePrice"])

	ntrain = train.shape[0]
	ntest = test.shape[0]
	y_train = train.SalePrice.values
	all_data = pd.concat((train, test)).reset_index(drop=True)
	all_data.drop(['SalePrice'], axis=1, inplace=True)


	############################
	#Dealing with missing data#
	############################

	all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
	all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
	all_data["Alley"] = all_data["Alley"].fillna("None")
	all_data["Fence"] = all_data["Fence"].fillna("None")
	all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
	all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
		lambda x: x.fillna(x.median()))
	for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
			all_data[col] = all_data[col].fillna('None')
	for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
			all_data[col] = all_data[col].fillna(0)
	for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
			all_data[col] = all_data[col].fillna(0)
	for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
			all_data[col] = all_data[col].fillna('None')
	all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
	all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

	all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
	all_data = all_data.drop(['Utilities'], axis=1)
	all_data["Functional"] = all_data["Functional"].fillna("Typ")
	all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

	all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

	all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
	all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

	all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
	#MSSubClass=The building class
	all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
	all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
	all_data['YrSold'] = all_data['YrSold'].astype(str)
	all_data['MoSold'] = all_data['MoSold'].astype(str)


	cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
				'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
				'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
				'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
				'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
	for c in cols:
			lbl = LabelEncoder() 
			lbl.fit(list(all_data[c].values)) 
			all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
	print('Shape all_data: {}'.format(all_data.shape))

	all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

	all_data = pd.get_dummies(all_data)
	print(all_data.shape)

	train = all_data[:ntrain]
	test = all_data[ntrain:]

	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

	newdf = all_data.select_dtypes(include=numerics)

	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

	newdf = all_data.select_dtypes(include=numerics)
	sc = StandardScaler()
	all_data_encoded = sc.fit_transform(newdf)


	np.save("sample/data.npy",all_data_encoded)
	np.save("sample/y_train.npy",y_train)
