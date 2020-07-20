
import numpy as np
from functools import partial
import optuna
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import StandardScaler
from optimizer import optimize
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



if __name__ == "__main__":
	test = pd.read_csv("test.csv")
	test_ID = test['Id']
	latent_features = np.load("sample/latent.npy")
	y_train = np.load("sample/y_train.npy")
	train = latent_features[:1458] #n_train
	test = latent_features[1458:] 

	######################################
	#####  		  Optimizer 		######	
	######################################


	###################
	## Random Forest ##
	###################

	optimization_function = partial(optimize , x=train,y=y_train,regressor="random_forest")
	study = optuna.create_study(direction="minimize")
	study.optimize(optimization_function,n_trials=100)

	###################
	##  	SVM 	 ##
	###################

	optimization_function = partial(optimize , x=train,y=y_train,regressor="SVM")
	study = optuna.create_study(direction="minimize")
	study.optimize(optimization_function,n_trials=100)

#{'criterion': 'mse', 'n_estimators': 522, 'max_depth': 26, 'max_features': 'log2'}

	model_rf = ensemble.RandomForestRegressor(
		criterion="mse",
		n_estimators=522,
		max_depth=26,
		max_features='log2')


#{'kernel': 'rbf', 'gamma': 'auto', 'coef0': 24, 'degree': 1}
	model_svm = SVR(
		
		 coef0= 15,
		 degree= 3,
			gamma= 'scale',
			 kernel= 'poly')

	model_svm.fit(train,y_train)
	model_rf.fit(train,y_train)



	y_pred_svm = np.expm1(model_svm.predict(test))

	y_pred_rf = np.expm1(model_rf.predict(test))

	ensemble = 0.7*y_pred_rf + 0.3*y_pred_svm

	sub = pd.DataFrame()
	sub['Id'] = test_ID
	sub['SalePrice'] = ensemble
	sub.to_csv('sample/submission.csv',index=False)

	class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
		def __init__(self, models):
				self.models = models
				
		# we define clones of the original models to fit the data in
		def fit(self, X, y):
				self.models_ = [clone(x) for x in self.models]
				
				# Train cloned base models
				for model in self.models_:
						model.fit(X, y)

				return self
		
		#Now we do the predictions for cloned models and average them
		def predict(self, X):
				predictions = np.column_stack([
						model.predict(X) for model in self.models_
				])
				return np.mean(predictions, axis=1)

	stack = AveragingModels(models = (model_rf, model_svm, model_rf))
	stack.fit(train,y_train)
	stack_pred = stack.predict(test)

	sub = pd.DataFrame()
	sub['Id'] = test_ID
	sub['SalePrice'] = stack_pred
	sub.to_csv('sample/submission-stack.csv',index=False)
	print("Finish.")


