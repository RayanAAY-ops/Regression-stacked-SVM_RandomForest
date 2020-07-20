from sklearn import ensemble 
from sklearn import model_selection
from sklearn import metrics
from sklearn.svm import SVR
import numpy as np
import optuna

def optimize(trial,x,y,regressor):

  if (regressor=="random_forest"):

    criterion = trial.suggest_categorical("criterion", ["mse","mae"])
    n_estimators = trial.suggest_int("n_estimators",100,1000)
    max_depth = trial.suggest_int("max_depth",3,30)
    max_features = trial.suggest_categorical("max_features",["sqrt","auto","log2"])

    model = ensemble.RandomForestRegressor(
      criterion=criterion,
      n_estimators=n_estimators,
      max_depth=max_depth,
      max_features=max_features
  )
  elif (regressor=="SVM"): ##SVM
    kernel = trial.suggest_categorical("kernel", ["rbf","linear","poly"])
    gamma = trial.suggest_categorical("gamma", ["scale","auto"])
    coef0 = trial.suggest_int("coef0",1,50)
    degree = trial.suggest_int("degree",1,4)


    model = SVR(
      kernel=kernel,
      gamma=gamma,
      coef0=coef0,
      degree=degree
  )
  else:
   # Categorical parameter

  

  # Int parameter
    max_depth = trial.suggest_int("max_depth",3,30)

    n_estimators = trial.suggest_int("n_estimators",100,3000)

    max_leaves= trial.suggest_int("max_leaves",1,10)
  # Loguniform parameter
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.09)
  # Uniform parameter
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.0, 1.0) 

    gamma = trial.suggest_uniform('gamma', 0.0, 0.05)  

    min_child_weight = trial.suggest_uniform('min_child_weight',1,3)

    reg_lambda = trial.suggest_uniform('reg_lambda',0.5,1)
  

    model = xgb.XGBRegressor(
      objective ='reg:squarederror',
      
      n_estimators=n_estimators,
      max_depth=max_depth,
      learning_rate=learning_rate,
      colsample_bytree=colsample_bytree,
      gamma=gamma,
      min_child_weight=min_child_weight,
      reg_lambda=reg_lambda,
      max_leaves=max_leaves

  )

      

  kf=model_selection.KFold(n_splits=5)
  error=[]
  for idx in kf.split(X=x , y=y):
    train_idx , test_idx= idx[0],idx[1]
    xtrain=x[train_idx]
    ytrain=y[train_idx]
    xtest=x[test_idx]
    ytest=y[test_idx]   
    model.fit(x,y)
    y_pred = model.predict(xtest)
    fold_err = metrics.mean_squared_error(ytest,y_pred)
    error.append(fold_err)
  return 1.0 * np.mean(error)
