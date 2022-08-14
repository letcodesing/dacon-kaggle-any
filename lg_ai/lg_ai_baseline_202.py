import pandas as pd
import random
import os
import numpy as np

from numpy import mean, std, absolute
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor,RegressorChain
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.ensemble import GradientBoostingRegressor
Gradi = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.1, loss='ls', max_depth=8,
                          max_features=None, max_leaf_nodes=None,
                        #   min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=600,
                          n_iter_no_change=None, 
                        #   presort='deprecated',
                          random_state=42, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
model = XGBRegressor()
from sklearn.compose import TransformedTargetRegressor
# model = TransformedTargetRegressor(regressor=None, transformer=None, func=None, inverse_func=None, check_inverse=True)
parameters_xgb = [
    {'classifier__n_estimators' : [100, 200, 300, 400, 500] ,
    'classifier__learning_rate' : [0.1, 0.2, 0.3, 0.4, 0.5, 1, 0.01, 0.001],
    'classifier__max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100],
    'classifier__min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100],}]

pipe  = make_pipeline(RobustScaler(),Gradi)
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정
path = 'C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/dacon-kaggle-any/lg_ai/'
train_df = pd.read_csv(path + 'train.csv')
train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature
# model = XGBRegressor()
# model = SVR()
# model = LinearRegression()
LR = RegressorChain(pipe).fit(train_x, train_y)
print('Done.')
cv = RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
cross_score = cross_val_score(LR,train_x,train_y, scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
ab_score = absolute(cross_score)
print(ab_score)
test_x = pd.read_csv(path + 'test.csv').drop(columns=['ID'])
preds = LR.predict(test_x)
print('Done.')
submit = pd.read_csv(path + 'sample_submission.csv')
for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')
submit.to_csv(path + 'submit202.csv', index=False)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, train_size=0.8, random_state=63)
print('r2',r2_score(y_test, LR.predict(x_test)))
# 0.4117647058823529
# r2 0.19062544752999727 robust xgb``
