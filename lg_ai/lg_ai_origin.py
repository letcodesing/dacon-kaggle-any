import imp
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.1, loss='ls', max_depth=6,
                          max_features=None, max_leaf_nodes=None,
                        #   min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=300,
                          n_iter_no_change=None, 
                        #   presort='deprecated',
                          random_state=42, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
from sklearn.compose import TransformedTargetRegressor
# model = TransformedTargetRegressor(regressor=None, transformer=None, func=None, inverse_func=None, check_inverse=True)
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정
path = 'C:/Users/asthe/OneDrive - KNOU/beat/colab testin/dacon-kaggle-any/lg_ai/'
train_df = pd.read_csv(path + 'train.csv')
train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature
print(train_y.iloc[:,-11].shape)
print(train_y.iloc[:,-11].value_counts())

# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-1])
# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-2])
# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-3])
# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-4])
# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-5])
# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-6])
# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-7])
# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-8])
# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-9])
# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-10])
# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-11])
# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-12])
# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-13])
# LinearDiscriminantAnalysis.fit(train_x,train_y.iloc[:,-14])
# train_x = LinearDiscriminantAnalysis.transform(train_x)
# model = XGBRegressor()
# model = SVR()
# model = LinearRegression()
from sklearn.neighbors import KNeighborsRegressor,KNeighborsTransformer
model = KNeighborsRegressor()
LR = RegressorChain(model).fit(train_x, train_y)
print('Done.')
cv = RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
cross_score = cross_val_score(LR,train_x,train_y, scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
ab_score = absolute(cross_score)
print(ab_score)
test_x = pd.read_csv(path + 'test.csv').drop(columns=['ID'])
# test_x = LinearDiscriminantAnalysis.transform(test_x)
preds = LR.predict(test_x)
print('Done.')
submit = pd.read_csv(path + 'sample_submission.csv')
for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')
submit.to_csv(path + 'submit_hm.csv', index=False)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, train_size=0.8, random_state=63)
print('r2',r2_score(y_test, LR.predict(x_test)))
from sklearn.metrics import mean_squared_error
def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(1,15): # ignore 'ID'
        rmse = mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
    return score
print('nrmse',lg_nrmse(y_test,LR.predict(x_test)))

# r2 0.03479853336851048 gradi 200 4

# 0.2438792149604076