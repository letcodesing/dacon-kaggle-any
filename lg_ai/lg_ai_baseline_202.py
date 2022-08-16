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
                          min_weight_fraction_leaf=0.0, n_estimators=300,
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
from sklearn.decomposition import PCA
pca = PCA(n_components=32)
pca.fit(train_x)
train_x = pca.transform(train_x)
import tensorflow 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
# train_x = np.array(train_x)
# train_x.reshape(-1,14,4,1)
# print(train_x.shape, train_y.shape)
# model = Sequential([
#     Conv2D(16,(3,3),activation='relu',input_shape=(14,4,1)),
#     MaxPooling2D(2,2),
#     Conv2D(32,(3,3),activation='relu'),
#     MaxPooling2D(2,2),
#     Conv2D(64,(3,3),activation='relu'),
#     MaxPooling2D(2,2),
#     Conv2D(128,(3,3),activation='relu'),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dropout(0.5),
#     Dense(128,activation='relu'),
#     Dense(16,activation='relu'),
#     Dense(1,activation='linear')
# ])
pipe  = make_pipeline(StandardScaler(),Gradi)
LR = MultiOutputRegressor(pipe).fit(train_x, train_y)
print('Done.')
cv = RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
cross_score = cross_val_score(LR,train_x,train_y, scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
ab_score = absolute(cross_score)
print(ab_score)
test_x = pd.read_csv(path + 'test.csv').drop(columns=['ID'])
test_x = pca.transform(test_x)
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
print('nrmse',lg_nrmse(y_test,np.array(LR.predict(x_test))))
# 0.4117647058823529
# r2 0.19062544752999727 robust xgb``
# r2 0.6209023572885323 Gradi 600 8
# r2 0.9697908391765253 gradi 250 12 MultiOutputRegressor robust 