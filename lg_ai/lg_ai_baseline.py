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
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정
path = 'C:/Users/asthe/OneDrive - KNOU/beat/colab testin/dacon-kaggle-any/lg_ai/'
train_df = pd.read_csv(path + 'train.csv')
train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature
model = XGBRegressor()
model = SVR()
# model = LinearRegression()
LR = MultiOutputRegressor(model).fit(train_x, train_y)
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
submit.to_csv(path + 'submit.csv', index=False)