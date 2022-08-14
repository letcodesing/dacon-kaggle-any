import time
import pandas as pd
import numpy as np
from xgboost import XGBClassifier,XGBRFClassifier
# import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV,KFold,train_test_split,HalvingGridSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/data/dacon_travel/train.csv', index_col=0)
test = pd.read_csv('C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/data/dacon_travel/test.csv', index_col=0)
submission = pd.read_csv('C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/data/dacon_travel/sample_submission.csv')

# print(train.columns)
# ['id', 'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch',
#        'Occupation', 'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',
#        'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus',
#        'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
#        'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome',
#        'ProdTaken'],
#       dtype='object'
# print(train.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1955 entries, 0 to 1954
# Data columns (total 20 columns):
#  #   Column                    Non-Null Count  Dtype
# ---  ------                    --------------  -----
#  0   id                        1955 non-null   int64
#  1   Age                       1861 non-null   float64
#  2   TypeofContact             1945 non-null   object
#  3   CityTier                  1955 non-null   int64
#  4   DurationOfPitch           1853 non-null   float64
#  5   Occupation                1955 non-null   object
#  6   Gender                    1955 non-null   object
#  7   NumberOfPersonVisiting    1955 non-null   int64
#  8   NumberOfFollowups         1942 non-null   float64
#  9   ProductPitched            1955 non-null   object
#  10  PreferredPropertyStar     1945 non-null   float64
#  11  MaritalStatus             1955 non-null   object
#  12  NumberOfTrips             1898 non-null   float64
#  13  Passport                  1955 non-null   int64
#  14  PitchSatisfactionScore    1955 non-null   int64
#  15  OwnCar                    1955 non-null   int64
#  16  NumberOfChildrenVisiting  1928 non-null   float64
#  17  Designation               1955 non-null   object
#  18  MonthlyIncome             1855 non-null   float64
#  19  ProdTaken                 1955 non-null   int64
# dtypes: float64(7), int64(7), object(6)
# print(train.isna().any()[lambda x:x])
# Age                         True
# TypeofContact               True
# DurationOfPitch             True
# NumberOfFollowups           True
# PreferredPropertyStar       True
# NumberOfTrips               True
# NumberOfChildrenVisiting    True
# MonthlyIncome               True
# print(test.isna().any()[lambda x:x])
# Age                         True
# TypeofContact               True
# DurationOfPitch             True
# NumberOfFollowups           True
# PreferredPropertyStar       True
# NumberOfTrips               True
# NumberOfChildrenVisiting    True
# MonthlyIncome               True
# print(train.isnull().sum())
# print(test.isnull().sum())
# Age                          94
# TypeofContact                10
# CityTier                      0
# DurationOfPitch             102
# Occupation                    0
# Gender                        0
# NumberOfPersonVisiting        0
# NumberOfFollowups            13
# ProductPitched                0
# PreferredPropertyStar        10
# MaritalStatus                 0
# NumberOfTrips                57
# Passport                      0
# PitchSatisfactionScore        0
# OwnCar                        0
# NumberOfChildrenVisiting     27
# Designation                   0
# MonthlyIncome               100
# ProdTaken                     0
# dtype: int64
# id                            0
# Age                         132
# TypeofContact                15
# CityTier                      0
# DurationOfPitch             149
# Occupation                    0
# Gender                        0
# NumberOfPersonVisiting        0
# NumberOfFollowups            32
# ProductPitched                0
# PreferredPropertyStar        16
# MaritalStatus                 0
# NumberOfTrips                83
# Passport                      0
# PitchSatisfactionScore        0
# OwnCar                        0
# NumberOfChildrenVisiting     39
# Designation                   0
# MonthlyIncome               133
train.interpolate(inplace=True)
test.interpolate(inplace=True)

def outlier(data):
    q1,q2,q3 = np.percentile(data,[25,50,75])
    print(q1,q2,q3)
    iqr = q3-q1
    print(iqr)
    lower_bound = q1-(iqr*1.5)
    upper_bound = q3+(iqr*1.5)
    return np.where((data>upper_bound)|(data<lower_bound))

#상관계수
print(type(train),type(test)) #pandas
# print(train.corr())
# print(test.corr())
# import seaborn as sns
# sns.heatmap(data=train.corr(),square=True,annot=True,cbar=True)

# print(train.iloc[:,-1].value_counts())

train.drop(['TypeofContact','Occupation','Designation','MaritalStatus','ProductPitched','Gender'],axis=1,inplace=True)
test.drop(['TypeofContact','Occupation','Designation','MaritalStatus','ProductPitched','Gender'], axis=1,inplace=True)
# train.drop(train.iloc[:,-5:-1],inplace=True,axis=1)
# test.drop(test.iloc[:,-5:-1],inplace=True,axis=1)
lda = LinearDiscriminantAnalysis()
out_loc = outlier(train)
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# lda = LinearDiscriminantAnalysis(n_components=1)


# lda.fit(train.iloc[:,:-1],train.iloc[:,-1])
# train.iloc[:,:-1] = lda.transform(train.iloc[:,:-1])

# train.fillna(0,inplace=True)
print('a')
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer,IterativeImputer
imp = SimpleImputer(strategy='constant')
# imp = KNNImputer(n_neighbors=5)
# imp = IterativeImputer(n_nearest_features=10)
# imp.fit_transform(train.iloc[:,:-1])
# imp.transform(test)
# lda.fit(train.iloc[:,:-1],train.iloc[:,-1])
# train.iloc[:,:-1] = lda.transform(train.iloc[:,:-1])
# test = lda.transform(test)
# train = np.where(train == np.nan,0,train)
# test = np.where(test == np.nan,0,test)
x_train, x_test, y_train, y_test = train_test_split(train.iloc[:,:-1],train.iloc[:,-1],random_state=829,stratify=train.iloc[:,-1],train_size=0.8)




xgb = XGBRFClassifier(random_state=12)
start = time.time()
xgb.fit(x_train,y_train)
end = time.time() - start
Parameters = [
    #   {'xgb__n_estimators':[100,200],'xgb__max_depth':[1,3,5,10]},
    #  {'xgb__max_depth':[6,8,12],'xgb__min_samples_leaf':[23,41,22]},
    #  {'xgb__min_samples_leaf':[3,5,7,10],'xgb__min_samples_split':[1,2,3,5]},
    #  {'xgb__min_samples_split':[2,3,5,10],'xgb__n_estimators':[400,20]},
     {'xgb__n_jobs':[-1],'xgb__n_estimators':[300,400,500],'xgb__min_samples_leaf':[6,1,15],'xgb__min_samples_split':[5,3,9]}
]
n_splits = 5
kfold = KFold(n_splits=n_splits,shuffle=True,random_state=2438)
pipe = Pipeline([('scaler',MaxAbsScaler()),('xgb',XGBRFClassifier())])
# (xgb, Parameters,cv=kfold, n_jobs=-1,refit=True, verbose=0,random_state=234,return_train_score=True,error_score=np.nan)
model = GridSearchCV(pipe,Parameters,n_jobs=-1,)
model.fit(x_train,y_train)
# pca_EVR = PCA.expalined_variance_ratio_
# print(np.cumsum(pca_EVR))
# plt.plot(np.cumsum(pca_EVR))
# plt.show()
# print(np.argmax(np.cumsum(pca_EVR)>0.95)+1)
# import matplotlib.pyplot as plt
# def plot_fi(model):
#     n_features = len(train.shape[1])
#     plt.barh(np.arange(n_features),model.feature_importances_,align='center')
#     plt.yticks(np.arange(n_features),train.fearue_names)
#     plt.xlabel('feture importans')
#     plt.ylabel('feature')
#     plt.ylim(-1,n_features)

# plot_fi(model)
# plt.show()
# from xgboost.plotting import plot_importance
# plot_importance(model)
# plt.show()

print('time',end)
print('score',model.score(x_test,y_test))
print('estimator',model.best_estimator_)
print('params',model.best_params_)
print('best score',model.best_score_)
y_pred = model.best_estimator_.predict(test)
print(x_train.shape,x_test.shape,test.shape)
print(out_loc)
y_pred = model.predict(test)
submission['ProdTaken'] = y_pred
submission.to_csv('C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/data/dacon_travel/submissionEnd2.csv', index=False)
# from sklearn.feature_selection import SelectFromModel
# from sklearn.metrics import accuracy_score
# thresholds = model.feature_importances_
# print(f'====================')
# for thresh in thresholds:
#     selection = SelectFromModel(model, threshold=thresh,prefit=True)
    
#     selec_x_train = selection.transform(x_train)
#     selec_x_test = selection.transform(x_test)
#     print(selec_x_train.shape, selec_x_test.shape)

#     model2 = XGBClassifier(n_jobs=-1,random_state=238,n_estimators=1000,
#                       learning_rate=0.1,
#                       max_depth=3,
#                       gemma=1,tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0)
    
#     model2.fit(selec_x_train,y_train)
#     y_pred = model2.predict(selec_x_test)
#     score = accuracy_score(y_test,y_pred)
#     print(thresh)
#     print("thresh=%.3f, n=%d, r2: %.2f%%"
#           % (thresh, selec_x_train.shape[1],score*100))
# x(1797, 64)