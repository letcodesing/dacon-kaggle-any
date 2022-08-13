import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
#파일 불러오기
path = 'C:/Users/aiapalm/OneDrive - KNOU/beat/colab testin/data/lg_ai'
train = pd.read_csv(path + '/train.csv',index_col=0)
test = pd.read_csv(path +'/test.csv',index_col=0)
submission = pd.read_csv(path +'/sample_submission.csv',index_col=0)

print(submission)
#결측치 확인
# print(train.isnull().sum())
scaler = StandardScaler()
# print(type(train)) 판다스
train.iloc[:,:-14] = scaler.fit_transform(train.iloc[:,:-14])
test = scaler.transform(test)
import joblib
path = 'C:/Users/aiapalm/OneDrive - KNOU/beat/study/_save/_xg/'
x_train, x_test, y_train, y_test = train_test_split(train.iloc[:,:-14],train.iloc[:,-14:],random_state=239784,train_size=0.8)
# xgb = joblib.load(path+'m39_pickle1_save.dat')
xgb = RandomForestRegressor(random_state=142)

xgb.fit(x_train,y_train)
print(r2_score(y_test, xgb.predict(x_test)))
print(xgb.predict(test))
print(submission)
submission = xgb.predict(test)
# submission = pd.DataFrame()
print(type(submission))
np.savetxt(path+"submission.csv", submission, delimiter=",")
# submission.to_csv(path+'/submission.csv',index=False)
import joblib
import joblib
path = 'C:/Users/aiapalm/OneDrive - KNOU/beat/colab testin/data/lg_ai'
joblib.dump(xgb,(path+'lg_ai.dat'))
# from sklearn.feature_selection import SelectFromModel
# from sklearn.metrics import accuracy_score
# thresholds = xgb.feature_importances_
# print(f'====================')
# for thresh in thresholds:
#     selection = SelectFromModel(xgb, threshold=thresh,prefit=True)
    
#     selec_x_train = selection.transform(x_train)
#     selec_x_test = selection.transform(x_test)
#     print(selec_x_train.shape, selec_x_test.shape)

#     model2 = XGBRegressor(n_jobs=-1,random_state=238,n_estimators=1000,
#                       learning_rate=0.1,
#                       max_depth=3,
#                       gemma=1,tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0)
    
#     model2.fit(selec_x_train,y_train)
#     y_pred = model2.predict(selec_x_test)
#     score = r2_score(y_test,y_pred)
#     print(thresh)
#     print("thresh=%.3f, n=%d, r2: %.2f%%"
#           % (thresh, selec_x_train.shape[1],score*100))