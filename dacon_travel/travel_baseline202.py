# 데이터를 불러오고 살펴보기 위한 pandas 라이브러리
import pandas as pd

# train 데이터 불러오기
train = pd.read_csv('C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/dacon-kaggle-any/dacon_travel/train.csv')

# test 데이터 불러오기
test = pd.read_csv('C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/dacon-kaggle-any/dacon_travel/test.csv')

# sample_submission 불러오기
sample_submission = pd.read_csv('C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/dacon-kaggle-any/dacon_travel/sample_submission.csv')

import matplotlib.pyplot as plt

# 이번엔 예측하고자 하는 값인 ProdTaken를 확인해봅니다.
# plt.hist(train.ProdTaken)
# plt.show()

# 결측치를 처리하는 함수를 작성합니다.

def handle_na(data):
    temp = data.copy()
    for col, dtype in temp.dtypes.items():
        if dtype == 'object':
            # 문자형 칼럼의 경우 'Unknown'을 채워줍니다.
            value = 'Unknown'
            temp.loc[:,col] = temp[col].fillna(value)
        elif dtype == int or dtype == float:
            # 수치형 칼럼의 경우 0을 채워줍니다.
            value = 0
            temp.loc[:,col] = temp[col].fillna(value)
    return temp

train_nona = handle_na(train)

# 결측치 처리가 잘 되었는지 확인해 줍니다.
train_nona.isna().sum()
object_columns = train_nona.columns[train_nona.dtypes == 'object']
print('object 칼럼은 다음과 같습니다 : ', list(object_columns))

# 해당 칼럼만 보아서 봅시다
train_nona[object_columns]
# LabelEncoder를 준비해줍니다.
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# LabelEcoder는 학습하는 과정을 필요로 합니다.
encoder.fit(train_nona['TypeofContact'])

#학습된 encoder를 사용하여 문자형 변수를 숫자로 변환해줍니다.
encoder.transform(train_nona['TypeofContact'])
train_enc = train_nona.copy()

# 모든 문자형 변수에 대해 encoder를 적용합니다.
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_enc[o_col])
    train_enc[o_col] = encoder.transform(train_enc[o_col])

# 결과를 확인합니다.
train_enc
# 결측치 처리
test = handle_na(test)

# 문자형 변수 전처리
for o_col in object_columns:
    encoder = LabelEncoder()
    
    # test 데이터를 이용해 encoder를 학습하는 것은 Data Leakage 입니다! 조심!
    encoder.fit(train_nona[o_col])
    
    # test 데이터는 오로지 transform 에서만 사용되어야 합니다.
    test[o_col] = encoder.transform(test[o_col])

# 결과를 확인합니다.
test
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(
    # alpha=0.9,
                                   ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.1, 
                        #   loss='ls', 
                          max_depth=3,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, 
                        #   min_impurity_split=None,
                          min_samples_leaf=1, 
                          min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=100,
                          n_iter_no_change=None, 
                        #   presort='deprecated',
                          random_state=42, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
# 모델 선언
# model = XGBClassifier()
# 분석할 의미가 없는 칼럼을 제거합니다.
train = train_enc.drop(columns=['id'])
test = test.drop(columns=['id'])

# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.
x_train = train.drop(columns=['ProdTaken'])
y_train = train[['ProdTaken']]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.98, random_state=98)
# 모델 학습
model.fit(x_train,y_train)
# 학습된 모델을 이용해 결과값 예측후 상위 10개의 값 확인
prediction = model.predict(test)
print('----------------------예측된 데이터의 상위 10개의 값 확인--------------------\n')
print(prediction[:10])

# 예측된 값을 정답파일과 병합
sample_submission['ProdTaken'] = prediction

# 정답파일 데이터프레임 확인
sample_submission.head()
# submission을 csv 파일로 저장합니다.
# index=False란 추가적인 id를 부여할 필요가 없다는 뜻입니다. 
# 정확한 채점을 위해 꼭 index=False를 넣어주세요.
sample_submission.to_csv('C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/dacon-kaggle-any/dacon_travel/submission_202.csv',index = False)

from sklearn.metrics import r2_score
print(r2_score(y_test,model.predict(x_test)))