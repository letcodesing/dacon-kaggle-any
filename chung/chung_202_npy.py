import numpy as np
import pandas as pd
x = pd.read_csv('C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/dacon-kaggle-any/chung/train_input/CASE_01.csv')
y = pd.read_csv('C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/dacon-kaggle-any/chung/train_target/CASE_01.csv')
test = pd.read_csv('C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/dacon-kaggle-any/chung/test_input/TEST_01.csv')
submit = pd.read_csv('C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/dacon-kaggle-any/chung/sample_submission/TEST_01.csv')

from sklearn.model_selection import train_test_split

print(type(x))
x = x.fillna(0)
print(x.isnull().sum())
print(x.shape, y.shape)
path = 'C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/dacon-kaggle-any/chung/npy/'
np.save(path +'case01_x.npy', arr=x)
np.save(path +'case01_y.npy', arr=y)

