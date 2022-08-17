import numpy as np
import pandas as pd
path = 'C:/Users/aia200/OneDrive - 한국방송통신대학교/beat/colab testin/dacon-kaggle-any/chung/npy/'

x = np.load(path+'case01_x.npy',allow_pickle=True)
y = np.load(path+'case01_y.npy',allow_pickle=True)

print(x.shape, y.shape)
from keras.models import Sequential
from keras.layers import Dense,Dropout

# x = x.fillna(0)
def split_x(dataset, size):
    aaa = []
    for i in range(29):
        subset = dataset[i*size:i*size+size+2]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

x = split_x(x,1439)
print(x.shape)
print(x[0])
print(x[1])
print(x[2])
print(x[3])
print(x[4])
print(x[5])
print(x[6])
print(x[7])
# 1~1442 1443~2884 2884~4002






model = Sequential()
model.add(Dense(20))
