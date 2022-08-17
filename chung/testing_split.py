import numpy as np
import pandas as pd
import tensorflow as tf            
tf.random.set_seed(1987)    

data = np.load('c:/project/개인1/npy, weight 저장/2/data.npy')
# print(data)
print(data.shape)
print(len(data))

def split_x(dataset, size):
    aaa = []
    for i in range(29):
        subset = dataset[i:i+size]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)


x = split_x(data,1441)
print(x)

econo_x = x[:-5,:,:3]
democ_x = x[:-5,:,3:6]
econo_x_pred = x[-5:,:,:3]
democ_x_pred = x[-5:,:,3:6]
print(econo_x)
print('============================')
print(democ_x)

def split_y(dataset, size):
    aaa = []
    for i in range(len(dataset)-9):
        subset = dataset[i+5:i+5+size,7:]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

y = split_y(data,5)
print(y)
print(y.shape)
print(data.shape)
print(len(data))
president_y  = y[:,:,:1]
congressmember_y  = y[:,:,1:5]
print(president_y)
print(congressmember_y)
print(president_y.shape, congressmember_y.shape)

from sklearn.model_selection import train_test_split
econo_x_train, econo_x_test, democ_x_train, democ_x_test, president_y_train, president_y_test, congressmember_y_train, congressmember_y_test = train_test_split(econo_x, democ_x,president_y,congressmember_y, train_size=0.8, random_state=187)


from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,RobustScaler,StandardScaler,Normalizer
Minscaler = MinMaxScaler()
Maxscaler = MaxAbsScaler()
Stanscaler = StandardScaler()
Roscaler = RobustScaler()
Norscaler = Normalizer()

econo_x_train = econo_x_train.reshape(-1,15)
econo_x_test = econo_x_test.reshape(-1,15)
econo_x_pred = econo_x_pred.reshape(-1,15)

democ_x_train = democ_x_train.reshape(-1,15)
democ_x_test = democ_x_test.reshape(-1,15)
democ_x_pred = democ_x_pred.reshape(-1,15)

econo_x_train[:,0:1] = Roscaler.fit_transform(econo_x_train[:,0:1])
econo_x_test[:,0:1] = Roscaler.transform(econo_x_test[:,0:1])
econo_x_pred[:,0:1] = Roscaler.transform(econo_x_pred[:,0:1])

econo_x_train[:,1:3] = Stanscaler.fit_transform(econo_x_train[:,1:3])
econo_x_test[:,1:3] = Stanscaler.transform(econo_x_test[:,1:3])
econo_x_pred[:,1:3] = Stanscaler.transform(econo_x_pred[:,1:3])




democ_x_train[:,0:1] = Maxscaler.fit_transform(democ_x_train[:,0:1])
democ_x_test[:,0:1] = Maxscaler.transform(democ_x_test[:,0:1])
democ_x_pred[:,0:1] = Maxscaler.transform(democ_x_pred[:,0:1])

democ_x_train[:,1:3] = Minscaler.fit_transform(democ_x_train[:,1:3])
democ_x_test[:,1:3] = Minscaler.transform(democ_x_test[:,1:3])
democ_x_pred[:,1:3] = Minscaler.transform(democ_x_pred[:,1:3])
econo_x_train = econo_x_train.reshape(-1,5,3)
econo_x_test = econo_x_test.reshape(-1,5,3)
econo_x_pred = econo_x_pred.reshape(-1,5,3)

democ_x_train = democ_x_train.reshape(-1,5,3)
democ_x_test = democ_x_test.reshape(-1,5,3)
democ_x_pred = democ_x_pred.reshape(-1,5,3)

# print(president_y)
# print(president_y.shape)
# print(congressmember_y)
# print(congressmember_y.shape)

# from sklearn.model_selection import train_test_split

    
print(econo_x_train.shape, democ_x_train.shape, econo_x_test.shape, democ_x_test.shape)
print(president_y_train.shape, president_y_test.shape, congressmember_y_train.shape, congressmember_y_test.shape)
import torch
from torch import nn
import torch.nn.functional as F
econo_x_train = torch.tensor(econo_x_train)
econo_x_test = torch.tensor(econo_x_test)
econo_x_pred = torch.tensor(econo_x_pred)
democ_x_train = torch.tensor(democ_x_train)
democ_x_test = torch.tensor(democ_x_test)
democ_x_pred = torch.tensor(democ_x_pred)
econo_x_train = F.interpolate(econo_x_train, scale_factor=50, mode='linear')
econo_x_test = F.interpolate(econo_x_test, scale_factor=50, mode='linear')
econo_x_pred = F.interpolate(econo_x_pred, scale_factor=50, mode='linear')
democ_x_train = F.interpolate(democ_x_train, scale_factor=50, mode='linear')
democ_x_test = F.interpolate(democ_x_test, scale_factor=50, mode='linear')
democ_x_pred = F.interpolate(democ_x_pred, scale_factor=50, mode='linear')
econo_x_train = econo_x_train.numpy()
econo_x_test = econo_x_test.numpy()
econo_x_pred = econo_x_pred.numpy()
democ_x_train = democ_x_train.numpy()
democ_x_test = democ_x_test.numpy()
democ_x_pred = democ_x_pred.numpy()
# print(econo_x_r2.shape, congressmember_y_r2.shape)
# print(econo_x_predic.shape, congressmember_y_predic.shape)
#프레지던트 y 시그모이드, 콩그래스멤버 리니어
#앙상블모델 구성
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv1D, Conv2D, Flatten, Reshape,ReLU, LSTM, GRU, concatenate,Dropout,MaxPooling1D,MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=150,mode='auto',restore_best_weights=True,verbose=1)
mc = ModelCheckpoint('bestmodel.hdf5',monitor='val_loss',mode='min',save_best_only=True)
#모델1
econo = Input(shape=(5,150))
econo = Dense(200,activation='relu')(econo)
econo = Dense(15)(econo)
econo = Reshape((75,1))(econo)
econo = ReLU(2800,5)(econo)
econo = LSTM(64,activation='relu')(econo)
# econo = Flatten()(econo)
econo = Dense(200,activation='relu')(econo)
econo = Dense(200,activation='relu')(econo)
econo = Dense(1000,activation='relu')(econo)
econo = Dense(750)(econo)
econo = Reshape((5,150))(econo)
# econo  = Flatten()(econo)
# econo  = Dense(200)(econo)
# econo = Reshape((50,4))(econo)
# econo =  ReLU(8,10)(econo)

#모델2
democ = Input(shape=(5,150))
democ = Dense(200,activation='relu')(democ)
democ = Dense(200,activation='relu')(democ)
democ = Dense(200,activation='relu')(democ)
democ = Dense(15)(democ)
democ = Reshape((75,1))(democ)
democ = ReLU(28,10)(democ)
democ = LSTM(64,return_sequences=True)(democ)
democ = Flatten()(democ)
democ = Dense(1000, activation='relu')(democ)
democ = Dense(750)(democ)
democ = Reshape((5,150))(democ)

# democ  = Flatten()(democ)
# democ  = Dense(200)(democ)
# democ = Reshape((50,4))(democ)
# democ =  ReLU(8,10)(democ)


#merge
president = concatenate((econo,democ))
president = LSTM(80, return_sequences=True)(president)
president = LSTM(80)(president)
# president = Flatten()(president)
president = Dense(200,activation='tanh')(president)
president = Dense(200,activation='tanh')(president)
president = Dense(200,activation='tanh')(president)
president = Dropout(0.2)(president)
president = Dense(200,activation='elu')(president)
president = Dense(100)(president)
president = Reshape((5,20))(president)
president = LSTM(1,return_sequences=True,activation='sigmoid')(president)

congress = concatenate((econo,democ))
congress = LSTM(80)(congress)
congress = Flatten()(congress)
congress = Dense(800,activation='relu')(congress)
congress = Dense(800,activation='relu')(congress)
congress = Dense(800,activation='relu')(congress)
congress = Dropout(0.4)(congress)
congress = Dense(80)(congress)
congress = Reshape((5,16))(congress)
congress = LSTM(4,activation='relu',return_sequences=True)(congress)

model = Model(inputs=[econo,democ], outputs=[president,congress],)
model.summary()

model.load_weights('c:/project/개인1/npy, weight 저장/5/weight.h5')

model.compile(loss=['binary_crossentropy','mae'], optimizer='AdaMax')
# hist = model.fit([econo_x_train,democ_x_train],[president_y_train,congressmember_y_train],epochs=30000,batch_size=10, 
#                  validation_split=0.1,
#                  callbacks=[es,mc])

# model.save_weights('c:/project/개인1/npy, weight 저장/5/weight.h5')


loss = model.evaluate([econo_x_test,democ_x_test],[president_y_test,congressmember_y_test])
# pred_presi, pred_congress = model.predict([econo_x_test, democ_x_test])
print('loss',loss)
pred_presi, pred_congress = model.predict([econo_x_pred,democ_x_pred])

print(pred_presi)
print('=============')
print(pred_congress)
from sklearn.metrics import accuracy_score, r2_score
presi_acc, congress_r2 = model.predict([econo_x_test,democ_x_test])
print(congress_r2)
# presi_acc = np.where(presi_acc<7,0,1).astype(int)
# presi_acc = np.argmax(presi_acc,axis=1)
print(presi_acc)
# presi_acc[(presi_acc<7)] = 0 
# presi_acc[(presi_acc>=7)] = 1 

presi_acc = np.array(presi_acc)
print(type(presi_acc),type(president_y_test))
presi_acc = presi_acc.round()
presi_acc = presi_acc.reshape(-1,5)
president_y_test = president_y_test.reshape(-1,5)
# presi_acc = pd.DataFrame(presi_acc, dtype='int64')
print(presi_acc)
print(president_y_test)
# def get_accuracy(y_true, y_prob):
#     accuracy = metrics.accuracy_score(y_true, y_prob > 0.5)
#     return accuracy
# def hamming_score(y_true, y_pred):
#     return (
#         (y_true & y_pred).sum(axis=1) / (y_true or y_pred).sum(axis=1)
#     ).mean()
# print(pres)
presi_acc_score = accuracy_score(president_y_test,presi_acc)

congress_r2 = congress_r2.reshape(-1,5)
congressmember_y_test = congressmember_y_test.reshape(-1,5)
congress_r2_score = r2_score(congressmember_y_test,congress_r2)
print(congress_r2)
print(congressmember_y_test)
# # print(pred0[0])
# # print(president_y_r2)
# # pred_presi = np.where(pred0[0]>0.5,1,0)
# print(president_y_test.shape)
# print(pred_presi.shape)
# pred_presi = pred_presi.reshape(10,5,1)
# pred_presi = np.array(pred_presi)
# # # print(pred0[1])
# # print(congressmember_y_test)
# # pred_congress = pred0[1]
# # print(pred_congress.shape, congressmember_y_test.shape)
# print(congressmember_y_test.shape)
# print(pred_congress.shape)
# congress_r2_score = r2_score(congressmember_y_test,pred_congress)
# print(f'r2',congress_r2_score)
# # print(pred.shape)
# # print(presi_r2_score)
# # result = round(pred)
# # print(np.round(pred,))
# print('loss:',loss)
# pred = model.predict([econo_x_predic, democ_x_predic])
# print(pred)

print('acc_score', presi_acc_score)
print('r2_score', congress_r2_score)

print(np.where(pred_presi[0][-1]>=0.5,'27년 대선에서는 야당후보가 당선됩니다','27년 대선에서는 여당후보가 당선됩니다'))
# print(pred[1][-4].round())
total_congress = pred_congress[1][-4][0].round()+pred_congress[1][-4][1].round()+pred_congress[1][-4][2].round()+pred_congress[1][-4][3].round()
print('24년 총선에서 민주당계는',(pred_congress[1][-4][0].round()).astype(int),'명,','보수당계는',(pred_congress[1][-4][1].round()).astype(int),'명,','진보당계',(pred_congress[1][-4][2].round()).astype(int),'명,','무소속', (pred_congress[1][-4][3].round()).astype(int),'명',
      '기타 21명','합계', (total_congress).astype(int) 
      +21,'명'      )




# print('합계', (total_congress).astype(int),'명', '기타21명')

# print(pred.shape)

print()


# import matplotlib.pyplot  as plt
# plt.figure(figsize=(10,10))
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# # plt.legend()
# # plt.scatter()
# plt.show()

