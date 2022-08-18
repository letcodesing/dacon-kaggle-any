
import pandas as pd
import numpy as np
import glob

path = 'https://we.tl/t-PXNbvaTt8m/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8

def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)
    print('끗.')

train_data, label_data = aaa(train_input_list, train_target_list) #, False)

print(train_data[0])
print(len(train_data), len(label_data)) # 1607 1607
print(len(train_data[0]))   # 1440
print(label_data)   # 1440
print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)




# train_data = train_data.reshape(-1,train_data.shape[1]*train_data.shape[2])
print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)

from keras.models import Model, load_model
from keras.layers import Input, Conv1D, Flatten,Reshape, Conv2D,LSTM
from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor()
in1 = Input(shape=(1440,37))
in1 = Conv1D(64,2)(in1)
in1 = Flatten()(in1)
d1 = Dense(1)(in1)
model = Model(inputs=in1,outputs=d1)
model.summary()
model.compile(loss='rmse',optimizer='adam')
model.fit(train_data,label_data,epochs=10)
test_input_list = sorted(glob.glob('C:/Users/aiapalm/OneDrive - KNOU/beat/colab testin/dacon-kaggle-any/chung/test_input/*.csv'))
test_target_list = sorted(glob.glob('C:/Users/aiapalm/OneDrive - KNOU/beat/colab testin/dacon-kaggle-any/chung/test_target/*.csv'))

test, submit = aaa(test_input_list, test_target_list) #, False)
pred = model.predict(test)
print(submit)
print(pred)